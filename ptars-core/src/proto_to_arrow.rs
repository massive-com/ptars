use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer};
use arrow_array::builder::ArrayBuilder;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, LargeBinaryBuilder, LargeStringBuilder, NullBufferBuilder,
    PrimitiveBuilder, StringBuilder,
};
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrowPrimitiveType, BinaryArray, LargeBinaryArray, LargeStringArray, MapArray,
    RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, TimeUnit};
use chrono::Datelike;

use prost_reflect::{EnumDescriptor, FieldDescriptor, Kind, MessageDescriptor};

use crate::config::{ConfluentWirePolicy, EnumRepr, PtarsConfig};

// ---------------------------------------------------------------------------
// Wire format decoding primitives
// ---------------------------------------------------------------------------

#[allow(deprecated)]
fn decode_error(msg: impl Into<Cow<'static, str>>) -> prost::DecodeError {
    prost::DecodeError::new(msg)
}

#[inline(always)]
fn decode_varint(buf: &[u8]) -> Result<(u64, usize), prost::DecodeError> {
    match buf.first() {
        // one byte is very common; most field tags, and small values
        Some(&byte) if byte < 0x80 => Ok((byte as u64, 1)),
        _ => decode_varint_chunk(buf),
    }
}

/// Packs the low 7 bits of each byte of `x` into the low 56 bits.
#[inline(always)]
fn pack_varint_groups(x: u64) -> u64 {
    let x = x & 0x7f7f_7f7f_7f7f_7f7f;
    let x = (x & 0x007f_007f_007f_007f) | ((x & 0x7f00_7f00_7f00_7f00) >> 1);
    let x = (x & 0x0000_3fff_0000_3fff) | ((x & 0x3fff_0000_3fff_0000) >> 2);
    (x & 0x0000_0000_0fff_ffff) | ((x & 0x0fff_ffff_0000_0000) >> 4)
}

/// Tries to decode the first 8 bytes without a branching, looping otherwise.
#[cold]
fn decode_varint_chunk(buf: &[u8]) -> Result<(u64, usize), prost::DecodeError> {
    let Some((word, rest)) = buf.split_first_chunk() else {
        return decode_varint_loop(buf);
    };
    let word = u64::from_le_bytes(*word);
    let stops = !word & 0x8080_8080_8080_8080;
    if stops != 0 {
        let len = (stops.trailing_zeros() / 8 + 1) as usize;
        let keep = u64::MAX >> (64 - 8 * len);
        return Ok((pack_varint_groups(word & keep), len));
    }
    let low = pack_varint_groups(word);
    match *rest {
        [b8, ..] if b8 < 0x80 => Ok((low | (b8 as u64) << 56, 9)),
        [b8, b9, ..] if b9 < 0x80 => Ok((low | ((b8 & 0x7f) as u64) << 56 | (b9 as u64) << 63, 10)),
        [_, _, ..] => Err(decode_error("varint too large")),
        _ => Err(decode_error("unexpected EOF in varint")),
    }
}

/// Byte at a time, for varints with fewer than 8 bytes left in the buffer.
fn decode_varint_loop(buf: &[u8]) -> Result<(u64, usize), prost::DecodeError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in buf.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(decode_error("varint too large"));
        }
    }
    Err(decode_error("unexpected EOF in varint"))
}

/// Validates `data` as UTF-8, checking for ASCII first: for short strings
/// that is much cheaper than the out-of-line call to `str::from_utf8`.
fn validate_utf8(data: &[u8]) -> Result<&str, prost::DecodeError> {
    if data.is_ascii() {
        // SAFETY: ASCII is valid UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked(data) })
    } else {
        std::str::from_utf8(data).map_err(|_| decode_error("invalid UTF-8"))
    }
}

fn decode_tag(buf: &[u8]) -> Result<(u32, u8, usize), prost::DecodeError> {
    let (key, n) = decode_varint(buf)?;
    let wire_type = (key & 0x07) as u8;
    let field_number = (key >> 3) as u32;
    if field_number == 0 {
        return Err(decode_error("invalid field number 0"));
    }
    Ok((field_number, wire_type, n))
}

fn skip_field(wire_type: u8, buf: &[u8]) -> Result<usize, prost::DecodeError> {
    match wire_type {
        0 => {
            let (_, n) = decode_varint(buf)?;
            Ok(n)
        }
        1 => {
            if buf.len() < 8 {
                return Err(decode_error("unexpected EOF"));
            }
            Ok(8)
        }
        2 => {
            let (len, n) = decode_varint(buf)?;
            let total = n + len as usize;
            if buf.len() < total {
                return Err(decode_error("unexpected EOF"));
            }
            Ok(total)
        }
        5 => {
            if buf.len() < 4 {
                return Err(decode_error("unexpected EOF"));
            }
            Ok(4)
        }
        _ => Err(decode_error("unsupported wire type")),
    }
}

/// Read a length-delimited field, returning (data_slice, bytes_consumed).
fn read_length_delimited(buf: &[u8]) -> Result<(&[u8], usize), prost::DecodeError> {
    let (len, n) = decode_varint(buf)?;
    let len = len as usize;
    let total = n + len;
    if buf.len() < total {
        return Err(decode_error("unexpected EOF"));
    }
    Ok((&buf[n..total], total))
}

/// Strip the Confluent Schema Registry wire format prefix from a message.
fn strip_confluent_prefix(
    buf: &[u8],
    policy: ConfluentWirePolicy,
) -> Result<&[u8], prost::DecodeError> {
    match policy {
        ConfluentWirePolicy::Raw => Ok(buf),
        ConfluentWirePolicy::Standard => {
            if buf.len() < 5 {
                return Err(decode_error(
                    "message too short for Confluent wire format header",
                ));
            }
            Ok(&buf[5..])
        }
        ConfluentWirePolicy::Protobuf => {
            if buf.len() < 5 {
                return Err(decode_error(
                    "message too short for Confluent wire format header",
                ));
            }
            let remaining = &buf[5..];
            // Read varint-encoded count of message indexes
            let (count, mut offset) = decode_varint(remaining)?;
            // Skip `count` varints (the message indexes themselves)
            for _ in 0..count {
                let (_, n) = decode_varint(&remaining[offset..])?;
                offset += n;
            }
            Ok(&remaining[offset..])
        }
    }
}

#[inline]
fn decode_zigzag32(v: u64) -> i32 {
    let v = v as u32;
    ((v >> 1) as i32) ^ (-((v & 1) as i32))
}

#[inline]
fn decode_zigzag64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

fn convert_seconds_nanos_to_unit(seconds: i64, nanos: i32, unit: TimeUnit, type_name: &str) -> i64 {
    match unit {
        TimeUnit::Second => seconds,
        TimeUnit::Millisecond => seconds
            .checked_mul(1_000)
            .and_then(|s| s.checked_add(i64::from(nanos) / 1_000_000))
            .unwrap_or_else(|| panic!("{type_name} overflow")),
        TimeUnit::Microsecond => seconds
            .checked_mul(1_000_000)
            .and_then(|s| s.checked_add(i64::from(nanos) / 1_000))
            .unwrap_or_else(|| panic!("{type_name} overflow")),
        TimeUnit::Nanosecond => seconds
            .checked_mul(1_000_000_000)
            .and_then(|s| s.checked_add(i64::from(nanos)))
            .unwrap_or_else(|| panic!("{type_name} overflow")),
    }
}

static CE_OFFSET: i32 = 719163;

fn enum_name(enum_descriptor: &EnumDescriptor, number: i32) -> String {
    match enum_descriptor.get_value(number) {
        Some(v) => v.name().to_string(),
        None => number.to_string(),
    }
}

// ---------------------------------------------------------------------------
// String/Binary builder inner enums
// ---------------------------------------------------------------------------

enum StringBuilderInner {
    Regular(StringBuilder),
    Large(LargeStringBuilder),
}

impl StringBuilderInner {
    fn new(use_large: bool) -> Self {
        if use_large {
            Self::Large(LargeStringBuilder::new())
        } else {
            Self::Regular(StringBuilder::new())
        }
    }
    fn append_value(&mut self, v: &str) {
        match self {
            Self::Regular(b) => b.append_value(v),
            Self::Large(b) => b.append_value(v),
        }
    }
    fn append_null(&mut self) {
        match self {
            Self::Regular(b) => b.append_null(),
            Self::Large(b) => b.append_null(),
        }
    }
    fn finish(&mut self) -> Arc<dyn Array> {
        match self {
            Self::Regular(b) => Arc::new(b.finish()),
            Self::Large(b) => Arc::new(b.finish()),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Regular(b) => ArrayBuilder::len(b),
            Self::Large(b) => ArrayBuilder::len(b),
        }
    }
}

enum BinaryBuilderInner {
    Regular(BinaryBuilder),
    Large(LargeBinaryBuilder),
}

impl BinaryBuilderInner {
    fn new(use_large: bool) -> Self {
        if use_large {
            Self::Large(LargeBinaryBuilder::new())
        } else {
            Self::Regular(BinaryBuilder::new())
        }
    }
    fn append_value(&mut self, v: &[u8]) {
        match self {
            Self::Regular(b) => b.append_value(v),
            Self::Large(b) => b.append_value(v),
        }
    }
    fn append_null(&mut self) {
        match self {
            Self::Regular(b) => b.append_null(),
            Self::Large(b) => b.append_null(),
        }
    }
    fn finish(&mut self) -> Arc<dyn Array> {
        match self {
            Self::Regular(b) => Arc::new(b.finish()),
            Self::Large(b) => Arc::new(b.finish()),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Regular(b) => ArrayBuilder::len(b),
            Self::Large(b) => ArrayBuilder::len(b),
        }
    }
}

/// A binary column of a singular field, built one row at a time. Unlike
/// arrow's builders, the value appended last can be replaced, for a field
/// that occurs more than once in a message (last one wins). Strings use
/// [`StringColumn`].
struct ByteColumn {
    /// Starts with 0; one end offset per row. `i64` until [`Self::finish`].
    offsets: Vec<i64>,
    values: Vec<u8>,
    nulls: NullBufferBuilder,
    large: bool,
}

impl ByteColumn {
    fn new(large: bool) -> Self {
        Self::with_capacity(large, 0)
    }

    /// Room for `rows` rows; the value bytes still grow as needed.
    fn with_capacity(large: bool, rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        Self {
            offsets,
            values: Vec::new(),
            nulls: NullBufferBuilder::new(rows),
            large,
        }
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline(always)]
    fn push(&mut self, value: &[u8]) {
        self.values.extend_from_slice(value);
        self.offsets.push(self.values.len() as i64);
        self.nulls.append_non_null();
    }

    fn push_null(&mut self) {
        self.offsets.push(self.values.len() as i64);
        self.nulls.append_null();
    }

    #[inline]
    fn push_or_replace(&mut self, value: &[u8], has_value: &mut bool) {
        if *has_value {
            self.replace_cold(value);
        } else {
            self.push(value);
            *has_value = true;
        }
    }

    #[cold]
    fn replace_cold(&mut self, value: &[u8]) {
        let start = self.offsets[self.offsets.len() - 2] as usize;
        self.values.truncate(start);
        self.values.extend_from_slice(value);
        *self.offsets.last_mut().unwrap() = self.values.len() as i64;
    }

    /// Builds a binary array, large or not as configured.
    fn finish(&mut self) -> Arc<dyn Array> {
        let (offsets, values, nulls) = self.take_parts();
        if self.large {
            let offsets = OffsetBuffer::new(offsets.into());
            Arc::new(LargeBinaryArray::new(offsets, values, nulls))
        } else {
            Arc::new(BinaryArray::new(narrow_offsets(offsets), values, nulls))
        }
    }

    /// The offsets, values and nulls so far; the column is left empty.
    fn take_parts(&mut self) -> (Vec<i64>, Buffer, Option<NullBuffer>) {
        let offsets = std::mem::replace(&mut self.offsets, vec![0]);
        let values = Buffer::from_vec(std::mem::take(&mut self.values));
        (offsets, values, self.nulls.finish())
    }
}

/// `i32` offsets for an array that isn't large.
fn narrow_offsets(offsets: Vec<i64>) -> OffsetBuffer<i32> {
    let offsets: Vec<i32> = offsets
        .into_iter()
        .map(|o| i32::try_from(o).expect("byte array offset overflow"))
        .collect();
    OffsetBuffer::new(offsets.into())
}

/// A [`ByteColumn`] of strings. It only takes `&str`, so every value is
/// UTF-8 and [`Self::finish`] can build a string array without checking.
struct StringColumn(ByteColumn);

impl StringColumn {
    fn new(large: bool) -> Self {
        Self(ByteColumn::new(large))
    }

    fn with_capacity(large: bool, rows: usize) -> Self {
        Self(ByteColumn::with_capacity(large, rows))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn push(&mut self, value: &str) {
        self.0.push(value.as_bytes());
    }

    fn push_null(&mut self) {
        self.0.push_null();
    }

    #[inline]
    fn push_or_replace(&mut self, value: &str, has_value: &mut bool) {
        self.0.push_or_replace(value.as_bytes(), has_value);
    }

    /// Builds a string array, large or not as configured.
    fn finish(&mut self) -> Arc<dyn Array> {
        let (offsets, values, nulls) = self.0.take_parts();
        // SAFETY: only `&str` values were pushed, so every value is UTF-8.
        if self.0.large {
            let offsets = OffsetBuffer::new(offsets.into());
            Arc::new(unsafe { LargeStringArray::new_unchecked(offsets, values, nulls) })
        } else {
            let offsets = narrow_offsets(offsets);
            Arc::new(unsafe { StringArray::new_unchecked(offsets, values, nulls) })
        }
    }
}

// ---------------------------------------------------------------------------
// ListOffsets
// ---------------------------------------------------------------------------

enum ListOffsets {
    Regular(Vec<i32>),
    Large(Vec<i64>),
}

impl ListOffsets {
    fn new(use_large: bool) -> Self {
        if use_large {
            Self::Large(vec![0])
        } else {
            Self::Regular(vec![0])
        }
    }
    fn push(&mut self, value: usize) {
        match self {
            Self::Regular(v) => v.push(value as i32),
            Self::Large(v) => v.push(value as i64),
        }
    }
    fn finish(self, values: Arc<dyn Array>, name: &str, nullable: bool) -> Arc<dyn Array> {
        let field = Arc::new(Field::new(name, values.data_type().clone(), nullable));
        match self {
            Self::Regular(offsets) => {
                let buf = Buffer::from_vec(offsets);
                let data = ArrayData::builder(DataType::List(field))
                    .len(buf.len() / 4 - 1)
                    .add_buffer(buf)
                    .add_child_data(values.to_data())
                    .build()
                    .unwrap();
                Arc::new(arrow_array::ListArray::from(data))
            }
            Self::Large(offsets) => {
                let buf = Buffer::from_vec(offsets);
                let data = ArrayData::builder(DataType::LargeList(field))
                    .len(buf.len() / 8 - 1)
                    .add_buffer(buf)
                    .add_child_data(values.to_data())
                    .build()
                    .unwrap();
                Arc::new(arrow_array::LargeListArray::from(data))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RepeatedInner enum — repeated field value storage
// ---------------------------------------------------------------------------

enum RepeatedInner {
    Int32 {
        values_builder: PrimitiveBuilder<Int32Type>,
    },
    Int64 {
        values_builder: PrimitiveBuilder<Int64Type>,
    },
    UInt32 {
        values_builder: PrimitiveBuilder<UInt32Type>,
    },
    UInt64 {
        values_builder: PrimitiveBuilder<UInt64Type>,
    },
    Float {
        values_builder: PrimitiveBuilder<Float32Type>,
    },
    Double {
        values_builder: PrimitiveBuilder<Float64Type>,
    },
    Bool {
        values_builder: BooleanBuilder,
    },
    String {
        values_builder: StringBuilderInner,
    },
    Bytes {
        values_builder: BinaryBuilderInner,
    },
    Sint32 {
        values_builder: PrimitiveBuilder<Int32Type>,
    },
    Sint64 {
        values_builder: PrimitiveBuilder<Int64Type>,
    },
    Sfixed32 {
        values_builder: PrimitiveBuilder<Int32Type>,
    },
    Sfixed64 {
        values_builder: PrimitiveBuilder<Int64Type>,
    },
    Fixed32 {
        values_builder: PrimitiveBuilder<UInt32Type>,
    },
    Fixed64 {
        values_builder: PrimitiveBuilder<UInt64Type>,
    },
    EnumInt32 {
        values_builder: PrimitiveBuilder<Int32Type>,
    },
    EnumString {
        values_builder: StringBuilderInner,
        enum_descriptor: EnumDescriptor,
    },
    EnumBinary {
        values_builder: BinaryBuilderInner,
        enum_descriptor: EnumDescriptor,
    },
    Message {
        sub_decoder: MessageDecoder,
    },
    Timestamp {
        values_builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
        tz: Option<Arc<str>>,
    },
    Duration {
        values_builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },
    Date {
        values_builder: PrimitiveBuilder<Date32Type>,
    },
    TimeOfDay {
        values_builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },
    WrapperDouble {
        values_builder: PrimitiveBuilder<Float64Type>,
    },
    WrapperFloat {
        values_builder: PrimitiveBuilder<Float32Type>,
    },
    WrapperInt64 {
        values_builder: PrimitiveBuilder<Int64Type>,
    },
    WrapperUInt64 {
        values_builder: PrimitiveBuilder<UInt64Type>,
    },
    WrapperInt32 {
        values_builder: PrimitiveBuilder<Int32Type>,
    },
    WrapperUInt32 {
        values_builder: PrimitiveBuilder<UInt32Type>,
    },
    WrapperBool {
        values_builder: BooleanBuilder,
    },
    WrapperString {
        values_builder: StringBuilderInner,
    },
    WrapperBytes {
        values_builder: BinaryBuilderInner,
    },
}

impl RepeatedInner {
    fn decode(&mut self, wire_type: u8, buf: &[u8]) -> Result<usize, prost::DecodeError> {
        match self {
            Self::Int32 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, |v| v as i32)
            }
            Self::Int64 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, |v| v as i64)
            }
            Self::UInt32 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, |v| v as u32)
            }
            Self::UInt64 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, |v| v)
            }
            Self::EnumInt32 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, |v| v as i32)
            }
            Self::Sint32 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, decode_zigzag32)
            }
            Self::Sint64 { values_builder, .. } => {
                decode_repeated_varint(wire_type, buf, values_builder, decode_zigzag64)
            }
            Self::Sfixed32 { values_builder, .. } => decode_repeated_fixed::<Int32Type, 4>(
                wire_type,
                5,
                buf,
                values_builder,
                i32::from_le_bytes,
            ),
            Self::Sfixed64 { values_builder, .. } => decode_repeated_fixed::<Int64Type, 8>(
                wire_type,
                1,
                buf,
                values_builder,
                i64::from_le_bytes,
            ),
            Self::Fixed32 { values_builder, .. } => decode_repeated_fixed::<UInt32Type, 4>(
                wire_type,
                5,
                buf,
                values_builder,
                u32::from_le_bytes,
            ),
            Self::Fixed64 { values_builder, .. } => decode_repeated_fixed::<UInt64Type, 8>(
                wire_type,
                1,
                buf,
                values_builder,
                u64::from_le_bytes,
            ),
            Self::Float { values_builder, .. } => decode_repeated_fixed::<Float32Type, 4>(
                wire_type,
                5,
                buf,
                values_builder,
                f32::from_le_bytes,
            ),
            Self::Double { values_builder, .. } => decode_repeated_fixed::<Float64Type, 8>(
                wire_type,
                1,
                buf,
                values_builder,
                f64::from_le_bytes,
            ),
            Self::Bool { values_builder, .. } => {
                if wire_type == 2 {
                    let (data, total) = read_length_delimited(buf)?;
                    let mut p = 0;
                    while p < data.len() {
                        let (v, n) = decode_varint(&data[p..])?;
                        values_builder.append_value(v != 0);
                        p += n;
                    }
                    Ok(total)
                } else if wire_type == 0 {
                    let (v, n) = decode_varint(buf)?;
                    values_builder.append_value(v != 0);
                    Ok(n)
                } else {
                    skip_field(wire_type, buf)
                }
            }
            Self::String { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let s = validate_utf8(data)?;
                values_builder.append_value(s);
                Ok(total)
            }
            Self::Bytes { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                values_builder.append_value(data);
                Ok(total)
            }
            Self::EnumString {
                values_builder,
                enum_descriptor,
                ..
            } => {
                if wire_type == 2 {
                    let (data, total) = read_length_delimited(buf)?;
                    let mut p = 0;
                    while p < data.len() {
                        let (v, n) = decode_varint(&data[p..])?;
                        values_builder.append_value(&enum_name(enum_descriptor, v as i32));
                        p += n;
                    }
                    Ok(total)
                } else if wire_type == 0 {
                    let (v, n) = decode_varint(buf)?;
                    values_builder.append_value(&enum_name(enum_descriptor, v as i32));
                    Ok(n)
                } else {
                    skip_field(wire_type, buf)
                }
            }
            Self::EnumBinary {
                values_builder,
                enum_descriptor,
                ..
            } => {
                if wire_type == 2 {
                    let (data, total) = read_length_delimited(buf)?;
                    let mut p = 0;
                    while p < data.len() {
                        let (v, n) = decode_varint(&data[p..])?;
                        values_builder
                            .append_value(enum_name(enum_descriptor, v as i32).as_bytes());
                        p += n;
                    }
                    Ok(total)
                } else if wire_type == 0 {
                    let (v, n) = decode_varint(buf)?;
                    values_builder.append_value(enum_name(enum_descriptor, v as i32).as_bytes());
                    Ok(n)
                } else {
                    skip_field(wire_type, buf)
                }
            }
            Self::Message { sub_decoder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                sub_decoder.decode_row(data)?;
                Ok(total)
            }
            Self::Timestamp {
                values_builder,
                unit,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                values_builder.append_value(convert_seconds_nanos_to_unit(
                    vals[0],
                    vals[1] as i32,
                    *unit,
                    "Timestamp",
                ));
                Ok(total)
            }
            Self::Duration {
                values_builder,
                unit,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                values_builder.append_value(convert_seconds_nanos_to_unit(
                    vals[0],
                    vals[1] as i32,
                    *unit,
                    "Duration",
                ));
                Ok(total)
            }
            Self::Date { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 3)?;
                let (y, m, d) = (vals[0] as i32, vals[1] as i32, vals[2] as i32);
                if y == 0 && m == 0 && d == 0 {
                    values_builder.append_value(0);
                } else {
                    values_builder.append_value(
                        chrono::NaiveDate::from_ymd_opt(y, m as u32, d as u32)
                            .unwrap()
                            .num_days_from_ce()
                            - CE_OFFSET,
                    );
                }
                Ok(total)
            }
            Self::TimeOfDay {
                values_builder,
                unit,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 4)?;
                let total_seconds = vals[0] * 3600 + vals[1] * 60 + vals[2];
                values_builder.append_value(convert_seconds_nanos_to_unit(
                    total_seconds,
                    vals[3] as i32,
                    *unit,
                    "TimeOfDay",
                ));
                Ok(total)
            }
            Self::WrapperDouble { values_builder, .. } => {
                decode_repeated_wrapper_fixed64(wire_type, buf, values_builder, f64::from_le_bytes)
            }
            Self::WrapperFloat { values_builder, .. } => {
                decode_repeated_wrapper_fixed32(wire_type, buf, values_builder, f32::from_le_bytes)
            }
            Self::WrapperInt64 { values_builder, .. } => {
                decode_repeated_wrapper_varint(wire_type, buf, values_builder, |v| v as i64)
            }
            Self::WrapperUInt64 { values_builder, .. } => {
                decode_repeated_wrapper_varint(wire_type, buf, values_builder, |v| v)
            }
            Self::WrapperInt32 { values_builder, .. } => {
                decode_repeated_wrapper_varint(wire_type, buf, values_builder, |v| v as i32)
            }
            Self::WrapperUInt32 { values_builder, .. } => {
                decode_repeated_wrapper_varint(wire_type, buf, values_builder, |v| v as u32)
            }
            Self::WrapperBool { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                values_builder.append_value(v != 0);
                Ok(total)
            }
            Self::WrapperString { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_string(data)?;
                if found {
                    values_builder.append_value(unsafe { std::str::from_utf8_unchecked(&v) });
                } else {
                    values_builder.append_value("");
                }
                Ok(total)
            }
            Self::WrapperBytes { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_bytes(data)?;
                if found {
                    values_builder.append_value(&v);
                } else {
                    values_builder.append_value(b"");
                }
                Ok(total)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Int32 { values_builder, .. }
            | Self::Sint32 { values_builder, .. }
            | Self::Sfixed32 { values_builder, .. }
            | Self::EnumInt32 { values_builder, .. }
            | Self::WrapperInt32 { values_builder, .. } => values_builder.len(),
            Self::Int64 { values_builder, .. }
            | Self::Sint64 { values_builder, .. }
            | Self::Sfixed64 { values_builder, .. }
            | Self::Timestamp { values_builder, .. }
            | Self::Duration { values_builder, .. }
            | Self::WrapperInt64 { values_builder, .. }
            | Self::TimeOfDay { values_builder, .. } => values_builder.len(),
            Self::UInt32 { values_builder, .. }
            | Self::Fixed32 { values_builder, .. }
            | Self::WrapperUInt32 { values_builder, .. } => values_builder.len(),
            Self::UInt64 { values_builder, .. }
            | Self::Fixed64 { values_builder, .. }
            | Self::WrapperUInt64 { values_builder, .. } => values_builder.len(),
            Self::Float { values_builder, .. } | Self::WrapperFloat { values_builder, .. } => {
                values_builder.len()
            }
            Self::Double { values_builder, .. } | Self::WrapperDouble { values_builder, .. } => {
                values_builder.len()
            }
            Self::Bool { values_builder, .. } | Self::WrapperBool { values_builder, .. } => {
                values_builder.len()
            }
            Self::String { values_builder, .. }
            | Self::EnumString { values_builder, .. }
            | Self::WrapperString { values_builder, .. } => values_builder.len(),
            Self::Bytes { values_builder, .. }
            | Self::EnumBinary { values_builder, .. }
            | Self::WrapperBytes { values_builder, .. } => values_builder.len(),
            Self::Message { sub_decoder, .. } => sub_decoder.row_count(),
            Self::Date { values_builder, .. } => values_builder.len(),
        }
    }

    fn finish(&mut self) -> Arc<dyn Array> {
        match self {
            Self::Int32 { values_builder, .. }
            | Self::Sint32 { values_builder, .. }
            | Self::Sfixed32 { values_builder, .. }
            | Self::EnumInt32 { values_builder, .. }
            | Self::WrapperInt32 { values_builder, .. } => Arc::new(values_builder.finish()),
            Self::Int64 { values_builder, .. }
            | Self::Sint64 { values_builder, .. }
            | Self::Sfixed64 { values_builder, .. }
            | Self::WrapperInt64 { values_builder, .. } => Arc::new(values_builder.finish()),
            Self::UInt32 { values_builder, .. }
            | Self::Fixed32 { values_builder, .. }
            | Self::WrapperUInt32 { values_builder, .. } => Arc::new(values_builder.finish()),
            Self::UInt64 { values_builder, .. }
            | Self::Fixed64 { values_builder, .. }
            | Self::WrapperUInt64 { values_builder, .. } => Arc::new(values_builder.finish()),
            Self::Float { values_builder, .. } | Self::WrapperFloat { values_builder, .. } => {
                Arc::new(values_builder.finish())
            }
            Self::Double { values_builder, .. } | Self::WrapperDouble { values_builder, .. } => {
                Arc::new(values_builder.finish())
            }
            Self::Bool { values_builder, .. } | Self::WrapperBool { values_builder, .. } => {
                Arc::new(values_builder.finish())
            }
            Self::String { values_builder, .. }
            | Self::EnumString { values_builder, .. }
            | Self::WrapperString { values_builder, .. } => values_builder.finish(),
            Self::Bytes { values_builder, .. }
            | Self::EnumBinary { values_builder, .. }
            | Self::WrapperBytes { values_builder, .. } => values_builder.finish(),
            Self::Message { sub_decoder, .. } => Arc::new(sub_decoder.build_struct_array(None)),
            Self::Timestamp {
                values_builder,
                unit,
                tz,
            } => finish_timestamp(values_builder, *unit, tz),
            Self::Duration {
                values_builder,
                unit,
            } => finish_duration(values_builder, *unit),
            Self::Date { values_builder, .. } => Arc::new(values_builder.finish()),
            Self::TimeOfDay {
                values_builder,
                unit,
                ..
            } => finish_time_of_day(values_builder, *unit),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldDecoder enum — all types
// ---------------------------------------------------------------------------

enum FieldDecoder {
    // --- Singular scalars (buffered) ---
    Int32 {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int32Type>,
    },
    Int64 {
        value: i64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int64Type>,
    },
    UInt32 {
        value: u32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<UInt32Type>,
    },
    UInt64 {
        value: u64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<UInt64Type>,
    },
    Sint32 {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int32Type>,
    },
    Sint64 {
        value: i64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int64Type>,
    },
    Sfixed32 {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int32Type>,
    },
    Sfixed64 {
        value: i64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int64Type>,
    },
    Fixed32 {
        value: u32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<UInt32Type>,
    },
    Fixed64 {
        value: u64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<UInt64Type>,
    },
    Float {
        value: f32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Float32Type>,
    },
    Double {
        value: f64,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Float64Type>,
    },
    Bool {
        value: bool,
        has_value: bool,
        has_presence: bool,
        builder: BooleanBuilder,
    },
    String {
        has_presence: bool,
        builder: StringColumn,
    },
    Bytes {
        has_presence: bool,
        builder: ByteColumn,
    },
    EnumInt32 {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: PrimitiveBuilder<Int32Type>,
    },
    EnumString {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: StringBuilderInner,
        enum_descriptor: EnumDescriptor,
    },
    EnumBinary {
        value: i32,
        has_value: bool,
        has_presence: bool,
        builder: BinaryBuilderInner,
        enum_descriptor: EnumDescriptor,
    },

    // --- Well-known types (singular, buffered) ---
    Timestamp {
        seconds: i64,
        nanos: i32,
        has_value: bool,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
        tz: Option<Arc<str>>,
    },
    Duration {
        seconds: i64,
        nanos: i32,
        has_value: bool,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },
    Date {
        year: i32,
        month: i32,
        day: i32,
        has_value: bool,
        builder: PrimitiveBuilder<Date32Type>,
    },
    TimeOfDay {
        hours: i32,
        minutes: i32,
        seconds_val: i32,
        nanos: i32,
        has_value: bool,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },

    // --- Wrapper types (singular) ---
    WrapperDouble {
        value: f64,
        has_value: bool,
        builder: PrimitiveBuilder<Float64Type>,
    },
    WrapperFloat {
        value: f32,
        has_value: bool,
        builder: PrimitiveBuilder<Float32Type>,
    },
    WrapperInt64 {
        value: i64,
        has_value: bool,
        builder: PrimitiveBuilder<Int64Type>,
    },
    WrapperUInt64 {
        value: u64,
        has_value: bool,
        builder: PrimitiveBuilder<UInt64Type>,
    },
    WrapperInt32 {
        value: i32,
        has_value: bool,
        builder: PrimitiveBuilder<Int32Type>,
    },
    WrapperUInt32 {
        value: u32,
        has_value: bool,
        builder: PrimitiveBuilder<UInt32Type>,
    },
    WrapperBool {
        value: bool,
        has_value: bool,
        builder: BooleanBuilder,
    },
    WrapperString {
        value: Vec<u8>,
        has_value: bool,
        builder: StringBuilderInner,
    },
    WrapperBytes {
        value: Vec<u8>,
        has_value: bool,
        builder: BinaryBuilderInner,
    },

    // --- Nested message ---
    Message {
        sub_decoder: MessageDecoder,
        has_value: bool,
        is_valid: BooleanBuilder,
    },

    // --- Repeated fields (all collapsed into one variant) ---
    Repeated {
        inner: RepeatedInner,
        offsets: ListOffsets,
        list_name: Arc<str>,
        list_nullable: bool,
    },

    // --- Map fields ---
    Map {
        key_decoder: Box<FieldDecoder>,
        value_decoder: Box<FieldDecoder>,
        offsets: Vec<i32>,
        map_value_name: Arc<str>,
        map_value_nullable: bool,
    },
}

// ---------------------------------------------------------------------------
// Helpers for decoding wire values inline
// ---------------------------------------------------------------------------

/// Decode fields of a well-known submessage with up to 4 int fields.
/// Returns (field1, field2, field3, field4) initialized to 0, scanning for field numbers 1..=max_field.
fn decode_wkt_submessage(buf: &[u8], max_field: u32) -> Result<[i64; 4], prost::DecodeError> {
    let mut vals = [0i64; 4];
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum >= 1 && fnum <= max_field && wt == 0 {
            let (v, n) = decode_varint(&buf[pos..])?;
            vals[(fnum - 1) as usize] = v as i64;
            pos += n;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok(vals)
}

/// Decode a wrapper submessage: field 1 with the given wire type.
/// For varint wrapper types.
fn decode_wrapper_varint(buf: &[u8]) -> Result<(u64, bool), prost::DecodeError> {
    let mut val = 0u64;
    let mut found = false;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 0 {
            let (v, n) = decode_varint(&buf[pos..])?;
            val = v;
            found = true;
            pos += n;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok((val, found))
}

fn decode_wrapper_fixed32(buf: &[u8]) -> Result<([u8; 4], bool), prost::DecodeError> {
    let mut val = [0u8; 4];
    let mut found = false;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 5 {
            if buf.len() < pos + 4 {
                return Err(decode_error("unexpected EOF"));
            }
            val.copy_from_slice(&buf[pos..pos + 4]);
            found = true;
            pos += 4;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok((val, found))
}

fn decode_wrapper_fixed64(buf: &[u8]) -> Result<([u8; 8], bool), prost::DecodeError> {
    let mut val = [0u8; 8];
    let mut found = false;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 1 {
            if buf.len() < pos + 8 {
                return Err(decode_error("unexpected EOF"));
            }
            val.copy_from_slice(&buf[pos..pos + 8]);
            found = true;
            pos += 8;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok((val, found))
}

fn decode_wrapper_string(buf: &[u8]) -> Result<(Vec<u8>, bool), prost::DecodeError> {
    let mut val = Vec::new();
    let mut found = false;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 2 {
            let (data, consumed) = read_length_delimited(&buf[pos..])?;
            validate_utf8(data)?;
            val.clear();
            val.extend_from_slice(data);
            found = true;
            pos += consumed;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok((val, found))
}

fn decode_wrapper_bytes(buf: &[u8]) -> Result<(Vec<u8>, bool), prost::DecodeError> {
    let mut val = Vec::new();
    let mut found = false;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 2 {
            let (data, consumed) = read_length_delimited(&buf[pos..])?;
            val.clear();
            val.extend_from_slice(data);
            found = true;
            pos += consumed;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok((val, found))
}

// ---------------------------------------------------------------------------
// Macro for flush/finish boilerplate
// ---------------------------------------------------------------------------

/// Appends `value`, or replaces the value appended earlier in this row when
/// the field occurs again (`has_value`).
#[inline(always)]
fn push_or_replace<T: ArrowPrimitiveType>(
    builder: &mut PrimitiveBuilder<T>,
    value: T::Native,
    has_value: &mut bool,
) {
    if *has_value {
        *builder.values_slice_mut().last_mut().unwrap() = value;
    } else {
        builder.append_value(value);
        *has_value = true;
    }
}

#[inline(always)]
fn push_or_replace_with<T: ArrowPrimitiveType, const N: usize>(
    builder: &mut PrimitiveBuilder<T>,
    buf: &[u8],
    has_value: &mut bool,
    f: impl FnOnce([u8; N]) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if buf.len() < N {
        return Err(decode_error("unexpected EOF"));
    }
    push_or_replace(builder, (f)(buf[..N].try_into().unwrap()), has_value);
    Ok(N)
}

macro_rules! flush_primitive {
    ($value:expr, $has_value:expr, $has_presence:expr, $builder:expr, $default:expr) => {
        if *$has_value {
            $builder.append_value(*$value);
        } else if *$has_presence {
            $builder.append_null();
        } else {
            $builder.append_value($default);
        }
        *$has_value = false;
        *$value = $default;
    };
}

fn finish_primitive<T: ArrowPrimitiveType>(builder: &mut PrimitiveBuilder<T>) -> Arc<dyn Array> {
    Arc::new(builder.finish())
}

fn finish_timestamp(
    builder: &mut PrimitiveBuilder<Int64Type>,
    unit: TimeUnit,
    tz: &Option<Arc<str>>,
) -> Arc<dyn Array> {
    let values = builder.finish();
    let dt = DataType::Timestamp(unit, tz.clone());
    let data = ArrayData::builder(dt)
        .len(values.len())
        .add_buffer(values.values().inner().clone())
        .null_bit_buffer(values.nulls().map(|n| n.buffer().clone()))
        .build()
        .unwrap();
    arrow_array::make_array(data)
}

fn finish_duration(builder: &mut PrimitiveBuilder<Int64Type>, unit: TimeUnit) -> Arc<dyn Array> {
    let values = builder.finish();
    let dt = DataType::Duration(unit);
    let data = ArrayData::builder(dt)
        .len(values.len())
        .add_buffer(values.values().inner().clone())
        .null_bit_buffer(values.nulls().map(|n| n.buffer().clone()))
        .build()
        .unwrap();
    arrow_array::make_array(data)
}

fn finish_time_of_day(builder: &mut PrimitiveBuilder<Int64Type>, unit: TimeUnit) -> Arc<dyn Array> {
    let values = builder.finish();
    let dt = match unit {
        TimeUnit::Second => DataType::Time32(TimeUnit::Second),
        TimeUnit::Millisecond => DataType::Time32(TimeUnit::Millisecond),
        TimeUnit::Microsecond => DataType::Time64(TimeUnit::Microsecond),
        TimeUnit::Nanosecond => DataType::Time64(TimeUnit::Nanosecond),
    };
    if matches!(unit, TimeUnit::Second | TimeUnit::Millisecond) {
        let i32_values: Vec<Option<i32>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    let v = values.value(i);
                    Some(i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN }))
                }
            })
            .collect();
        let i32_array = arrow_array::Int32Array::from(i32_values);
        let data = ArrayData::builder(dt)
            .len(i32_array.len())
            .add_buffer(i32_array.values().inner().clone())
            .null_bit_buffer(i32_array.nulls().map(|n| n.buffer().clone()))
            .build()
            .unwrap();
        arrow_array::make_array(data)
    } else {
        let data = ArrayData::builder(dt)
            .len(values.len())
            .add_buffer(values.values().inner().clone())
            .null_bit_buffer(values.nulls().map(|n| n.buffer().clone()))
            .build()
            .unwrap();
        arrow_array::make_array(data)
    }
}

// ---------------------------------------------------------------------------
// FieldDecoder: decode + flush + finish
// ---------------------------------------------------------------------------

impl FieldDecoder {
    /// Decodes one occurrence of this field. Singular numbers, enum numbers,
    /// strings and bytes are appended directly and set `has_value`; a repeat in
    /// the same row replaces the value. Bools (arrow's `BooleanBuilder` can't
    /// replace a value), enum names, wrapper and well-known types buffer their
    /// value until [`Self::flush`] and leave `has_value` alone.
    #[inline(always)]
    fn decode(
        &mut self,
        wire_type: u8,
        buf: &[u8],
        has_value: &mut bool,
    ) -> Result<usize, prost::DecodeError> {
        match self {
            Self::Int32 { builder, .. } | Self::EnumInt32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as i32, has_value);
                Ok(n)
            }
            Self::EnumString {
                value, has_value, ..
            }
            | Self::EnumBinary {
                value, has_value, ..
            } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                *value = v as i32;
                *has_value = true;
                Ok(n)
            }
            Self::Int64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as i64, has_value);
                Ok(n)
            }
            Self::UInt32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as u32, has_value);
                Ok(n)
            }
            Self::UInt64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v, has_value);
                Ok(n)
            }
            Self::Sint32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, decode_zigzag32(v), has_value);
                Ok(n)
            }
            Self::Sint64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, decode_zigzag64(v), has_value);
                Ok(n)
            }
            Self::Sfixed32 { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, i32::from_le_bytes)
            }
            Self::Sfixed64 { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, i64::from_le_bytes)
            }
            Self::Fixed32 { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, u32::from_le_bytes)
            }
            Self::Fixed64 { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, u64::from_le_bytes)
            }
            Self::Float { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, f32::from_le_bytes)
            }
            Self::Double { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, has_value, f64::from_le_bytes)
            }
            Self::Bool {
                value, has_value, ..
            } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                *value = v != 0;
                *has_value = true;
                Ok(n)
            }
            Self::String { builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_or_replace(validate_utf8(data)?, has_value);
                Ok(total)
            }
            Self::Bytes { builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_or_replace(data, has_value);
                Ok(total)
            }
            // Well-known types: decode submessage
            Self::Timestamp {
                seconds,
                nanos,
                has_value,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                *seconds = vals[0];
                *nanos = vals[1] as i32;
                *has_value = true;
                Ok(total)
            }
            Self::Duration {
                seconds,
                nanos,
                has_value,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                *seconds = vals[0];
                *nanos = vals[1] as i32;
                *has_value = true;
                Ok(total)
            }
            Self::Date {
                year,
                month,
                day,
                has_value,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 3)?;
                *year = vals[0] as i32;
                *month = vals[1] as i32;
                *day = vals[2] as i32;
                *has_value = true;
                Ok(total)
            }
            Self::TimeOfDay {
                hours,
                minutes,
                seconds_val,
                nanos,
                has_value,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 4)?;
                *hours = vals[0] as i32;
                *minutes = vals[1] as i32;
                *seconds_val = vals[2] as i32;
                *nanos = vals[3] as i32;
                *has_value = true;
                Ok(total)
            }
            // Wrapper types
            Self::WrapperDouble {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (bytes, found) = decode_wrapper_fixed64(data)?;
                if found {
                    *value = f64::from_le_bytes(bytes);
                }
                *has_value = true;
                Ok(total)
            }
            Self::WrapperFloat {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (bytes, found) = decode_wrapper_fixed32(data)?;
                if found {
                    *value = f32::from_le_bytes(bytes);
                }
                *has_value = true;
                Ok(total)
            }
            Self::WrapperInt64 {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                *value = v as i64;
                *has_value = true;
                Ok(total)
            }
            Self::WrapperUInt64 {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                *value = v;
                *has_value = true;
                Ok(total)
            }
            Self::WrapperInt32 {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                *value = v as i32;
                *has_value = true;
                Ok(total)
            }
            Self::WrapperUInt32 {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                *value = v as u32;
                *has_value = true;
                Ok(total)
            }
            Self::WrapperBool {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_varint(data)?;
                *value = v != 0;
                *has_value = true;
                Ok(total)
            }
            Self::WrapperString {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_string(data)?;
                value.clear();
                value.extend_from_slice(&v);
                *has_value = true;
                Ok(total)
            }
            Self::WrapperBytes {
                value, has_value, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, _) = decode_wrapper_bytes(data)?;
                value.clear();
                value.extend_from_slice(&v);
                *has_value = true;
                Ok(total)
            }
            // Nested message
            Self::Message {
                sub_decoder,
                has_value,
                ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                // For singular messages, if seen multiple times, the spec says merge.
                // For simplicity (and matching DynamicMessage behavior), we decode fresh each time.
                // Reset sub_decoder if already decoded this row.
                if !*has_value {
                    *has_value = true;
                }
                sub_decoder.decode_message_bytes(data)?;
                Ok(total)
            }
            // Repeated fields — delegate to inner
            Self::Repeated { inner, .. } => inner.decode(wire_type, buf),
            // Map: each occurrence is a length-delimited entry submessage
            Self::Map {
                key_decoder,
                value_decoder,
                ..
            } => Self::decode_map_entry(wire_type, buf, key_decoder, value_decoder),
        }
    }

    /// Decodes one entry of a map field. Out of line, since it decodes the
    /// key and the value with [`Self::decode`].
    #[inline(never)]
    fn decode_map_entry(
        wire_type: u8,
        buf: &[u8],
        key_decoder: &mut FieldDecoder,
        value_decoder: &mut FieldDecoder,
    ) -> Result<usize, prost::DecodeError> {
        if wire_type != 2 {
            return skip_field(wire_type, buf);
        }
        let (data, total) = read_length_delimited(buf)?;
        // Parse entry submessage: field 1 = key, field 2 = value
        let mut pos = 0;
        let (mut key_seen, mut value_seen) = (false, false);
        while pos < data.len() {
            let (fnum, wt, n) = decode_tag(&data[pos..])?;
            pos += n;
            if fnum == 1 {
                pos += key_decoder.decode(wt, &data[pos..], &mut key_seen)?;
            } else if fnum == 2 {
                pos += value_decoder.decode(wt, &data[pos..], &mut value_seen)?;
            } else {
                pos += skip_field(wt, &data[pos..])?;
            }
        }
        if !key_seen {
            key_decoder.flush();
        }
        if !value_seen {
            value_decoder.flush();
        }
        Ok(total)
    }

    fn flush(&mut self) {
        match self {
            Self::Int32 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Sint32 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Sfixed32 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::EnumInt32 {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0i32);
            }
            Self::Int64 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Sint64 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Sfixed64 {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0i64);
            }
            Self::UInt32 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Fixed32 {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0u32);
            }
            Self::UInt64 {
                value,
                has_value,
                has_presence,
                builder,
            }
            | Self::Fixed64 {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0u64);
            }
            Self::Float {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0.0f32);
            }
            Self::Double {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                flush_primitive!(value, has_value, has_presence, builder, 0.0f64);
            }
            Self::EnumString {
                value,
                has_value,
                has_presence,
                builder,
                enum_descriptor,
            } => {
                if *has_value {
                    builder.append_value(&enum_name(enum_descriptor, *value));
                } else if *has_presence {
                    builder.append_null();
                } else {
                    builder.append_value(&enum_name(enum_descriptor, 0));
                }
                *has_value = false;
                *value = 0;
            }
            Self::EnumBinary {
                value,
                has_value,
                has_presence,
                builder,
                enum_descriptor,
            } => {
                if *has_value {
                    builder.append_value(enum_name(enum_descriptor, *value).as_bytes());
                } else if *has_presence {
                    builder.append_null();
                } else {
                    builder.append_value(enum_name(enum_descriptor, 0).as_bytes());
                }
                *has_value = false;
                *value = 0;
            }
            Self::Bool {
                value,
                has_value,
                has_presence,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else if *has_presence {
                    builder.append_null();
                } else {
                    builder.append_value(false);
                }
                *has_value = false;
                *value = false;
            }
            // Appended in decode; flushed only when absent from the row.
            Self::String {
                has_presence,
                builder,
            } => {
                if *has_presence {
                    builder.push_null();
                } else {
                    builder.push("");
                }
            }
            Self::Bytes {
                has_presence,
                builder,
            } => {
                if *has_presence {
                    builder.push_null();
                } else {
                    builder.push(b"");
                }
            }
            // Well-known types
            Self::Timestamp {
                seconds,
                nanos,
                has_value,
                builder,
                unit,
                ..
            } => {
                if *has_value {
                    builder.append_value(convert_seconds_nanos_to_unit(
                        *seconds,
                        *nanos,
                        *unit,
                        "Timestamp",
                    ));
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *seconds = 0;
                *nanos = 0;
            }
            Self::Duration {
                seconds,
                nanos,
                has_value,
                builder,
                unit,
                ..
            } => {
                if *has_value {
                    builder.append_value(convert_seconds_nanos_to_unit(
                        *seconds, *nanos, *unit, "Duration",
                    ));
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *seconds = 0;
                *nanos = 0;
            }
            Self::Date {
                year,
                month,
                day,
                has_value,
                builder,
            } => {
                if *has_value {
                    if *year == 0 && *month == 0 && *day == 0 {
                        builder.append_value(0);
                    } else {
                        builder.append_value(
                            chrono::NaiveDate::from_ymd_opt(*year, *month as u32, *day as u32)
                                .unwrap()
                                .num_days_from_ce()
                                - CE_OFFSET,
                        );
                    }
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *year = 0;
                *month = 0;
                *day = 0;
            }
            Self::TimeOfDay {
                hours,
                minutes,
                seconds_val,
                nanos,
                has_value,
                builder,
                unit,
            } => {
                if *has_value {
                    let total_seconds = i64::from(*hours) * 3600
                        + i64::from(*minutes) * 60
                        + i64::from(*seconds_val);
                    builder.append_value(convert_seconds_nanos_to_unit(
                        total_seconds,
                        *nanos,
                        *unit,
                        "TimeOfDay",
                    ));
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *hours = 0;
                *minutes = 0;
                *seconds_val = 0;
                *nanos = 0;
            }
            // Wrapper types: present → value, absent → null
            Self::WrapperDouble {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0.0;
            }
            Self::WrapperFloat {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0.0;
            }
            Self::WrapperInt64 {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0;
            }
            Self::WrapperUInt64 {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0;
            }
            Self::WrapperInt32 {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0;
            }
            Self::WrapperUInt32 {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = 0;
            }
            Self::WrapperBool {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *has_value = false;
                *value = false;
            }
            Self::WrapperString {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(unsafe { std::str::from_utf8_unchecked(value) });
                } else {
                    builder.append_null();
                }
                *has_value = false;
                value.clear();
            }
            Self::WrapperBytes {
                value,
                has_value,
                builder,
            } => {
                if *has_value {
                    builder.append_value(value.as_slice());
                } else {
                    builder.append_null();
                }
                *has_value = false;
                value.clear();
            }
            // Nested message
            Self::Message {
                sub_decoder,
                has_value,
                is_valid,
            } => {
                if *has_value {
                    is_valid.append_value(true);
                    sub_decoder.flush_row();
                } else {
                    is_valid.append_value(false);
                    sub_decoder.flush_defaults();
                }
                *has_value = false;
            }
            // Repeated fields: push offset
            Self::Repeated { inner, offsets, .. } => {
                offsets.push(inner.len());
            }
            // Map: push offset based on key builder length
            Self::Map {
                key_decoder,
                offsets,
                ..
            } => {
                let count = match key_decoder.as_ref() {
                    FieldDecoder::Int32 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::Int64 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::UInt32 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::UInt64 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::Sint32 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::Sint64 { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::Bool { builder, .. } => ArrayBuilder::len(builder),
                    FieldDecoder::String { builder, .. } => builder.len(),
                    _ => *offsets.last().unwrap() as usize,
                };
                offsets.push(count as i32);
            }
        }
    }

    fn finish(&mut self, nullable: bool) -> (Field, Arc<dyn Array>) {
        // This is called by MessageDecoder::finish, which provides the field name separately
        // We return a dummy field name here; the caller replaces it.
        let array: Arc<dyn Array> = match self {
            Self::Int32 { builder, .. } | Self::EnumInt32 { builder, .. } => {
                finish_primitive(builder)
            }
            Self::Int64 { builder, .. } => finish_primitive(builder),
            Self::UInt32 { builder, .. } => finish_primitive(builder),
            Self::UInt64 { builder, .. } => finish_primitive(builder),
            Self::Sint32 { builder, .. } | Self::Sfixed32 { builder, .. } => {
                finish_primitive(builder)
            }
            Self::Sint64 { builder, .. } | Self::Sfixed64 { builder, .. } => {
                finish_primitive(builder)
            }
            Self::Fixed32 { builder, .. } => finish_primitive(builder),
            Self::Fixed64 { builder, .. } => finish_primitive(builder),
            Self::Float { builder, .. } => finish_primitive(builder),
            Self::Double { builder, .. } => finish_primitive(builder),
            Self::Bool { builder, .. } => Arc::new(builder.finish()),
            Self::String { builder, .. } => builder.finish(),
            Self::EnumString { builder, .. } => builder.finish(),
            Self::Bytes { builder, .. } => builder.finish(),
            Self::EnumBinary { builder, .. } => builder.finish(),
            Self::Timestamp {
                builder, unit, tz, ..
            } => finish_timestamp(builder, *unit, tz),
            Self::Duration { builder, unit, .. } => finish_duration(builder, *unit),
            Self::Date { builder, .. } => finish_primitive(builder),
            Self::TimeOfDay { builder, unit, .. } => finish_time_of_day(builder, *unit),
            Self::WrapperDouble { builder, .. } => finish_primitive(builder),
            Self::WrapperFloat { builder, .. } => finish_primitive(builder),
            Self::WrapperInt64 { builder, .. } => finish_primitive(builder),
            Self::WrapperUInt64 { builder, .. } => finish_primitive(builder),
            Self::WrapperInt32 { builder, .. } => finish_primitive(builder),
            Self::WrapperUInt32 { builder, .. } => finish_primitive(builder),
            Self::WrapperBool { builder, .. } => Arc::new(builder.finish()),
            Self::WrapperString { builder, .. } => builder.finish(),
            Self::WrapperBytes { builder, .. } => builder.finish(),
            Self::Message {
                sub_decoder,
                is_valid,
                ..
            } => Arc::new(sub_decoder.build_struct_array(Some(is_valid.finish()))),
            Self::Repeated {
                inner,
                offsets,
                list_name,
                list_nullable,
            } => {
                let vals = inner.finish();
                std::mem::replace(offsets, ListOffsets::new(false)).finish(
                    vals,
                    list_name,
                    *list_nullable,
                )
            }
            Self::Map {
                key_decoder,
                value_decoder,
                offsets,
                map_value_name,
                map_value_nullable,
            } => {
                let (_, key_array) = key_decoder.finish(false);
                let (_, value_array) = value_decoder.finish(*map_value_nullable);
                let key_field = Arc::new(Field::new("key", key_array.data_type().clone(), false));
                let value_field = Arc::new(Field::new(
                    &**map_value_name,
                    value_array.data_type().clone(),
                    *map_value_nullable,
                ));
                let entries_struct_type = DataType::Struct(
                    vec![key_field.as_ref().clone(), value_field.as_ref().clone()].into(),
                );
                let entry_struct =
                    StructArray::from(vec![(key_field, key_array), (value_field, value_array)]);
                let map_dt = DataType::Map(
                    Arc::new(Field::new("entries", entries_struct_type, false)),
                    false,
                );
                let len = offsets.len() - 1;
                let offsets_buf = Buffer::from_vec(std::mem::take(offsets));
                let map_data = ArrayData::builder(map_dt)
                    .len(len)
                    .add_buffer(offsets_buf)
                    .add_child_data(entry_struct.into_data())
                    .build()
                    .unwrap();
                Arc::new(MapArray::from(map_data))
            }
        };
        let field = Field::new("", array.data_type().clone(), nullable);
        (field, array)
    }
}

// ---------------------------------------------------------------------------
// Helpers for repeated varint/fixed decoding
// ---------------------------------------------------------------------------

fn decode_repeated_varint<T: ArrowPrimitiveType>(
    wire_type: u8,
    buf: &[u8],
    builder: &mut PrimitiveBuilder<T>,
    convert: fn(u64) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if wire_type == 2 {
        // packed
        let (data, total) = read_length_delimited(buf)?;
        let mut p = 0;
        while p < data.len() {
            let (v, n) = decode_varint(&data[p..])?;
            builder.append_value(convert(v));
            p += n;
        }
        Ok(total)
    } else if wire_type == 0 {
        let (v, n) = decode_varint(buf)?;
        builder.append_value(convert(v));
        Ok(n)
    } else {
        skip_field(wire_type, buf)
    }
}

fn decode_repeated_fixed<T: ArrowPrimitiveType, const WIDTH: usize>(
    wire_type: u8,
    expected_wt: u8,
    buf: &[u8],
    builder: &mut PrimitiveBuilder<T>,
    convert: fn([u8; WIDTH]) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if wire_type == 2 {
        let (data, total) = read_length_delimited(buf)?;
        let mut p = 0;
        while p + WIDTH <= data.len() {
            let mut bytes = [0u8; WIDTH];
            bytes.copy_from_slice(&data[p..p + WIDTH]);
            builder.append_value(convert(bytes));
            p += WIDTH;
        }
        Ok(total)
    } else if wire_type == expected_wt {
        if buf.len() < WIDTH {
            return Err(decode_error("unexpected EOF"));
        }
        let mut bytes = [0u8; WIDTH];
        bytes.copy_from_slice(&buf[..WIDTH]);
        builder.append_value(convert(bytes));
        Ok(WIDTH)
    } else {
        skip_field(wire_type, buf)
    }
}

fn decode_repeated_wrapper_varint<T: ArrowPrimitiveType>(
    wire_type: u8,
    buf: &[u8],
    builder: &mut PrimitiveBuilder<T>,
    convert: fn(u64) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if wire_type != 2 {
        return skip_field(wire_type, buf);
    }
    let (data, total) = read_length_delimited(buf)?;
    let (v, _) = decode_wrapper_varint(data)?;
    builder.append_value(convert(v));
    Ok(total)
}

fn decode_repeated_wrapper_fixed32<T: ArrowPrimitiveType>(
    wire_type: u8,
    buf: &[u8],
    builder: &mut PrimitiveBuilder<T>,
    convert: fn([u8; 4]) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if wire_type != 2 {
        return skip_field(wire_type, buf);
    }
    let (data, total) = read_length_delimited(buf)?;
    let (bytes, _) = decode_wrapper_fixed32(data)?;
    builder.append_value(convert(bytes));
    Ok(total)
}

fn decode_repeated_wrapper_fixed64<T: ArrowPrimitiveType>(
    wire_type: u8,
    buf: &[u8],
    builder: &mut PrimitiveBuilder<T>,
    convert: fn([u8; 8]) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if wire_type != 2 {
        return skip_field(wire_type, buf);
    }
    let (data, total) = read_length_delimited(buf)?;
    let (bytes, _) = decode_wrapper_fixed64(data)?;
    builder.append_value(convert(bytes));
    Ok(total)
}

// ---------------------------------------------------------------------------
// MessageDecoder
// ---------------------------------------------------------------------------

/// One field of a [`MessageDecoder`].
struct DecoderState {
    decoder: FieldDecoder,
    field: FieldDescriptor,
    /// The decoder appended its value directly in the current row (see
    /// [`FieldDecoder::decode`]).
    has_value: bool,
}

pub struct MessageDecoder {
    /// One per field, in field order.
    decoders: Vec<DecoderState>,
    tag_map: Vec<Option<usize>>,
    list_nullable: bool,
    map_nullable: bool,
    num_rows: usize,
}

impl MessageDecoder {
    pub fn new(descriptor: &MessageDescriptor, config: &PtarsConfig) -> Self {
        Self::with_capacity(descriptor, config, 0)
    }

    /// Like [`MessageDecoder::new`], with the builders of singular scalar
    /// fields sized for `capacity` rows.
    pub fn with_capacity(
        descriptor: &MessageDescriptor,
        config: &PtarsConfig,
        capacity: usize,
    ) -> Self {
        let mut decoders = Vec::new();
        let mut max_field_number: u32 = 0;

        for field in descriptor.fields() {
            if let Some(decoder) = build_field_decoder(&field, config, capacity) {
                if field.number() > max_field_number {
                    max_field_number = field.number();
                }
                decoders.push(DecoderState {
                    decoder,
                    field,
                    has_value: false,
                });
            }
        }

        let mut tag_map = vec![
            None;
            if max_field_number == 0 {
                0
            } else {
                max_field_number as usize + 1
            }
        ];
        for (idx, DecoderState { field, .. }) in decoders.iter().enumerate() {
            let num = field.number() as usize;
            tag_map[num] = Some(idx);
        }

        Self {
            decoders,
            tag_map,
            list_nullable: config.list_nullable,
            map_nullable: config.map_nullable,
            num_rows: 0,
        }
    }

    fn decode_row(&mut self, buf: &[u8]) -> Result<(), prost::DecodeError> {
        let mut pos = 0;
        while pos < buf.len() {
            let (field_num, wire_type, n) = decode_tag(&buf[pos..])?;
            pos += n;
            let idx = if (field_num as usize) < self.tag_map.len() {
                self.tag_map[field_num as usize]
            } else {
                None
            };
            if let Some(idx) = idx {
                let DecoderState {
                    decoder, has_value, ..
                } = &mut self.decoders[idx];
                pos += decoder.decode(wire_type, &buf[pos..], has_value)?;
            } else {
                pos += skip_field(wire_type, &buf[pos..])?;
            }
        }
        self.flush_unseen();
        self.num_rows += 1;
        Ok(())
    }

    /// Decode a submessage's bytes without flushing — used for singular message fields
    /// where the parent will call flush.
    fn decode_message_bytes(&mut self, buf: &[u8]) -> Result<(), prost::DecodeError> {
        let mut pos = 0;
        while pos < buf.len() {
            let (field_num, wire_type, n) = decode_tag(&buf[pos..])?;
            pos += n;
            let idx = if (field_num as usize) < self.tag_map.len() {
                self.tag_map[field_num as usize]
            } else {
                None
            };
            if let Some(idx) = idx {
                let DecoderState {
                    decoder, has_value, ..
                } = &mut self.decoders[idx];
                pos += decoder.decode(wire_type, &buf[pos..], has_value)?;
            } else {
                pos += skip_field(wire_type, &buf[pos..])?;
            }
        }
        Ok(())
    }

    /// Ends a row: flushes every decoder that did not append its value
    /// directly (see [`FieldDecoder::decode`]) and clears `has_value`.
    fn flush_unseen(&mut self) {
        for DecoderState {
            decoder, has_value, ..
        } in &mut self.decoders
        {
            if !*has_value {
                decoder.flush();
            }
            *has_value = false;
        }
    }

    fn flush_row(&mut self) {
        self.flush_unseen();
    }

    fn flush_defaults(&mut self) {
        self.flush_unseen(); // all absent → defaults
    }

    fn decode_null_row(&mut self) {
        self.flush_unseen();
        self.num_rows += 1;
    }

    fn row_count(&self) -> usize {
        self.num_rows
    }

    fn build_struct_array(&mut self, validity: Option<arrow_array::BooleanArray>) -> StructArray {
        if self.decoders.is_empty() {
            let len = validity.as_ref().map_or(self.num_rows, |v| v.len());
            return StructArray::new_empty_fields(
                len,
                validity.map(|v| arrow::buffer::NullBuffer::new(v.values().clone())),
            );
        }

        let (fields, columns): (Vec<_>, Vec<_>) = self
            .decoders
            .iter_mut()
            .map(
                |DecoderState {
                     decoder,
                     field: field_desc,
                     ..
                 }| {
                    let nullable = if field_desc.is_list() {
                        self.list_nullable
                    } else if field_desc.is_map() {
                        self.map_nullable
                    } else {
                        field_desc.supports_presence()
                    };
                    let (_, array) = decoder.finish(nullable);
                    let field = Field::new(field_desc.name(), array.data_type().clone(), nullable);
                    (field, array)
                },
            )
            .unzip();

        StructArray::new(
            arrow_schema::Fields::from(fields),
            columns,
            validity.map(|v| arrow::buffer::NullBuffer::new(v.values().clone())),
        )
    }

    pub fn finish(mut self) -> RecordBatch {
        if self.decoders.is_empty() {
            let schema = Arc::new(arrow_schema::Schema::empty());
            return RecordBatch::try_new_with_options(
                schema,
                vec![],
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(self.num_rows)),
            )
            .unwrap();
        }
        let struct_array = self.build_struct_array(None);
        RecordBatch::from(struct_array)
    }
}

// ---------------------------------------------------------------------------
// build_field_decoder
// ---------------------------------------------------------------------------

/// `capacity`: rows to size scalar builders for (0 when unknown).
fn build_field_decoder(
    field: &FieldDescriptor,
    config: &PtarsConfig,
    capacity: usize,
) -> Option<FieldDecoder> {
    if field.is_map() {
        return build_map_decoder(field, config);
    }
    if field.is_list() {
        return build_repeated_decoder(field, config);
    }

    let has_presence = field.supports_presence();
    match field.kind() {
        Kind::Int32 => Some(FieldDecoder::Int32 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Int64 => Some(FieldDecoder::Int64 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Uint32 => Some(FieldDecoder::UInt32 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Uint64 => Some(FieldDecoder::UInt64 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sint32 => Some(FieldDecoder::Sint32 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sint64 => Some(FieldDecoder::Sint64 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sfixed32 => Some(FieldDecoder::Sfixed32 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sfixed64 => Some(FieldDecoder::Sfixed64 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Fixed32 => Some(FieldDecoder::Fixed32 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Fixed64 => Some(FieldDecoder::Fixed64 {
            value: 0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Float => Some(FieldDecoder::Float {
            value: 0.0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Double => Some(FieldDecoder::Double {
            value: 0.0,
            has_value: false,
            has_presence,
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Bool => Some(FieldDecoder::Bool {
            value: false,
            has_value: false,
            has_presence,
            builder: BooleanBuilder::with_capacity(capacity),
        }),
        Kind::String => Some(FieldDecoder::String {
            has_presence,
            builder: StringColumn::with_capacity(config.use_large_string, capacity),
        }),
        Kind::Bytes => Some(FieldDecoder::Bytes {
            has_presence,
            builder: ByteColumn::with_capacity(config.use_large_binary, capacity),
        }),
        Kind::Enum(enum_desc) => match config.enum_repr {
            EnumRepr::Int32 => Some(FieldDecoder::EnumInt32 {
                value: 0,
                has_value: false,
                has_presence,
                builder: PrimitiveBuilder::with_capacity(capacity),
            }),
            EnumRepr::String => Some(FieldDecoder::EnumString {
                value: 0,
                has_value: false,
                has_presence,
                builder: StringBuilderInner::new(config.use_large_string),
                enum_descriptor: enum_desc,
            }),
            EnumRepr::Binary => Some(FieldDecoder::EnumBinary {
                value: 0,
                has_value: false,
                has_presence,
                builder: BinaryBuilderInner::new(config.use_large_binary),
                enum_descriptor: enum_desc,
            }),
        },
        Kind::Message(msg_desc) => build_message_field_decoder(msg_desc, config),
    }
}

fn build_message_field_decoder(
    msg_desc: MessageDescriptor,
    config: &PtarsConfig,
) -> Option<FieldDecoder> {
    match msg_desc.full_name() {
        "google.protobuf.Timestamp" => Some(FieldDecoder::Timestamp {
            seconds: 0,
            nanos: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
            unit: config.timestamp_unit.into(),
            tz: config.timestamp_tz.clone(),
        }),
        "google.protobuf.Duration" => Some(FieldDecoder::Duration {
            seconds: 0,
            nanos: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
            unit: config.duration_unit.into(),
        }),
        "google.type.Date" => Some(FieldDecoder::Date {
            year: 0,
            month: 0,
            day: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.type.TimeOfDay" => Some(FieldDecoder::TimeOfDay {
            hours: 0,
            minutes: 0,
            seconds_val: 0,
            nanos: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
            unit: config.time_unit.into(),
        }),
        "google.protobuf.DoubleValue" => Some(FieldDecoder::WrapperDouble {
            value: 0.0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.FloatValue" => Some(FieldDecoder::WrapperFloat {
            value: 0.0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.Int64Value" => Some(FieldDecoder::WrapperInt64 {
            value: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.UInt64Value" => Some(FieldDecoder::WrapperUInt64 {
            value: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.Int32Value" => Some(FieldDecoder::WrapperInt32 {
            value: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.UInt32Value" => Some(FieldDecoder::WrapperUInt32 {
            value: 0,
            has_value: false,
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.BoolValue" => Some(FieldDecoder::WrapperBool {
            value: false,
            has_value: false,
            builder: BooleanBuilder::new(),
        }),
        "google.protobuf.StringValue" => Some(FieldDecoder::WrapperString {
            value: Vec::new(),
            has_value: false,
            builder: StringBuilderInner::new(config.use_large_string),
        }),
        "google.protobuf.BytesValue" => Some(FieldDecoder::WrapperBytes {
            value: Vec::new(),
            has_value: false,
            builder: BinaryBuilderInner::new(config.use_large_binary),
        }),
        _ => {
            let sub_decoder = MessageDecoder::new(&msg_desc, config);
            Some(FieldDecoder::Message {
                sub_decoder,
                has_value: false,
                is_valid: BooleanBuilder::new(),
            })
        }
    }
}

fn build_repeated_decoder(field: &FieldDescriptor, config: &PtarsConfig) -> Option<FieldDecoder> {
    let ln = config.list_value_name.clone();
    let lnb = config.list_value_nullable;
    let offsets = || ListOffsets::new(config.use_large_list);

    let inner = match field.kind() {
        Kind::Int32 => RepeatedInner::Int32 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Sint32 => RepeatedInner::Sint32 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Sfixed32 => RepeatedInner::Sfixed32 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Int64 => RepeatedInner::Int64 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Sint64 => RepeatedInner::Sint64 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Sfixed64 => RepeatedInner::Sfixed64 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Uint32 => RepeatedInner::UInt32 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Fixed32 => RepeatedInner::Fixed32 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Uint64 => RepeatedInner::UInt64 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Fixed64 => RepeatedInner::Fixed64 {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Float => RepeatedInner::Float {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Double => RepeatedInner::Double {
            values_builder: PrimitiveBuilder::new(),
        },
        Kind::Bool => RepeatedInner::Bool {
            values_builder: BooleanBuilder::new(),
        },
        Kind::String => RepeatedInner::String {
            values_builder: StringBuilderInner::new(config.use_large_string),
        },
        Kind::Bytes => RepeatedInner::Bytes {
            values_builder: BinaryBuilderInner::new(config.use_large_binary),
        },
        Kind::Enum(enum_desc) => match config.enum_repr {
            EnumRepr::Int32 => RepeatedInner::EnumInt32 {
                values_builder: PrimitiveBuilder::new(),
            },
            EnumRepr::String => RepeatedInner::EnumString {
                values_builder: StringBuilderInner::new(config.use_large_string),
                enum_descriptor: enum_desc,
            },
            EnumRepr::Binary => RepeatedInner::EnumBinary {
                values_builder: BinaryBuilderInner::new(config.use_large_binary),
                enum_descriptor: enum_desc,
            },
        },
        Kind::Message(msg_desc) => {
            return build_repeated_message_decoder(&msg_desc, config, offsets(), ln, lnb);
        }
    };

    Some(FieldDecoder::Repeated {
        inner,
        offsets: offsets(),
        list_name: ln,
        list_nullable: lnb,
    })
}

fn build_repeated_message_decoder(
    msg_desc: &MessageDescriptor,
    config: &PtarsConfig,
    offsets: ListOffsets,
    ln: Arc<str>,
    lnb: bool,
) -> Option<FieldDecoder> {
    let inner = match msg_desc.full_name() {
        "google.protobuf.Timestamp" => RepeatedInner::Timestamp {
            values_builder: PrimitiveBuilder::new(),
            unit: config.timestamp_unit.into(),
            tz: config.timestamp_tz.clone(),
        },
        "google.protobuf.Duration" => RepeatedInner::Duration {
            values_builder: PrimitiveBuilder::new(),
            unit: config.duration_unit.into(),
        },
        "google.type.Date" => RepeatedInner::Date {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.type.TimeOfDay" => RepeatedInner::TimeOfDay {
            values_builder: PrimitiveBuilder::new(),
            unit: config.time_unit.into(),
        },
        "google.protobuf.DoubleValue" => RepeatedInner::WrapperDouble {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.FloatValue" => RepeatedInner::WrapperFloat {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.Int64Value" => RepeatedInner::WrapperInt64 {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.UInt64Value" => RepeatedInner::WrapperUInt64 {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.Int32Value" => RepeatedInner::WrapperInt32 {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.UInt32Value" => RepeatedInner::WrapperUInt32 {
            values_builder: PrimitiveBuilder::new(),
        },
        "google.protobuf.BoolValue" => RepeatedInner::WrapperBool {
            values_builder: BooleanBuilder::new(),
        },
        "google.protobuf.StringValue" => RepeatedInner::WrapperString {
            values_builder: StringBuilderInner::new(config.use_large_string),
        },
        "google.protobuf.BytesValue" => RepeatedInner::WrapperBytes {
            values_builder: BinaryBuilderInner::new(config.use_large_binary),
        },
        _ => {
            let sub_decoder = MessageDecoder::new(msg_desc, config);
            RepeatedInner::Message { sub_decoder }
        }
    };

    Some(FieldDecoder::Repeated {
        inner,
        offsets,
        list_name: ln,
        list_nullable: lnb,
    })
}

fn build_map_decoder(field: &FieldDescriptor, config: &PtarsConfig) -> Option<FieldDecoder> {
    let map_entry = match field.kind() {
        Kind::Message(desc) => desc,
        _ => return None,
    };
    let key_field = map_entry.get_field_by_name("key")?;
    let value_field = map_entry.get_field_by_name("value")?;

    // Build singular decoders for key and value (they buffer per-entry, not per-row)
    let key_decoder = build_singular_decoder_for_map(&key_field, config)?;
    let value_decoder = build_singular_decoder_for_map(&value_field, config)?;

    Some(FieldDecoder::Map {
        key_decoder: Box::new(key_decoder),
        value_decoder: Box::new(value_decoder),
        offsets: vec![0],
        map_value_name: config.map_value_name.clone(),
        map_value_nullable: config.map_value_nullable,
    })
}

/// Build a singular decoder for use inside a map entry (no presence tracking).
fn build_singular_decoder_for_map(
    field: &FieldDescriptor,
    config: &PtarsConfig,
) -> Option<FieldDecoder> {
    // Map keys/values are never "optional" in protobuf sense — they use proto3 defaults
    match field.kind() {
        Kind::Int32 => Some(FieldDecoder::Int32 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Int64 => Some(FieldDecoder::Int64 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Uint32 => Some(FieldDecoder::UInt32 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Uint64 => Some(FieldDecoder::UInt64 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Sint32 => Some(FieldDecoder::Sint32 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Sint64 => Some(FieldDecoder::Sint64 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Sfixed32 => Some(FieldDecoder::Sfixed32 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Sfixed64 => Some(FieldDecoder::Sfixed64 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Fixed32 => Some(FieldDecoder::Fixed32 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Fixed64 => Some(FieldDecoder::Fixed64 {
            value: 0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Float => Some(FieldDecoder::Float {
            value: 0.0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Double => Some(FieldDecoder::Double {
            value: 0.0,
            has_value: false,
            has_presence: false,
            builder: PrimitiveBuilder::new(),
        }),
        Kind::Bool => Some(FieldDecoder::Bool {
            value: false,
            has_value: false,
            has_presence: false,
            builder: BooleanBuilder::new(),
        }),
        Kind::String => Some(FieldDecoder::String {
            has_presence: false,
            builder: StringColumn::new(config.use_large_string),
        }),
        Kind::Bytes => Some(FieldDecoder::Bytes {
            has_presence: false,
            builder: ByteColumn::new(config.use_large_binary),
        }),
        Kind::Enum(enum_desc) => match config.enum_repr {
            EnumRepr::Int32 => Some(FieldDecoder::EnumInt32 {
                value: 0,
                has_value: false,
                has_presence: false,
                builder: PrimitiveBuilder::new(),
            }),
            EnumRepr::String => Some(FieldDecoder::EnumString {
                value: 0,
                has_value: false,
                has_presence: false,
                builder: StringBuilderInner::new(config.use_large_string),
                enum_descriptor: enum_desc,
            }),
            EnumRepr::Binary => Some(FieldDecoder::EnumBinary {
                value: 0,
                has_value: false,
                has_presence: false,
                builder: BinaryBuilderInner::new(config.use_large_binary),
                enum_descriptor: enum_desc,
            }),
        },
        Kind::Message(msg_desc) => build_message_field_decoder(msg_desc, config),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decode a BinaryArray of serialized protobuf messages directly into a RecordBatch.
///
/// Parses protobuf wire format directly into Arrow builders — no intermediate
/// message objects are created.
pub fn binary_array_to_record_batch_direct(
    array: &BinaryArray,
    descriptor: &MessageDescriptor,
    config: &PtarsConfig,
) -> Result<RecordBatch, prost::DecodeError> {
    let mut decoder = MessageDecoder::with_capacity(descriptor, config, array.len());
    let policy = config.confluent_wire_policy;
    for i in 0..array.len() {
        if array.is_null(i) {
            decoder.decode_null_row();
        } else {
            let bytes = strip_confluent_prefix(array.value(i), policy)?;
            decoder.decode_row(bytes)?;
        }
    }
    Ok(decoder.finish())
}

/// Convert DynamicMessage instances to a RecordBatch using the default configuration.
///
/// Each message is serialized to protobuf wire format, then decoded directly
/// into Arrow arrays.
pub fn messages_to_record_batch(
    messages: &[prost_reflect::DynamicMessage],
    message_descriptor: &MessageDescriptor,
) -> RecordBatch {
    messages_to_record_batch_with_config(messages, message_descriptor, &PtarsConfig::default())
}

/// Convert DynamicMessage instances to a RecordBatch using the specified configuration.
///
/// Each message is serialized to protobuf wire format, then decoded directly
/// into Arrow arrays.
pub fn messages_to_record_batch_with_config(
    messages: &[prost_reflect::DynamicMessage],
    message_descriptor: &MessageDescriptor,
    config: &PtarsConfig,
) -> RecordBatch {
    use arrow_array::builder::BinaryBuilder;
    use prost::Message;

    let mut bin_builder = BinaryBuilder::new();
    for msg in messages {
        bin_builder.append_value(msg.encode_to_vec());
    }
    let binary_array = bin_builder.finish();
    binary_array_to_record_batch_direct(&binary_array, message_descriptor, config)
        .expect("failed to decode messages")
}

/// Decode a BinaryArray into a vector of DynamicMessage.
///
/// Each element in the binary array is expected to be a serialized protobuf message.
/// Null values in the array result in default (empty) messages.
pub fn binary_array_to_messages(
    array: &BinaryArray,
    message_descriptor: &MessageDescriptor,
) -> Result<Vec<prost_reflect::DynamicMessage>, prost::DecodeError> {
    let mut messages = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let message = if array.is_null(i) {
            prost_reflect::DynamicMessage::new(message_descriptor.clone())
        } else {
            prost_reflect::DynamicMessage::decode(message_descriptor.clone(), array.value(i))?
        };
        messages.push(message);
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_confluent_prefix_raw() {
        let buf = b"\x00\x00\x00\x00\x01\x08\x96\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Raw).unwrap();
        assert_eq!(result, buf);
    }

    #[test]
    fn test_strip_confluent_prefix_standard() {
        // magic byte + 4-byte schema ID + payload
        let buf = b"\x00\x00\x00\x00\x01\x08\x96\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Standard).unwrap();
        assert_eq!(result, b"\x08\x96\x01");
    }

    #[test]
    fn test_strip_confluent_prefix_standard_too_short() {
        let buf = b"\x00\x01\x02";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Standard);
        assert!(result.is_err());
    }

    #[test]
    fn test_strip_confluent_prefix_protobuf_zero_indexes() {
        // magic byte + 4-byte schema ID + varint 0 (count=0) + payload
        let buf = b"\x00\x00\x00\x00\x01\x00\x08\x96\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Protobuf).unwrap();
        assert_eq!(result, b"\x08\x96\x01");
    }

    #[test]
    fn test_strip_confluent_prefix_protobuf_one_index() {
        // magic byte + 4-byte schema ID + varint 1 (count=1) + varint 0 (index) + payload
        let buf = b"\x00\x00\x00\x00\x01\x01\x00\x08\x96\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Protobuf).unwrap();
        assert_eq!(result, b"\x08\x96\x01");
    }

    #[test]
    fn test_strip_confluent_prefix_protobuf_two_indexes() {
        // magic byte + 4-byte schema ID + varint 2 (count) + varint 4 + varint 2 + payload
        let buf = b"\x00\x00\x00\x00\x01\x02\x04\x02\x08\x96\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Protobuf).unwrap();
        assert_eq!(result, b"\x08\x96\x01");
    }

    #[test]
    fn test_strip_confluent_prefix_protobuf_too_short() {
        let buf = b"\x00\x01";
        let result = strip_confluent_prefix(buf, ConfluentWirePolicy::Protobuf);
        assert!(result.is_err());
    }

    #[test]
    fn validate_utf8_matches_std() {
        let cases: [&[u8]; 9] = [
            b"",
            b"EUR/USD",
            "caf\u{e9}".as_bytes(),
            "\u{65e5}\u{672c} long enough to span words".as_bytes(),
            &[0xff],
            &[b'a', 0xc3],
            &[0xc0, 0x80],
            &[0xed, 0xa0, 0x80],
            b"ascii prefix then \xe2\x82",
        ];
        for case in cases {
            assert_eq!(
                validate_utf8(case).ok(),
                std::str::from_utf8(case).ok(),
                "{case:?}"
            );
        }
    }

    #[test]
    fn decode_varint_matches_loop() {
        let mut values: Vec<u64> = vec![0, 1, 127, 128, u64::MAX, 1 << 63];
        for bits in 0..64 {
            values.extend([(1u64 << bits) - 1, 1u64 << bits, (1u64 << bits) + 1]);
        }
        values.push(1_731_700_000_000_000_000);
        for v in values {
            let mut enc = Vec::new();
            prost::encoding::encode_varint(v, &mut enc);
            // Exact length, a few trailing bytes (fewer than 8 in all take the
            // byte loop) or many, and every truncation.
            let mut short = enc.clone();
            short.extend_from_slice(&[0xff; 2]);
            let mut padded = enc.clone();
            padded.extend_from_slice(&[0xff; 12]);
            for buf in [&enc[..], &short[..], &padded[..]] {
                assert_eq!(decode_varint(buf).unwrap(), (v, enc.len()), "{v}");
                assert_eq!(decode_varint_loop(buf).unwrap(), (v, enc.len()), "{v}");
            }
            for cut in 0..enc.len() {
                assert!(decode_varint(&enc[..cut]).is_err(), "{v} cut {cut}");
            }
        }
        // Overlong: more than 10 bytes with the continuation bit set.
        assert!(decode_varint(&[0xff; 11]).is_err());
        assert!(decode_varint_loop(&[0xff; 11]).is_err());
    }
}
