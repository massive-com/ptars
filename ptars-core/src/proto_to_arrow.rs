use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::buffer::Buffer;
use arrow_array::builder::ArrayBuilder;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, LargeBinaryBuilder, LargeStringBuilder, PrimitiveBuilder,
    StringBuilder,
};
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrowPrimitiveType, BinaryArray, MapArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use chrono::Datelike;

use prost_reflect::{EnumDescriptor, FieldDescriptor, Kind, MessageDescriptor};

use crate::config::{ConfluentWirePolicy, EnumRepr, PtarsConfig};

mod column_builder;

use column_builder::{validate_utf8, ByteColumn, StringColumn};

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

fn decode_tag(buf: &[u8]) -> Result<(u32, u8, usize), prost::DecodeError> {
    let (key, n) = decode_varint(buf)?;
    let wire_type = (key & 0x07) as u8;
    let field_number = (key >> 3) as u32;
    if field_number == 0 {
        return Err(decode_error("invalid field number 0"));
    }
    Ok((field_number, wire_type, n))
}

#[cold]
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
        3 => skip_group(buf),
        4 => Err(decode_error("unexpected end-group tag")),
        _ => Err(decode_error("unsupported wire type")),
    }
}

/// Skip a group (wire type 3) whose start tag has already been consumed: read fields until
/// the matching end-group tag (wire type 4), recursing into nested groups. Groups are legal
/// on the wire (proto2 `group`, or any proto2 producer) and an unknown one must be skipped
/// like any other unknown field instead of failing the record.
fn skip_group(buf: &[u8]) -> Result<usize, prost::DecodeError> {
    let mut pos = 0;
    loop {
        if pos >= buf.len() {
            return Err(decode_error("unexpected EOF inside group"));
        }
        let (_, wire_type, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if wire_type == 4 {
            return Ok(pos);
        }
        pos += skip_field(wire_type, &buf[pos..])?;
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

/// Convert a `(seconds, nanos)` pair into a single `i64` in `unit`.
///
/// Saturates instead of panicking when the value does not fit. `google.protobuf.Timestamp`
/// spans years 0001..=9999 but Arrow `Timestamp(Nanosecond)` only 1677..=2262, and both ends
/// occur in real feeds: a far-future "never expires" sentinel (9999-12-31) and the far-past
/// null-date sentinel some producers emit (0001-01-01). The old `panic!("… overflow")` aborted
/// the decode of a whole batch over one such record. Clamping keeps the sentinel's ordering
/// (a clamped 9999-12-31 still sorts after every real timestamp), which a null would not.
fn convert_seconds_nanos_to_unit(seconds: i64, nanos: i32, unit: TimeUnit) -> i64 {
    let (mul, div): (i128, i128) = match unit {
        TimeUnit::Second => (1, 1_000_000_000),
        TimeUnit::Millisecond => (1_000, 1_000_000),
        TimeUnit::Microsecond => (1_000_000, 1_000),
        TimeUnit::Nanosecond => (1_000_000_000, 1),
    };
    let v = i128::from(seconds) * mul + i128::from(nanos) / div;
    v.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

static CE_OFFSET: i32 = 719163;

/// `google.type.Date` → days since 1970-01-01 (Arrow `Date32`).
///
/// The type deliberately allows partial dates: year 0 means "no year", month 0 "no month and
/// day", day 0 "whole month". Unspecified month/day are normalised to 1, so "2026, month
/// unspecified" becomes 2026-01-01. A triple that still is not a calendar date (no year,
/// 2026-02-30, month 13) falls back to 0 — the epoch — where the all-zero date already went.
/// The previous `from_ymd_opt(...).unwrap()` panicked on every one of these and took the batch
/// down with it.
fn date_to_days_from_epoch(year: i32, month: i32, day: i32) -> i32 {
    if year == 0 {
        return 0;
    }
    let month = if month == 0 { 1 } else { month };
    let day = if day == 0 { 1 } else { day };
    match (u32::try_from(month), u32::try_from(day)) {
        (Ok(m), Ok(d)) => chrono::NaiveDate::from_ymd_opt(year, m, d)
            .map(|date| date.num_days_from_ce() - CE_OFFSET)
            .unwrap_or(0),
        _ => 0,
    }
}

/// Precomputed number → name table for one enum, built once when the field decoder is
/// constructed.
///
/// Looking the name up through `EnumDescriptor::get_value` and allocating a fresh `String` for
/// every decoded value was the hot path of string-represented enums on busy topics. The table
/// returns a borrowed `&str` for declared values; numbers outside the declared set — which
/// protobuf permits — are formatted on the fly, as before.
struct EnumNames {
    by_number: HashMap<i32, String>,
    /// The enum's first value, for an absent field without presence.
    default: i32,
}

impl EnumNames {
    fn new(descriptor: &EnumDescriptor) -> Self {
        let by_number = descriptor
            .values()
            .map(|v| (v.number(), v.name().to_string()))
            .collect();
        Self {
            by_number,
            default: descriptor.default_value().number(),
        }
    }

    /// The name of the enum's first value, which in proto2 needn't be 0.
    fn default_name(&self) -> Cow<'_, str> {
        self.name(self.default)
    }

    fn name(&self, number: i32) -> Cow<'_, str> {
        match self.by_number.get(&number) {
            Some(name) => Cow::Borrowed(name.as_str()),
            None => Cow::Owned(number.to_string()),
        }
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
        enum_names: EnumNames,
    },
    EnumBinary {
        values_builder: BinaryBuilderInner,
        enum_names: EnumNames,
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
                enum_names,
                ..
            } => {
                if wire_type == 2 {
                    let (data, total) = read_length_delimited(buf)?;
                    let mut p = 0;
                    while p < data.len() {
                        let (v, n) = decode_varint(&data[p..])?;
                        values_builder.append_value(enum_names.name(v as i32).as_ref());
                        p += n;
                    }
                    Ok(total)
                } else if wire_type == 0 {
                    let (v, n) = decode_varint(buf)?;
                    values_builder.append_value(enum_names.name(v as i32).as_ref());
                    Ok(n)
                } else {
                    skip_field(wire_type, buf)
                }
            }
            Self::EnumBinary {
                values_builder,
                enum_names,
                ..
            } => {
                if wire_type == 2 {
                    let (data, total) = read_length_delimited(buf)?;
                    let mut p = 0;
                    while p < data.len() {
                        let (v, n) = decode_varint(&data[p..])?;
                        values_builder.append_value(enum_names.name(v as i32).as_bytes());
                        p += n;
                    }
                    Ok(total)
                } else if wire_type == 0 {
                    let (v, n) = decode_varint(buf)?;
                    values_builder.append_value(enum_names.name(v as i32).as_bytes());
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
                    values_builder.append_value(date_to_days_from_epoch(y, m, d));
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
                values_builder.append_value(decode_wrapper_string(data)?.unwrap_or_default());
                Ok(total)
            }
            Self::WrapperBytes { values_builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                values_builder.append_value(decode_wrapper_bytes(data)?.unwrap_or_default());
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
            Self::Message { sub_decoder, .. } => {
                Arc::new(sub_decoder.build_struct_array(None, false))
            }
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
    // --- Singular scalars ---
    // Appended directly in `decode`, except `Bool`, which buffers `value`
    // until `flush`.
    Int32 {
        builder: PrimitiveBuilder<Int32Type>,
    },
    Int64 {
        builder: PrimitiveBuilder<Int64Type>,
    },
    UInt32 {
        builder: PrimitiveBuilder<UInt32Type>,
    },
    UInt64 {
        builder: PrimitiveBuilder<UInt64Type>,
    },
    Sint32 {
        builder: PrimitiveBuilder<Int32Type>,
    },
    Sint64 {
        builder: PrimitiveBuilder<Int64Type>,
    },
    Sfixed32 {
        builder: PrimitiveBuilder<Int32Type>,
    },
    Sfixed64 {
        builder: PrimitiveBuilder<Int64Type>,
    },
    Fixed32 {
        builder: PrimitiveBuilder<UInt32Type>,
    },
    Fixed64 {
        builder: PrimitiveBuilder<UInt64Type>,
    },
    Float {
        builder: PrimitiveBuilder<Float32Type>,
    },
    Double {
        builder: PrimitiveBuilder<Float64Type>,
    },
    Bool {
        value: bool,
        builder: BooleanBuilder,
    },
    String {
        builder: StringColumn,
    },
    Bytes {
        builder: ByteColumn,
    },
    EnumInt32 {
        /// The enum's first value, for an absent field without presence.
        default: i32,
        builder: PrimitiveBuilder<Int32Type>,
    },
    EnumString {
        builder: StringColumn,
        enum_names: EnumNames,
    },
    EnumBinary {
        builder: ByteColumn,
        enum_names: EnumNames,
    },

    // --- Well-known types (singular, buffered) ---
    Timestamp {
        seconds: i64,
        nanos: i32,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
        tz: Option<Arc<str>>,
    },
    Duration {
        seconds: i64,
        nanos: i32,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },
    Date {
        year: i32,
        month: i32,
        day: i32,
        builder: PrimitiveBuilder<Date32Type>,
    },
    TimeOfDay {
        hours: i32,
        minutes: i32,
        seconds_val: i32,
        nanos: i32,
        builder: PrimitiveBuilder<Int64Type>,
        unit: TimeUnit,
    },

    // --- Wrapper types (singular) ---
    // Appended directly in `decode`, except `WrapperBool`, which buffers
    // `value` until `flush`.
    WrapperDouble {
        builder: PrimitiveBuilder<Float64Type>,
    },
    WrapperFloat {
        builder: PrimitiveBuilder<Float32Type>,
    },
    WrapperInt64 {
        builder: PrimitiveBuilder<Int64Type>,
    },
    WrapperUInt64 {
        builder: PrimitiveBuilder<UInt64Type>,
    },
    WrapperInt32 {
        builder: PrimitiveBuilder<Int32Type>,
    },
    WrapperUInt32 {
        builder: PrimitiveBuilder<UInt32Type>,
    },
    WrapperBool {
        value: bool,
        builder: BooleanBuilder,
    },
    WrapperString {
        builder: StringColumn,
    },
    WrapperBytes {
        builder: ByteColumn,
    },

    // --- Nested message ---
    Message {
        sub_decoder: Box<MessageDecoder>,
        is_valid: BooleanBuilder,
    },

    // --- Repeated fields (all collapsed into one variant) ---
    Repeated {
        inner: Box<RepeatedInner>,
        offsets: ListOffsets,
        list_name: Arc<str>,
        list_nullable: bool,
    },

    // --- Map fields ---
    Map {
        key_decoder: Box<FieldDecoder>,
        value_decoder: Box<FieldDecoder>,
        /// What each entry's key and value flags start from. Map keys and
        /// values have no presence, even in proto2.
        key_flags: FieldFlags,
        value_flags: FieldFlags,
        offsets: Vec<i32>,
        /// Entries decoded in the current row; [`FieldDecoder::flush`] adds
        /// them to `offsets`.
        row_entries: i32,
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

/// The last `value` field of a `StringValue` message, if any, borrowed from
/// `buf` and checked to be UTF-8.
fn decode_wrapper_string(buf: &[u8]) -> Result<Option<&str>, prost::DecodeError> {
    let mut val = None;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 2 {
            let (data, consumed) = read_length_delimited(&buf[pos..])?;
            val = Some(validate_utf8(data)?);
            pos += consumed;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok(val)
}

/// The last `value` field of a `BytesValue` message, if any, borrowed from
/// `buf`.
fn decode_wrapper_bytes(buf: &[u8]) -> Result<Option<&[u8]>, prost::DecodeError> {
    let mut val = None;
    let mut pos = 0;
    while pos < buf.len() {
        let (fnum, wt, n) = decode_tag(&buf[pos..])?;
        pos += n;
        if fnum == 1 && wt == 2 {
            let (data, consumed) = read_length_delimited(&buf[pos..])?;
            val = Some(data);
            pos += consumed;
        } else {
            pos += skip_field(wt, &buf[pos..])?;
        }
    }
    Ok(val)
}

// ---------------------------------------------------------------------------
// Macro for flush/finish boilerplate
// ---------------------------------------------------------------------------

/// Appends `value`, or replaces the value appended earlier in this row when
/// the field occurs again ([`FieldFlags::HAS_VALUE`]).
#[inline(always)]
fn push_or_replace<T: ArrowPrimitiveType>(
    builder: &mut PrimitiveBuilder<T>,
    value: T::Native,
    flags: &mut FieldFlags,
) {
    if flags.contains(FieldFlags::HAS_VALUE) {
        *builder.values_slice_mut().last_mut().unwrap() = value;
    } else {
        builder.append_value(value);
        flags.insert(FieldFlags::HAS_VALUE);
    }
}

#[inline(always)]
fn push_or_replace_with<T: ArrowPrimitiveType, const N: usize>(
    builder: &mut PrimitiveBuilder<T>,
    buf: &[u8],
    flags: &mut FieldFlags,
    f: impl FnOnce([u8; N]) -> T::Native,
) -> Result<usize, prost::DecodeError> {
    if buf.len() < N {
        return Err(decode_error("unexpected EOF"));
    }
    push_or_replace(builder, (f)(buf[..N].try_into().unwrap()), flags);
    Ok(N)
}

/// Appends one occurrence of a wrapper field: its value replaces the row's,
/// as in [`push_or_replace`]; without one, the row gets the default unless it
/// already has a value, as when protobuf merges the messages.
#[inline(always)]
fn push_wrapper<T: ArrowPrimitiveType>(
    builder: &mut PrimitiveBuilder<T>,
    value: Option<T::Native>,
    flags: &mut FieldFlags,
) {
    match value {
        Some(v) => push_or_replace(builder, v, flags),
        None if !flags.contains(FieldFlags::HAS_VALUE) => {
            push_or_replace(builder, T::Native::default(), flags)
        }
        None => {}
    }
}

/// Ends a row for a directly appended primitive field that was absent from
/// the message: null where the field has presence, the default otherwise.
fn append_absent<T: ArrowPrimitiveType>(builder: &mut PrimitiveBuilder<T>, has_presence: bool) {
    if has_presence {
        builder.append_null();
    } else {
        builder.append_value(T::Native::default());
    }
}

/// Finishes `builder`, leaving it sized for `capacity` rows (empty for 0).
fn finish_primitive<T: ArrowPrimitiveType>(
    builder: &mut PrimitiveBuilder<T>,
    capacity: usize,
) -> Arc<dyn Array> {
    let array = builder.finish();
    if capacity > 0 {
        *builder = PrimitiveBuilder::with_capacity(capacity);
    }
    Arc::new(array)
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
    /// Decodes one occurrence of this field and sets [`FieldFlags::HAS_VALUE`].
    /// Unless the field is [`FieldFlags::BUFFERED`], the value is appended
    /// directly, and a repeat in the same row replaces it.
    #[inline(always)]
    fn decode(
        &mut self,
        wire_type: u8,
        buf: &[u8],
        flags: &mut FieldFlags,
    ) -> Result<usize, prost::DecodeError> {
        match self {
            Self::Int32 { builder, .. } | Self::EnumInt32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as i32, flags);
                Ok(n)
            }
            Self::EnumString {
                builder,
                enum_names,
            } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                builder.push_or_replace(enum_names.name(v as i32).as_ref(), flags);
                Ok(n)
            }
            Self::EnumBinary {
                builder,
                enum_names,
            } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                builder.push_or_replace(enum_names.name(v as i32).as_bytes(), flags);
                Ok(n)
            }
            Self::Int64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as i64, flags);
                Ok(n)
            }
            Self::UInt32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v as u32, flags);
                Ok(n)
            }
            Self::UInt64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, v, flags);
                Ok(n)
            }
            Self::Sint32 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, decode_zigzag32(v), flags);
                Ok(n)
            }
            Self::Sint64 { builder, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                push_or_replace(builder, decode_zigzag64(v), flags);
                Ok(n)
            }
            Self::Sfixed32 { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, i32::from_le_bytes)
            }
            Self::Sfixed64 { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, i64::from_le_bytes)
            }
            Self::Fixed32 { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, u32::from_le_bytes)
            }
            Self::Fixed64 { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, u64::from_le_bytes)
            }
            Self::Float { builder, .. } => {
                if wire_type != 5 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, f32::from_le_bytes)
            }
            Self::Double { builder, .. } => {
                if wire_type != 1 {
                    return skip_field(wire_type, buf);
                }
                push_or_replace_with(builder, buf, flags, f64::from_le_bytes)
            }
            Self::Bool { value, .. } => {
                if wire_type != 0 {
                    return skip_field(wire_type, buf);
                }
                let (v, n) = decode_varint(buf)?;
                *value = v != 0;
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(n)
            }
            Self::String { builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_or_replace(validate_utf8(data)?, flags);
                Ok(total)
            }
            Self::Bytes { builder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_or_replace(data, flags);
                Ok(total)
            }
            // Well-known types: decode submessage
            Self::Timestamp { seconds, nanos, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                *seconds = vals[0];
                *nanos = vals[1] as i32;
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(total)
            }
            Self::Duration { seconds, nanos, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 2)?;
                *seconds = vals[0];
                *nanos = vals[1] as i32;
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(total)
            }
            Self::Date {
                year, month, day, ..
            } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let vals = decode_wkt_submessage(data, 3)?;
                *year = vals[0] as i32;
                *month = vals[1] as i32;
                *day = vals[2] as i32;
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(total)
            }
            Self::TimeOfDay {
                hours,
                minutes,
                seconds_val,
                nanos,
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
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(total)
            }
            // Wrapper types
            Self::WrapperDouble { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (bytes, found) = decode_wrapper_fixed64(data)?;
                push_wrapper(builder, found.then(|| f64::from_le_bytes(bytes)), flags);
                Ok(total)
            }
            Self::WrapperFloat { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (bytes, found) = decode_wrapper_fixed32(data)?;
                push_wrapper(builder, found.then(|| f32::from_le_bytes(bytes)), flags);
                Ok(total)
            }
            Self::WrapperInt64 { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_varint(data)?;
                push_wrapper(builder, found.then_some(v as i64), flags);
                Ok(total)
            }
            Self::WrapperUInt64 { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_varint(data)?;
                push_wrapper(builder, found.then_some(v), flags);
                Ok(total)
            }
            Self::WrapperInt32 { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_varint(data)?;
                push_wrapper(builder, found.then_some(v as i32), flags);
                Ok(total)
            }
            Self::WrapperUInt32 { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_varint(data)?;
                push_wrapper(builder, found.then_some(v as u32), flags);
                Ok(total)
            }
            Self::WrapperBool { value, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                let (v, found) = decode_wrapper_varint(data)?;
                if found {
                    *value = v != 0;
                }
                flags.insert(FieldFlags::HAS_VALUE);
                Ok(total)
            }
            Self::WrapperString { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_wrapper(decode_wrapper_string(data)?, flags);
                Ok(total)
            }
            Self::WrapperBytes { builder } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                builder.push_wrapper(decode_wrapper_bytes(data)?, flags);
                Ok(total)
            }
            // Nested message
            Self::Message { sub_decoder, .. } => {
                if wire_type != 2 {
                    return skip_field(wire_type, buf);
                }
                let (data, total) = read_length_delimited(buf)?;
                // For singular messages, if seen multiple times, the spec says merge.
                // For simplicity (and matching DynamicMessage behavior), we decode fresh each time.
                // Reset sub_decoder if already decoded this row.
                flags.insert(FieldFlags::HAS_VALUE);
                sub_decoder.decode_nested(data)?;
                Ok(total)
            }
            // Repeated fields — delegate to inner
            Self::Repeated { inner, .. } => inner.decode(wire_type, buf),
            // Map: each occurrence is a length-delimited entry submessage
            Self::Map {
                key_decoder,
                value_decoder,
                key_flags,
                value_flags,
                row_entries,
                ..
            } => Self::decode_map_entry(
                wire_type,
                buf,
                key_decoder,
                value_decoder,
                *key_flags,
                *value_flags,
                row_entries,
            ),
        }
    }

    /// Decodes one entry of a map field. Out of line, since it decodes the
    /// key and the value with [`Self::decode`]. A key repeated in one row
    /// keeps every entry: protobuf keeps only the last, but builders can't
    /// drop an earlier entry.
    #[inline(never)]
    fn decode_map_entry(
        wire_type: u8,
        buf: &[u8],
        key_decoder: &mut FieldDecoder,
        value_decoder: &mut FieldDecoder,
        key_start: FieldFlags,
        value_start: FieldFlags,
        row_entries: &mut i32,
    ) -> Result<usize, prost::DecodeError> {
        if wire_type != 2 {
            return skip_field(wire_type, buf);
        }
        let (data, total) = read_length_delimited(buf)?;
        // Parse entry submessage: field 1 = key, field 2 = value
        let mut pos = 0;
        let (mut key_flags, mut value_flags) = (key_start, value_start);
        let entry = (|| -> Result<(), prost::DecodeError> {
            while pos < data.len() {
                let (fnum, wt, n) = decode_tag(&data[pos..])?;
                pos += n;
                if fnum == 1 {
                    pos += key_decoder.decode(wt, &data[pos..], &mut key_flags)?;
                } else if fnum == 2 {
                    pos += value_decoder.decode(wt, &data[pos..], &mut value_flags)?;
                } else {
                    pos += skip_field(wt, &data[pos..])?;
                }
            }
            Ok(())
        })();
        // Complete the entry even if it failed, keeping keys and values aligned.
        if key_flags.needs_flush() {
            key_decoder.flush(key_flags);
        }
        if value_flags.needs_flush() {
            value_decoder.flush(value_flags);
        }
        *row_entries += 1;
        entry.map(|()| total)
    }

    /// Ends a row for this field. [`FieldFlags::HAS_VALUE`]: the field
    /// occurred in the row; otherwise its value is null with
    /// [`FieldFlags::HAS_PRESENCE`], the default without.
    fn flush(&mut self, flags: FieldFlags) {
        let has_value = flags.contains(FieldFlags::HAS_VALUE);
        let has_presence = flags.contains(FieldFlags::HAS_PRESENCE);
        match self {
            // Appended in decode; flushed only when absent from the row.
            Self::Int32 { builder } | Self::Sint32 { builder } | Self::Sfixed32 { builder } => {
                append_absent(builder, has_presence)
            }
            Self::EnumInt32 { default, builder } => {
                if has_presence {
                    builder.append_null();
                } else {
                    builder.append_value(*default);
                }
            }
            Self::Int64 { builder } | Self::Sint64 { builder } | Self::Sfixed64 { builder } => {
                append_absent(builder, has_presence)
            }
            Self::UInt32 { builder } | Self::Fixed32 { builder } => {
                append_absent(builder, has_presence)
            }
            Self::UInt64 { builder } | Self::Fixed64 { builder } => {
                append_absent(builder, has_presence)
            }
            Self::Float { builder } => append_absent(builder, has_presence),
            Self::Double { builder } => append_absent(builder, has_presence),
            Self::EnumString {
                builder,
                enum_names,
            } => {
                if has_presence {
                    builder.push_null();
                } else {
                    builder.push(enum_names.default_name().as_ref());
                }
            }
            Self::EnumBinary {
                builder,
                enum_names,
            } => {
                if has_presence {
                    builder.push_null();
                } else {
                    builder.push(enum_names.default_name().as_bytes());
                }
            }
            Self::Bool { value, builder } => {
                if has_value {
                    builder.append_value(*value);
                } else if has_presence {
                    builder.append_null();
                } else {
                    builder.append_value(false);
                }
                *value = false;
            }
            // Appended in decode; flushed only when absent from the row.
            Self::String { builder } => {
                if has_presence {
                    builder.push_null();
                } else {
                    builder.push("");
                }
            }
            Self::Bytes { builder } => {
                if has_presence {
                    builder.push_null();
                } else {
                    builder.push(b"");
                }
            }
            // Well-known types
            Self::Timestamp {
                seconds,
                nanos,
                builder,
                unit,
                ..
            } => {
                if has_value {
                    builder.append_value(convert_seconds_nanos_to_unit(*seconds, *nanos, *unit));
                } else {
                    builder.append_null();
                }
                *seconds = 0;
                *nanos = 0;
            }
            Self::Duration {
                seconds,
                nanos,
                builder,
                unit,
                ..
            } => {
                if has_value {
                    builder.append_value(convert_seconds_nanos_to_unit(*seconds, *nanos, *unit));
                } else {
                    builder.append_null();
                }
                *seconds = 0;
                *nanos = 0;
            }
            Self::Date {
                year,
                month,
                day,
                builder,
            } => {
                if has_value {
                    if *year == 0 && *month == 0 && *day == 0 {
                        builder.append_value(0);
                    } else {
                        builder.append_value(date_to_days_from_epoch(*year, *month, *day));
                    }
                } else {
                    builder.append_null();
                }
                *year = 0;
                *month = 0;
                *day = 0;
            }
            Self::TimeOfDay {
                hours,
                minutes,
                seconds_val,
                nanos,
                builder,
                unit,
            } => {
                if has_value {
                    let total_seconds = i64::from(*hours) * 3600
                        + i64::from(*minutes) * 60
                        + i64::from(*seconds_val);
                    builder.append_value(convert_seconds_nanos_to_unit(
                        total_seconds,
                        *nanos,
                        *unit,
                    ));
                } else {
                    builder.append_null();
                }
                *hours = 0;
                *minutes = 0;
                *seconds_val = 0;
                *nanos = 0;
            }
            // Wrapper types: appended in decode, flushed only when absent,
            // except `WrapperBool`.
            Self::WrapperDouble { builder } => builder.append_null(),
            Self::WrapperFloat { builder } => builder.append_null(),
            Self::WrapperInt64 { builder } => builder.append_null(),
            Self::WrapperUInt64 { builder } => builder.append_null(),
            Self::WrapperInt32 { builder } => builder.append_null(),
            Self::WrapperUInt32 { builder } => builder.append_null(),
            Self::WrapperBool { value, builder } => {
                if has_value {
                    builder.append_value(*value);
                } else {
                    builder.append_null();
                }
                *value = false;
            }
            Self::WrapperString { builder } => builder.push_null(),
            Self::WrapperBytes { builder } => builder.push_null(),
            // Nested message
            Self::Message {
                sub_decoder,
                is_valid,
            } => {
                // Absent: every field of the struct gets its default.
                is_valid.append_value(has_value);
                sub_decoder.flush_row();
            }
            // Repeated fields: push offset
            Self::Repeated { inner, offsets, .. } => {
                offsets.push(inner.len());
            }
            Self::Map {
                offsets,
                row_entries,
                ..
            } => {
                let end = offsets.last().unwrap() + std::mem::take(row_entries);
                offsets.push(end);
            }
        }
    }

    /// The column decoded so far. Builders that [`MessageDecoder::with_capacity`]
    /// sizes are left sized for `capacity` rows, the rest empty. The caller
    /// names the field.
    fn finish(&mut self, capacity: usize) -> Arc<dyn Array> {
        match self {
            Self::Int32 { builder, .. } | Self::EnumInt32 { builder, .. } => {
                finish_primitive(builder, capacity)
            }
            Self::Int64 { builder, .. } => finish_primitive(builder, capacity),
            Self::UInt32 { builder, .. } => finish_primitive(builder, capacity),
            Self::UInt64 { builder, .. } => finish_primitive(builder, capacity),
            Self::Sint32 { builder, .. } | Self::Sfixed32 { builder, .. } => {
                finish_primitive(builder, capacity)
            }
            Self::Sint64 { builder, .. } | Self::Sfixed64 { builder, .. } => {
                finish_primitive(builder, capacity)
            }
            Self::Fixed32 { builder, .. } => finish_primitive(builder, capacity),
            Self::Fixed64 { builder, .. } => finish_primitive(builder, capacity),
            Self::Float { builder, .. } => finish_primitive(builder, capacity),
            Self::Double { builder, .. } => finish_primitive(builder, capacity),
            Self::Bool { builder, .. } => {
                let array = builder.finish();
                if capacity > 0 {
                    *builder = BooleanBuilder::with_capacity(capacity);
                }
                Arc::new(array)
            }
            Self::String { builder, .. } => builder.finish(capacity),
            Self::EnumString { builder, .. } => builder.finish(capacity),
            Self::Bytes { builder, .. } => builder.finish(capacity),
            Self::EnumBinary { builder, .. } => builder.finish(capacity),
            Self::Timestamp {
                builder, unit, tz, ..
            } => finish_timestamp(builder, *unit, tz),
            Self::Duration { builder, unit, .. } => finish_duration(builder, *unit),
            Self::Date { builder, .. } => finish_primitive(builder, 0),
            Self::TimeOfDay { builder, unit, .. } => finish_time_of_day(builder, *unit),
            Self::WrapperDouble { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperFloat { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperInt64 { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperUInt64 { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperInt32 { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperUInt32 { builder, .. } => finish_primitive(builder, 0),
            Self::WrapperBool { builder, .. } => Arc::new(builder.finish()),
            Self::WrapperString { builder, .. } => builder.finish(0),
            Self::WrapperBytes { builder, .. } => builder.finish(0),
            Self::Message {
                sub_decoder,
                is_valid,
                ..
            } => Arc::new(sub_decoder.build_struct_array(Some(is_valid.finish()), capacity > 0)),
            Self::Repeated {
                inner,
                offsets,
                list_name,
                list_nullable,
            } => {
                let vals = inner.finish();
                let large = matches!(offsets, ListOffsets::Large(_));
                std::mem::replace(offsets, ListOffsets::new(large)).finish(
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
                ..
            } => Self::finish_map(
                key_decoder,
                value_decoder,
                offsets,
                map_value_name,
                *map_value_nullable,
            ),
        }
    }

    /// The map column for [`Self::finish`]. Its own function, like
    /// [`Self::decode_map_entry`], since it finishes the key and the value.
    fn finish_map(
        key_decoder: &mut FieldDecoder,
        value_decoder: &mut FieldDecoder,
        offsets: &mut Vec<i32>,
        map_value_name: &str,
        map_value_nullable: bool,
    ) -> Arc<dyn Array> {
        let key_array = key_decoder.finish(0);
        let value_array = value_decoder.finish(0);
        let key_field = Arc::new(Field::new("key", key_array.data_type().clone(), false));
        let value_field = Arc::new(Field::new(
            map_value_name,
            value_array.data_type().clone(),
            map_value_nullable,
        ));
        let entries_struct_type =
            DataType::Struct(vec![key_field.as_ref().clone(), value_field.as_ref().clone()].into());
        let entry_struct =
            StructArray::from(vec![(key_field, key_array), (value_field, value_array)]);
        let map_dt = DataType::Map(
            Arc::new(Field::new("entries", entries_struct_type, false)),
            false,
        );
        let len = offsets.len() - 1;
        let offsets_buf = Buffer::from_vec(std::mem::replace(offsets, vec![0]));
        let map_data = ArrayData::builder(map_dt)
            .len(len)
            .add_buffer(offsets_buf)
            .add_child_data(entry_struct.into_data())
            .build()
            .unwrap();
        Arc::new(MapArray::from(map_data))
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

bitflags::bitflags! {
    /// Per-field flags of a [`MessageDecoder`].
    #[derive(Clone, Copy)]
    struct FieldFlags: u8 {
        /// The field occurred in the current row (see [`FieldDecoder::decode`]).
        const HAS_VALUE = 1;
        /// [`FieldDescriptor::supports_presence`]: an absent value is null.
        const HAS_PRESENCE = 1 << 1;
        /// [`FieldDecoder::flush`] runs at the end of every row, not only
        /// when the field is absent. Set by [`build_field_decoder`].
        const BUFFERED = 1 << 2;
    }
}

impl FieldFlags {
    /// The end of a row must flush the field: it is absent, or buffered.
    fn needs_flush(self) -> bool {
        !self.contains(Self::HAS_VALUE) || self.contains(Self::BUFFERED)
    }
}

/// One field of a [`MessageDecoder`].
struct DecoderState {
    decoder: FieldDecoder,
    field: FieldDescriptor,
    flags: FieldFlags,
}

/// Decodes serialized messages of one type into a record batch, one row per
/// [`MessageDecoder::decode_row`] call; [`MessageDecoder::finish`] returns
/// the batch.
pub struct MessageDecoder {
    /// One per field, in field order.
    decoders: Vec<DecoderState>,
    tag_map: Vec<Option<usize>>,
    list_nullable: bool,
    map_nullable: bool,
    num_rows: usize,
    capacity: usize,
    fields: Option<Fields>,
}

impl MessageDecoder {
    /// A decoder for messages of type `descriptor`, with every field and
    /// builders that grow as rows are added.
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
        Self::new_inner(descriptor, config, capacity, &mut Vec::new())
    }

    /// `ancestors` is the chain of message types currently being built, root first. A field
    /// whose message type is already in the chain is a back-edge of a recursive proto (a
    /// message containing itself, directly or through a cycle — `Tree.children`,
    /// `google.protobuf.Struct` → `Value` → `Struct`). Recursing into it would never terminate
    /// and overflowed the stack; such a field decodes as the raw protobuf bytes of the nested
    /// message instead (`Binary`, or `List<Binary>` when repeated).
    fn new_inner(
        descriptor: &MessageDescriptor,
        config: &PtarsConfig,
        capacity: usize,
        ancestors: &mut Vec<String>,
    ) -> Self {
        ancestors.push(descriptor.full_name().to_string());
        let mut decoders = Vec::new();
        let mut max_field_number: u32 = 0;

        for field in descriptor.fields() {
            if let Some((decoder, mut flags)) =
                build_field_decoder(&field, config, capacity, ancestors)
            {
                if field.number() > max_field_number {
                    max_field_number = field.number();
                }
                flags.set(FieldFlags::HAS_PRESENCE, field.supports_presence());
                decoders.push(DecoderState {
                    decoder,
                    field,
                    flags,
                });
            }
        }
        ancestors.pop();

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
            capacity,
            fields: None,
        }
    }

    /// Decodes one serialized message as the next row.
    ///
    /// An error still ends the row (`flush_row`), so the decoder stays
    /// usable; drop the failed rows to keep only valid messages.
    ///
    /// Panics on a length prefix that overflows `usize`, a `google.type.Date`
    /// that `NaiveDate::from_ymd_opt` rejects, or an overflow in
    /// `convert_seconds_nanos_to_unit`.
    pub fn decode_row(&mut self, buf: &[u8]) -> Result<(), prost::DecodeError> {
        let result = self.decode_fields(buf);
        self.flush_row();
        self.num_rows += 1;
        result
    }

    /// Decodes a message's fields without ending the row, which
    /// [`MessageDecoder::decode_row`] or, for a nested message, the parent's
    /// [`FieldDecoder::flush`] does.
    #[inline(always)]
    fn decode_fields(&mut self, buf: &[u8]) -> Result<(), prost::DecodeError> {
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
                let DecoderState { decoder, flags, .. } = &mut self.decoders[idx];
                pos += decoder.decode(wire_type, &buf[pos..], flags)?;
            } else {
                pos += skip_field(wire_type, &buf[pos..])?;
            }
        }
        Ok(())
    }

    /// [`MessageDecoder::decode_fields`] for a nested message, out of line
    /// so that only `decode_row` gets the inlined copy.
    #[inline(never)]
    fn decode_nested(&mut self, buf: &[u8]) -> Result<(), prost::DecodeError> {
        self.decode_fields(buf)
    }

    /// Ends a row: flushes every decoder that did not append its value
    /// directly (see [`FieldDecoder::decode`]) and clears
    /// [`FieldFlags::HAS_VALUE`].
    fn flush_row(&mut self) {
        for DecoderState { decoder, flags, .. } in &mut self.decoders {
            if flags.needs_flush() {
                decoder.flush(*flags);
            }
            flags.remove(FieldFlags::HAS_VALUE);
        }
    }

    /// Appends a null row: null in fields with presence, defaults elsewhere.
    pub fn decode_null_row(&mut self) {
        self.flush_row();
        self.num_rows += 1;
    }

    fn row_count(&self) -> usize {
        self.num_rows
    }

    /// `presize`: leave the builders sized for this decoder's capacity.
    fn build_struct_array(
        &mut self,
        validity: Option<arrow_array::BooleanArray>,
        presize: bool,
    ) -> StructArray {
        let capacity = if presize { self.capacity } else { 0 };
        // Rows of a repeated message field are counted here; the next batch
        // starts again from zero.
        let num_rows = std::mem::take(&mut self.num_rows);
        if self.decoders.is_empty() {
            let len = validity.as_ref().map_or(num_rows, |v| v.len());
            return StructArray::new_empty_fields(
                len,
                validity.map(|v| arrow::buffer::NullBuffer::new(v.values().clone())),
            );
        }

        let columns: Vec<Arc<dyn Array>> = self
            .decoders
            .iter_mut()
            .map(|DecoderState { decoder, .. }| decoder.finish(capacity))
            .collect();

        let fields = self.fields.get_or_insert_with(|| {
            self.decoders
                .iter()
                .zip(&columns)
                .map(|(DecoderState { field, .. }, column)| {
                    let nullable = if field.is_list() {
                        self.list_nullable
                    } else if field.is_map() {
                        self.map_nullable
                    } else {
                        field.supports_presence()
                    };
                    Field::new(field.name(), column.data_type().clone(), nullable)
                })
                .collect()
        });

        StructArray::new(
            fields.clone(),
            columns,
            validity.map(|v| arrow::buffer::NullBuffer::new(v.values().clone())),
        )
    }

    /// The decoded rows as a batch, which takes over the builders' buffers.
    pub fn finish(mut self) -> RecordBatch {
        self.finish_rows(false)
    }

    /// Returns the rows decoded since the last call as a batch and leaves
    /// the decoder ready for the next one. Builders that
    /// [`MessageDecoder::with_capacity`] sizes are sized for the same number
    /// of rows again; the rest start empty.
    pub fn finish_batch(&mut self) -> RecordBatch {
        self.finish_rows(true)
    }

    /// `presize`: see [`MessageDecoder::build_struct_array`].
    fn finish_rows(&mut self, presize: bool) -> RecordBatch {
        let num_rows = std::mem::take(&mut self.num_rows);
        if self.decoders.is_empty() {
            let schema = Arc::new(arrow_schema::Schema::empty());
            return RecordBatch::try_new_with_options(
                schema,
                vec![],
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(num_rows)),
            )
            .unwrap();
        }
        let struct_array = self.build_struct_array(None, presize);
        RecordBatch::from(struct_array)
    }
}

// ---------------------------------------------------------------------------
// build_field_decoder
// ---------------------------------------------------------------------------

/// The decoder for `field`, with [`FieldFlags::BUFFERED`] set where it
/// needs it. `capacity`: rows to size scalar builders for (0 when unknown).
fn build_field_decoder(
    field: &FieldDescriptor,
    config: &PtarsConfig,
    capacity: usize,
    ancestors: &mut Vec<String>,
) -> Option<(FieldDecoder, FieldFlags)> {
    if field.is_map() {
        return Some((
            build_map_decoder(field, config, ancestors)?,
            FieldFlags::BUFFERED,
        ));
    }
    if field.is_list() {
        return Some((
            build_repeated_decoder(field, config, ancestors)?,
            FieldFlags::BUFFERED,
        ));
    }

    let direct = |decoder| Some((decoder, FieldFlags::empty()));
    let buffered = |decoder| Some((decoder, FieldFlags::BUFFERED));
    match field.kind() {
        Kind::Int32 => direct(FieldDecoder::Int32 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Int64 => direct(FieldDecoder::Int64 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Uint32 => direct(FieldDecoder::UInt32 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Uint64 => direct(FieldDecoder::UInt64 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sint32 => direct(FieldDecoder::Sint32 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sint64 => direct(FieldDecoder::Sint64 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sfixed32 => direct(FieldDecoder::Sfixed32 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Sfixed64 => direct(FieldDecoder::Sfixed64 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Fixed32 => direct(FieldDecoder::Fixed32 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Fixed64 => direct(FieldDecoder::Fixed64 {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Float => direct(FieldDecoder::Float {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Double => direct(FieldDecoder::Double {
            builder: PrimitiveBuilder::with_capacity(capacity),
        }),
        Kind::Bool => buffered(FieldDecoder::Bool {
            value: false,
            builder: BooleanBuilder::with_capacity(capacity),
        }),
        Kind::String => direct(FieldDecoder::String {
            builder: StringColumn::with_capacity(config.use_large_string, capacity),
        }),
        Kind::Bytes => direct(FieldDecoder::Bytes {
            builder: ByteColumn::with_capacity(config.use_large_binary, capacity),
        }),
        Kind::Enum(enum_desc) => match config.enum_repr {
            EnumRepr::Int32 => direct(FieldDecoder::EnumInt32 {
                default: enum_desc.default_value().number(),
                builder: PrimitiveBuilder::with_capacity(capacity),
            }),
            EnumRepr::String => direct(FieldDecoder::EnumString {
                builder: StringColumn::with_capacity(config.use_large_string, capacity),
                enum_names: EnumNames::new(&enum_desc),
            }),
            EnumRepr::Binary => direct(FieldDecoder::EnumBinary {
                builder: ByteColumn::with_capacity(config.use_large_binary, capacity),
                enum_names: EnumNames::new(&enum_desc),
            }),
        },
        Kind::Message(msg_desc) => build_message_field_decoder(msg_desc, config, ancestors),
    }
}

fn build_message_field_decoder(
    msg_desc: MessageDescriptor,
    config: &PtarsConfig,
    ancestors: &mut Vec<String>,
) -> Option<(FieldDecoder, FieldFlags)> {
    let direct = |decoder| Some((decoder, FieldFlags::empty()));
    let buffered = |decoder| Some((decoder, FieldFlags::BUFFERED));
    match msg_desc.full_name() {
        "google.protobuf.Timestamp" => buffered(FieldDecoder::Timestamp {
            seconds: 0,
            nanos: 0,
            builder: PrimitiveBuilder::new(),
            unit: config.timestamp_unit.into(),
            tz: config.timestamp_tz.clone(),
        }),
        "google.protobuf.Duration" => buffered(FieldDecoder::Duration {
            seconds: 0,
            nanos: 0,
            builder: PrimitiveBuilder::new(),
            unit: config.duration_unit.into(),
        }),
        "google.type.Date" => buffered(FieldDecoder::Date {
            year: 0,
            month: 0,
            day: 0,
            builder: PrimitiveBuilder::new(),
        }),
        "google.type.TimeOfDay" => buffered(FieldDecoder::TimeOfDay {
            hours: 0,
            minutes: 0,
            seconds_val: 0,
            nanos: 0,
            builder: PrimitiveBuilder::new(),
            unit: config.time_unit.into(),
        }),
        "google.protobuf.DoubleValue" => direct(FieldDecoder::WrapperDouble {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.FloatValue" => direct(FieldDecoder::WrapperFloat {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.Int64Value" => direct(FieldDecoder::WrapperInt64 {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.UInt64Value" => direct(FieldDecoder::WrapperUInt64 {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.Int32Value" => direct(FieldDecoder::WrapperInt32 {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.UInt32Value" => direct(FieldDecoder::WrapperUInt32 {
            builder: PrimitiveBuilder::new(),
        }),
        "google.protobuf.BoolValue" => buffered(FieldDecoder::WrapperBool {
            value: false,
            builder: BooleanBuilder::new(),
        }),
        "google.protobuf.StringValue" => direct(FieldDecoder::WrapperString {
            builder: StringColumn::with_capacity(config.use_large_string, 0),
        }),
        "google.protobuf.BytesValue" => direct(FieldDecoder::WrapperBytes {
            builder: ByteColumn::with_capacity(config.use_large_binary, 0),
        }),
        _ => {
            // Recursion guard: a message type already being built higher in the chain
            // decodes as its raw bytes (see MessageDecoder::new_inner).
            if ancestors.iter().any(|a| a == msg_desc.full_name()) {
                return direct(FieldDecoder::Bytes {
                    builder: ByteColumn::with_capacity(config.use_large_binary, 0),
                });
            }
            let sub_decoder = MessageDecoder::new_inner(&msg_desc, config, 0, ancestors);
            buffered(FieldDecoder::Message {
                sub_decoder: Box::new(sub_decoder),
                is_valid: BooleanBuilder::new(),
            })
        }
    }
}

fn build_repeated_decoder(
    field: &FieldDescriptor,
    config: &PtarsConfig,
    ancestors: &mut Vec<String>,
) -> Option<FieldDecoder> {
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
                enum_names: EnumNames::new(&enum_desc),
            },
            EnumRepr::Binary => RepeatedInner::EnumBinary {
                values_builder: BinaryBuilderInner::new(config.use_large_binary),
                enum_names: EnumNames::new(&enum_desc),
            },
        },
        Kind::Message(msg_desc) => {
            return build_repeated_message_decoder(
                &msg_desc,
                config,
                offsets(),
                ln,
                lnb,
                ancestors,
            );
        }
    };

    Some(FieldDecoder::Repeated {
        inner: Box::new(inner),
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
    ancestors: &mut Vec<String>,
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
            // Recursion guard (see MessageDecoder::new_inner): a recursive `repeated Foo`
            // becomes a list of the elements' raw bytes.
            if ancestors.iter().any(|a| a == msg_desc.full_name()) {
                RepeatedInner::Bytes {
                    values_builder: BinaryBuilderInner::new(config.use_large_binary),
                }
            } else {
                let sub_decoder = MessageDecoder::new_inner(msg_desc, config, 0, ancestors);
                RepeatedInner::Message { sub_decoder }
            }
        }
    };

    Some(FieldDecoder::Repeated {
        inner: Box::new(inner),
        offsets,
        list_name: ln,
        list_nullable: lnb,
    })
}

fn build_map_decoder(
    field: &FieldDescriptor,
    config: &PtarsConfig,
    ancestors: &mut Vec<String>,
) -> Option<FieldDecoder> {
    let map_entry = match field.kind() {
        Kind::Message(desc) => desc,
        _ => return None,
    };
    let key_field = map_entry.get_field_by_name("key")?;
    let value_field = map_entry.get_field_by_name("value")?;

    // One row per entry; the number of entries is unknown.
    let (key_decoder, key_flags) = build_field_decoder(&key_field, config, 0, ancestors)?;
    let (value_decoder, value_flags) = build_field_decoder(&value_field, config, 0, ancestors)?;

    Some(FieldDecoder::Map {
        key_flags,
        value_flags,
        key_decoder: Box::new(key_decoder),
        value_decoder: Box::new(value_decoder),
        offsets: vec![0],
        row_entries: 0,
        map_value_name: config.map_value_name.clone(),
        map_value_nullable: config.map_value_nullable,
    })
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

    /// Seconds for 9999-12-31T23:59:59Z — the "never" sentinel inventory feeds use for
    /// open-ended lifecycle dates; it used to panic the decoder mid-batch.
    const YEAR_9999: i64 = 253_402_300_799;
    /// Seconds for 0001-01-01T00:00:00Z — the far-past null-date sentinel.
    const YEAR_0001: i64 = -62_135_596_800;

    #[test]
    fn test_nanosecond_overflow_saturates_instead_of_panicking() {
        assert_eq!(
            convert_seconds_nanos_to_unit(YEAR_9999, 0, TimeUnit::Nanosecond),
            i64::MAX
        );
        assert_eq!(
            convert_seconds_nanos_to_unit(YEAR_0001, 0, TimeUnit::Nanosecond),
            i64::MIN
        );
    }

    #[test]
    fn test_coarser_units_represent_the_sentinels_exactly() {
        assert_eq!(
            convert_seconds_nanos_to_unit(YEAR_9999, 0, TimeUnit::Microsecond),
            YEAR_9999 * 1_000_000
        );
        assert_eq!(
            convert_seconds_nanos_to_unit(YEAR_0001, 0, TimeUnit::Millisecond),
            YEAR_0001 * 1_000
        );
        assert_eq!(
            convert_seconds_nanos_to_unit(YEAR_9999, 0, TimeUnit::Second),
            YEAR_9999
        );
    }

    #[test]
    fn test_ordinary_values_unchanged_by_the_saturating_path() {
        let secs = 1_786_608_000_i64; // 2026-08-13T00:00:00Z
        assert_eq!(
            convert_seconds_nanos_to_unit(secs, 123_456_789, TimeUnit::Nanosecond),
            secs * 1_000_000_000 + 123_456_789
        );
        assert_eq!(
            convert_seconds_nanos_to_unit(secs, 123_456_789, TimeUnit::Microsecond),
            secs * 1_000_000 + 123_456
        );
        // Negative durations carry negative nanos; truncation stays toward zero.
        assert_eq!(
            convert_seconds_nanos_to_unit(-5, -500_000_000, TimeUnit::Millisecond),
            -5_500
        );
    }

    #[test]
    fn test_partial_dates_normalise_instead_of_panicking() {
        let days = |y: i32, m: u32, d: u32| {
            chrono::NaiveDate::from_ymd_opt(y, m, d)
                .unwrap()
                .num_days_from_ce()
                - CE_OFFSET
        };
        assert_eq!(date_to_days_from_epoch(1970, 1, 1), 0);
        assert_eq!(date_to_days_from_epoch(2026, 3, 15), days(2026, 3, 15));
        // month/day "not significant" → first of the period
        assert_eq!(date_to_days_from_epoch(2026, 0, 0), days(2026, 1, 1));
        assert_eq!(date_to_days_from_epoch(2026, 3, 0), days(2026, 3, 1));
        // no year, or not a calendar date → epoch (never a panic)
        assert_eq!(date_to_days_from_epoch(0, 3, 15), 0);
        assert_eq!(date_to_days_from_epoch(0, 0, 0), 0);
        assert_eq!(date_to_days_from_epoch(2026, 2, 30), 0);
        assert_eq!(date_to_days_from_epoch(2026, 13, 1), 0);
        assert_eq!(date_to_days_from_epoch(2026, -1, 1), 0);
        assert_eq!(date_to_days_from_epoch(9999, 12, 31), days(9999, 12, 31));
        assert_eq!(date_to_days_from_epoch(1, 1, 1), days(1, 1, 1));
    }

    #[test]
    fn test_enum_names_borrow_declared_and_format_unknown() {
        use prost_reflect::prost_types::{
            EnumDescriptorProto, EnumValueDescriptorProto, FileDescriptorProto,
        };
        let file = FileDescriptorProto {
            name: Some("e.proto".to_string()),
            package: Some("t".to_string()),
            enum_type: vec![EnumDescriptorProto {
                name: Some("Color".to_string()),
                value: vec![
                    EnumValueDescriptorProto {
                        name: Some("UNSPECIFIED".to_string()),
                        number: Some(0),
                        ..Default::default()
                    },
                    EnumValueDescriptorProto {
                        name: Some("RED".to_string()),
                        number: Some(1),
                        ..Default::default()
                    },
                    EnumValueDescriptorProto {
                        name: Some("NEGATIVE".to_string()),
                        number: Some(-1),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut pool = prost_reflect::DescriptorPool::new();
        pool.add_file_descriptor_proto(file).unwrap();
        let names = EnumNames::new(&pool.get_enum_by_name("t.Color").unwrap());
        assert!(matches!(names.name(1), Cow::Borrowed("RED")));
        assert_eq!(names.name(-1), "NEGATIVE");
        assert_eq!(names.name(0), "UNSPECIFIED");
        assert!(matches!(names.name(999), Cow::Owned(ref s) if s == "999"));
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
