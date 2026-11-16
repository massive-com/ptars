use std::sync::Arc;

use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer};
use arrow_array::builder::NullBufferBuilder;
use arrow_array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};

use super::decode_error;

/// Validates `data` as UTF-8, checking for ASCII first: for short strings
/// that is much cheaper than the out-of-line call to `str::from_utf8`.
pub(super) fn validate_utf8(data: &[u8]) -> Result<&str, prost::DecodeError> {
    if data.is_ascii() {
        // SAFETY: ASCII is valid UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked(data) })
    } else {
        std::str::from_utf8(data).map_err(|_| decode_error("invalid UTF-8"))
    }
}

/// A binary column of a singular field, built one row at a time. Unlike
/// arrow's builders, the value appended last can be replaced, for a field
/// that occurs more than once in a message (last one wins). Strings use
/// [`StringColumn`].
pub(super) struct ByteColumn {
    /// Starts with 0; one end offset per row. `i64` until [`Self::finish`].
    offsets: Vec<i64>,
    values: Vec<u8>,
    nulls: NullBufferBuilder,
    large: bool,
}

impl ByteColumn {
    pub(super) fn new(large: bool) -> Self {
        Self::with_capacity(large, 0)
    }

    /// Room for `rows` rows; the value bytes still grow as needed.
    pub(super) fn with_capacity(large: bool, rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        Self {
            offsets,
            values: Vec::new(),
            nulls: NullBufferBuilder::new(rows),
            large,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline(always)]
    pub(super) fn push(&mut self, value: &[u8]) {
        self.values.extend_from_slice(value);
        self.offsets.push(self.values.len() as i64);
        self.nulls.append_non_null();
    }

    pub(super) fn push_null(&mut self) {
        self.offsets.push(self.values.len() as i64);
        self.nulls.append_null();
    }

    #[inline]
    pub(super) fn push_or_replace(&mut self, value: &[u8], has_value: &mut bool) {
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
    pub(super) fn finish(&mut self) -> Arc<dyn Array> {
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
pub(super) struct StringColumn(ByteColumn);

impl StringColumn {
    pub(super) fn new(large: bool) -> Self {
        Self(ByteColumn::new(large))
    }

    pub(super) fn with_capacity(large: bool, rows: usize) -> Self {
        Self(ByteColumn::with_capacity(large, rows))
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(super) fn push(&mut self, value: &str) {
        self.0.push(value.as_bytes());
    }

    pub(super) fn push_null(&mut self) {
        self.0.push_null();
    }

    #[inline]
    pub(super) fn push_or_replace(&mut self, value: &str, has_value: &mut bool) {
        self.0.push_or_replace(value.as_bytes(), has_value);
    }

    /// Builds a string array, large or not as configured.
    pub(super) fn finish(&mut self) -> Arc<dyn Array> {
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

#[cfg(test)]
mod tests {
    use super::validate_utf8;

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
}
