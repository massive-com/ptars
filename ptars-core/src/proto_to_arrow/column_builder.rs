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
