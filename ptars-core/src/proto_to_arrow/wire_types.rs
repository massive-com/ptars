use prost_reflect::{FieldDescriptor, Kind};

use super::decode_error;

/// The wire types a field may arrive with, one bit per wire type.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(super) struct WireTypes(u8);

impl WireTypes {
    /// Every wire type, unless [`crate::PtarsConfig::strict_wire_types`].
    pub(super) const ALL: Self = Self(u8::MAX);
    const VARINT: Self = Self(1 << 0);
    const I64: Self = Self(1 << 1);
    const LEN: Self = Self(1 << 2);
    const I32: Self = Self(1 << 5);

    /// The wire types `field` can arrive with: its type's own, and
    /// length-delimited too for a repeated field (packed).
    pub(super) fn of(field: &FieldDescriptor) -> Self {
        if field.is_map() {
            return Self::LEN;
        }
        let own = match field.kind() {
            Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => Self::I64,
            Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => Self::I32,
            Kind::String | Kind::Bytes | Kind::Message(_) => Self::LEN,
            Kind::Int32
            | Kind::Int64
            | Kind::Uint32
            | Kind::Uint64
            | Kind::Sint32
            | Kind::Sint64
            | Kind::Bool
            | Kind::Enum(_) => Self::VARINT,
        };
        if field.is_list() {
            Self(own.0 | Self::LEN.0)
        } else {
            own
        }
    }

    /// Fails field `field_number` if it can't arrive with `wire_type`.
    #[inline(always)]
    pub(super) fn check(self, field_number: u32, wire_type: u8) -> Result<(), prost::DecodeError> {
        if self.0 & (1 << wire_type) == 0 {
            return Err(wire_type_error(field_number, wire_type));
        }
        Ok(())
    }
}

#[cold]
fn wire_type_error(field_number: u32, wire_type: u8) -> prost::DecodeError {
    decode_error(format!(
        "field {field_number} has wire type {wire_type}, which its type can't use"
    ))
}
