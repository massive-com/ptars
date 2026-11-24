use std::sync::{Arc, OnceLock};

use arrow_schema::Fields;
use prost_reflect::{FieldDescriptor, Kind, MessageDescriptor};

use super::{build_field_decoder, DecoderState, EnumNames, FieldFlags, MessageDecoder, WireTypes};
use crate::config::PtarsConfig;

/// Reads a [`MessageDescriptor`] and config once to allow building
/// [`MessageDecoder`] and [`Fields`] (schema) with cheap clones.
///
/// A plan holds no builders between batches,
/// unlike reusing a decoder with [`MessageDecoder::finish_batch`].
#[derive(Debug, Clone)]
pub struct DecoderPlan(pub(super) Arc<PlanParts>);

#[derive(Debug)]
pub(super) struct PlanParts {
    /// One per field that gets a column, in field order.
    pub(super) fields: Vec<FieldPlan>,
    pub(super) tag_map: Vec<Option<usize>>,
    pub(super) config: PtarsConfig,
    /// The fields of the decoded batches, built by the first of the plan's
    /// decoders to finish.
    pub(super) schema: OnceLock<Fields>,
}

impl DecoderPlan {
    /// The plan for messages of type `descriptor`.
    ///
    /// A field whose message type contains it, directly or through a cycle
    /// (`Tree.children`, `google.protobuf.Struct` → `Value` → `Struct`), is a
    /// back-edge of a recursive proto. Planning it would never terminate; it
    /// decodes as the raw protobuf bytes of the nested message instead
    /// (`Binary`, or `List<Binary>` when repeated).
    pub fn new(descriptor: &MessageDescriptor, config: &PtarsConfig) -> Self {
        Self::new_inner(descriptor, config, &mut Vec::new())
    }

    /// `ancestors`: the message types being planned, root first.
    fn new_inner(
        descriptor: &MessageDescriptor,
        config: &PtarsConfig,
        ancestors: &mut Vec<String>,
    ) -> Self {
        ancestors.push(descriptor.full_name().to_string());
        // Sized up front: `filter_map` would make `collect` grow it.
        let mut fields = Vec::with_capacity(descriptor.fields().len());
        fields.extend(
            descriptor
                .fields()
                .filter_map(|field| FieldPlan::resolve(field, config, ancestors)),
        );
        ancestors.pop();
        let max_field_number = fields.iter().map(|plan| plan.field.number()).max();
        let mut tag_map = vec![None; max_field_number.map_or(0, |n| n as usize + 1)];
        for (idx, plan) in fields.iter().enumerate() {
            tag_map[plan.field.number() as usize] = Some(idx);
        }
        Self(Arc::new(PlanParts {
            fields,
            tag_map,
            config: config.clone(),
            schema: OnceLock::new(),
        }))
    }

    /// A decoder of the plan's messages, as [`MessageDecoder::with_capacity`]
    /// builds one.
    pub fn decoder(&self, capacity: usize) -> MessageDecoder {
        let parts = &*self.0;
        let decoders = parts
            .fields
            .iter()
            .map(|plan| {
                let (decoder, mut flags) = build_field_decoder(plan, &parts.config, capacity);
                flags.set(FieldFlags::HAS_PRESENCE, plan.has_presence);
                DecoderState {
                    decoder,
                    flags,
                    wire_types: plan.wire_types,
                }
            })
            .collect();
        MessageDecoder {
            decoders,
            plan: self.clone(),
            num_rows: 0,
            capacity,
        }
    }
}

/// What [`build_field_decoder`] needs from one field's descriptor.
#[derive(Debug)]
pub(super) struct FieldPlan {
    pub(super) field: FieldDescriptor,
    pub(super) kind: Kind,
    pub(super) shape: FieldShape,
    /// The plan of a message field's message; `None` for a back-edge of a
    /// recursive message (see [`DecoderPlan::new`]).
    pub(super) message: Option<DecoderPlan>,
    /// The names of an enum field's values, shared by the plan's decoders.
    enum_names: Option<Arc<EnumNames>>,
    /// [`FieldDescriptor::supports_presence`].
    pub(super) has_presence: bool,
    /// See [`DecoderState::wire_types`].
    pub(super) wire_types: WireTypes,
}

#[derive(Debug)]
pub(super) enum FieldShape {
    Singular,
    List,
    /// A map field, with its entries' key and value fields.
    Map {
        key: Box<FieldPlan>,
        value: Box<FieldPlan>,
    },
}

impl FieldPlan {
    /// `None` for a map field whose entry lacks a key or value: it gets no
    /// column.
    fn resolve(
        field: FieldDescriptor,
        config: &PtarsConfig,
        ancestors: &mut Vec<String>,
    ) -> Option<Self> {
        let kind = field.kind();
        let shape = if field.is_map() {
            let Kind::Message(entry) = &kind else {
                return None;
            };
            let key = Self::resolve(entry.get_field_by_name("key")?, config, ancestors)?;
            let value = Self::resolve(entry.get_field_by_name("value")?, config, ancestors)?;
            FieldShape::Map {
                key: Box::new(key),
                value: Box::new(value),
            }
        } else if field.is_list() {
            FieldShape::List
        } else {
            FieldShape::Singular
        };
        let message = match &kind {
            Kind::Message(message) if !matches!(shape, FieldShape::Map { .. }) => {
                let back_edge = ancestors.iter().any(|a| a == message.full_name());
                (!back_edge).then(|| DecoderPlan::new_inner(message, config, ancestors))
            }
            _ => None,
        };
        let enum_names = match &kind {
            Kind::Enum(descriptor) => Some(Arc::new(EnumNames::new(descriptor))),
            _ => None,
        };
        Some(Self {
            wire_types: if config.strict_wire_types {
                WireTypes::of(&field)
            } else {
                WireTypes::ALL
            },
            has_presence: field.supports_presence(),
            field,
            kind,
            shape,
            message,
            enum_names,
        })
    }

    /// The names of this enum field's values.
    pub(super) fn enum_names(&self) -> Arc<EnumNames> {
        let names = self.enum_names.as_ref();
        Arc::clone(names.expect("resolve sets enum_names for every enum field"))
    }
}
