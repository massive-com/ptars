//! Decoding protobuf messages to an Arrow `RecordBatch`.
//!
//! `build` and `build_finish` measure the fixed cost of a batch.

use arrow_array::BinaryArray;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prost::Message;
use prost_reflect::prost_types::field_descriptor_proto::{Label, Type};
use prost_reflect::prost_types::{DescriptorProto, FieldDescriptorProto, FileDescriptorProto};
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, Value};
use ptars_core::{binary_array_to_record_batch_direct, MessageDecoder, PtarsConfig};

const BATCH_SIZES: [usize; 4] = [1, 64, 1024, 8192];

fn field(name: &str, number: i32, ty: Type) -> FieldDescriptorProto {
    FieldDescriptorProto {
        name: Some(name.to_string()),
        number: Some(number),
        label: Some(Label::Optional.into()),
        r#type: Some(ty.into()),
        ..Default::default()
    }
}

fn repeated(name: &str, number: i32, ty: Type) -> FieldDescriptorProto {
    FieldDescriptorProto {
        label: Some(Label::Repeated.into()),
        ..field(name, number, ty)
    }
}

fn message(name: &str, fields: Vec<FieldDescriptorProto>) -> DescriptorProto {
    DescriptorProto {
        name: Some(name.to_string()),
        field: fields,
        ..Default::default()
    }
}

fn pool() -> DescriptorPool {
    let mut base = field("base", 1, Type::Message);
    base.type_name = Some(".bench.Base".to_string());
    let file = FileDescriptorProto {
        name: Some("bench.proto".to_string()),
        package: Some("bench".to_string()),
        syntax: Some("proto3".to_string()),
        message_type: vec![
            message(
                "Flat",
                vec![
                    field("name", 1, Type::String),
                    field("venue", 2, Type::Int32),
                    field("ask_price", 3, Type::Double),
                    field("ask_size", 4, Type::Double),
                    field("bid_price", 5, Type::Double),
                    field("bid_size", 6, Type::Double),
                    field("event_time", 7, Type::Int64),
                    field("receive_time", 8, Type::Int64),
                ],
            ),
            // Reads `Flat` messages knowing 3 of their 8 fields: the rest are unknown fields.
            message(
                "FlatSubset",
                vec![
                    field("ask_price", 3, Type::Double),
                    field("bid_price", 5, Type::Double),
                    field("event_time", 7, Type::Int64),
                ],
            ),
            message(
                "Base",
                vec![
                    field("name", 1, Type::String),
                    field("event_time", 2, Type::Int64),
                    field("source_time", 3, Type::Int64),
                    field("other_time", 4, Type::Int64),
                    field("sequence", 5, Type::Int64),
                ],
            ),
            message(
                "Nested",
                vec![
                    base,
                    repeated("flags", 2, Type::Int32),
                    repeated("tags", 3, Type::Int32),
                    field("bid_price", 4, Type::Double),
                    field("bid_size", 5, Type::Uint32),
                    field("bid_venue", 6, Type::Int32),
                    field("ask_price", 7, Type::Double),
                    field("ask_size", 8, Type::Uint32),
                    field("ask_venue", 9, Type::Int32),
                    field("kind", 10, Type::Int32),
                ],
            ),
        ],
        ..Default::default()
    };
    let mut pool = DescriptorPool::new();
    pool.add_file_descriptor_proto(file).unwrap();
    pool
}

const NAMES: [&str; 10] = [
    "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOX", "GOLF", "HOTEL", "INDIA", "JULIET",
];

/// Nanosecond timestamps, so 9-byte varints.
fn time(i: usize) -> i64 {
    1_731_700_000_000_000_000 + i as i64 * 137_000
}

fn flat(descriptor: &MessageDescriptor, i: usize) -> Vec<u8> {
    let mut m = DynamicMessage::new(descriptor.clone());
    let price = 1.0 + (i % 997) as f64 * 1e-5;
    m.set_field_by_name("name", Value::String(NAMES[i % NAMES.len()].to_string()));
    m.set_field_by_name("venue", Value::I32(48));
    m.set_field_by_name("ask_price", Value::F64(price + 2e-5));
    m.set_field_by_name("ask_size", Value::F64(1_000_000.0));
    m.set_field_by_name("bid_price", Value::F64(price - 2e-5));
    m.set_field_by_name("bid_size", Value::F64(500_000.0 + (i % 7) as f64 * 1e5));
    m.set_field_by_name("event_time", Value::I64(time(i)));
    m.set_field_by_name("receive_time", Value::I64(time(i) + 850_000));
    m.encode_to_vec()
}

fn nested(descriptor: &MessageDescriptor, i: usize) -> Vec<u8> {
    let base_descriptor = descriptor.parent_pool().get_message_by_name("bench.Base");
    let mut base = DynamicMessage::new(base_descriptor.unwrap());
    base.set_field_by_name("name", Value::String(NAMES[i % NAMES.len()].to_string()));
    base.set_field_by_name("event_time", Value::I64(time(i)));
    base.set_field_by_name("source_time", Value::I64(time(i) - 41_000));
    base.set_field_by_name("sequence", Value::I64(1_000_000 + i as i64));

    let mut m = DynamicMessage::new(descriptor.clone());
    let price = 100.0 + (i % 997) as f64 * 0.01;
    m.set_field_by_name("base", Value::Message(base));
    m.set_field_by_name("flags", Value::List(vec![Value::I32(1)]));
    if !i.is_multiple_of(11) {
        m.set_field_by_name("tags", Value::List(vec![Value::I32(1)]));
    }
    m.set_field_by_name("bid_price", Value::F64(price - 0.01));
    m.set_field_by_name("bid_size", Value::U32(100 * (1 + (i % 40) as u32)));
    m.set_field_by_name("bid_venue", Value::I32(1 + (i % 19) as i32));
    m.set_field_by_name("ask_price", Value::F64(price + 0.01));
    m.set_field_by_name("ask_size", Value::U32(100 * (1 + (i % 23) as u32)));
    m.set_field_by_name("ask_venue", Value::I32(1 + (i % 17) as i32));
    m.set_field_by_name("kind", Value::I32(1 + (i % 3) as i32));
    m.encode_to_vec()
}

fn decode_rows(decoder: &mut MessageDecoder, messages: &[&[u8]]) -> Result<(), prost::DecodeError> {
    for message in messages {
        decoder.decode_row(message)?;
    }
    Ok(())
}

fn bench_shape(
    c: &mut Criterion,
    name: &str,
    descriptor: &MessageDescriptor,
    writer: &MessageDescriptor,
    encode: fn(&MessageDescriptor, usize) -> Vec<u8>,
) {
    let config = PtarsConfig::default();
    let mut group = c.benchmark_group(name);
    for n in BATCH_SIZES {
        let owned: Vec<Vec<u8>> = (0..n).map(|i| encode(writer, i)).collect();
        let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let binary = BinaryArray::from_iter_values(&owned);
        let bytes: usize = owned.iter().map(Vec::len).sum();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("direct", n), &binary, |b, binary| {
            b.iter(|| binary_array_to_record_batch_direct(binary, descriptor, &config).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("decode_row", n), &messages, |b, m| {
            b.iter(|| {
                let mut decoder = MessageDecoder::with_capacity(descriptor, &config, n);
                decode_rows(&mut decoder, m).unwrap();
                decoder.finish()
            });
        });
        group.bench_with_input(BenchmarkId::new("reuse", n), &messages, |b, m| {
            let mut decoder = MessageDecoder::with_capacity(descriptor, &config, n);
            b.iter(|| {
                decode_rows(&mut decoder, m).unwrap();
                decoder.finish_batch()
            });
        });
        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, &n| {
            b.iter(|| MessageDecoder::with_capacity(descriptor, &config, n));
        });
        group.bench_with_input(BenchmarkId::new("build_finish", n), &n, |b, &n| {
            b.iter(|| MessageDecoder::with_capacity(descriptor, &config, n).finish());
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let pool = pool();
    for (name, message, writer, encode) in [
        (
            "flat",
            "bench.Flat",
            "bench.Flat",
            flat as fn(&MessageDescriptor, usize) -> Vec<u8>,
        ),
        ("nested", "bench.Nested", "bench.Nested", nested),
        ("unknown", "bench.FlatSubset", "bench.Flat", flat),
    ] {
        let descriptor = pool.get_message_by_name(message).unwrap();
        let writer = pool.get_message_by_name(writer).unwrap();
        bench_shape(c, name, &descriptor, &writer, encode);
    }
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
