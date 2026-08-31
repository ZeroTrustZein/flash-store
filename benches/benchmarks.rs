use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flash_store::prelude::*;
use tempfile::tempdir;

fn bench_puts(c: &mut Criterion) {
    c.bench_function("flashstore_put_seq", |b| {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let db = FlashStore::open(options).unwrap();
        let mut i = 0u64;

        b.iter(|| {
            i += 1;
            let key = format!("bench_key_{:08}", i);
            let val = format!("bench_val_{:08}", i);
            db.put(black_box(key.into_bytes()), black_box(val.into_bytes()))
                .unwrap();
        });
    });
}

fn bench_gets(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options).unwrap();

    for i in 0..10_000 {
        let key = format!("bench_key_{:08}", i);
        let val = format!("bench_val_{:08}", i);
        db.put(key.into_bytes(), val.into_bytes()).unwrap();
    }

    c.bench_function("flashstore_get_hit", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 1) % 10_000;
            let key = format!("bench_key_{:08}", i);
            let _ = db.get(black_box(key.as_bytes())).unwrap();
        });
    });
}

criterion_group!(benches, bench_puts, bench_gets);
criterion_main!(benches);
