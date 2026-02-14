use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use matching_engine::orderbook::OrderBook;
use matching_engine::types::*;
use rand::Rng;

fn bench_insert_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_limit");

    group.bench_function("single_insert", |b| {
        let mut book = OrderBook::new();
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let order = Order::new(id, Side::Buy, 100.0, 10, OrderType::Limit, id);
            black_box(book.submit(order, id));
        });
    });

    group.finish();
}

fn bench_match_market(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_market");

    for depth in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &depth,
            |b, &depth| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let mut book = OrderBook::new();
                        // Populate ask side
                        for i in 0..depth {
                            let order = Order::new(
                                i as u64,
                                Side::Sell,
                                100.0 + (i as f64) * 0.01,
                                100,
                                OrderType::Limit,
                                i as u64,
                            );
                            book.submit(order, i as u64);
                        }

                        let market = Order::new(
                            depth as u64 + 1,
                            Side::Buy,
                            0.0,
                            50,
                            OrderType::Market,
                            depth as u64 + 1,
                        );

                        let start = std::time::Instant::now();
                        black_box(book.submit(market, depth as u64 + 1));
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_cancel(c: &mut Criterion) {
    c.bench_function("cancel_order", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for i in 0..iters {
                let mut book = OrderBook::new();
                let order = Order::new(i, Side::Buy, 100.0, 10, OrderType::Limit, i);
                book.submit(order, i);

                let start = std::time::Instant::now();
                black_box(book.cancel(i));
                total += start.elapsed();
            }
            total
        });
    });
}

fn bench_throughput(c: &mut Criterion) {
    c.bench_function("mixed_workload_1000", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let mut book = OrderBook::new();
            for i in 0..1000u64 {
                let side = if rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
                let price = 100.0 + (rng.gen::<f64>() - 0.5) * 10.0;
                let order = Order::new(i, side, price, 100, OrderType::Limit, i);
                black_box(book.submit(order, i));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_insert_limit,
    bench_match_market,
    bench_cancel,
    bench_throughput,
);
criterion_main!(benches);
