//! Side benchmark (not for commit): per-launch overhead of the normal launch
//! path vs graph replay on HIP, across kernel sizes and graph lengths.
//!
//! One "pass" is a chain of dependent kernels (ping-pong between two buffers),
//! the decode-loop shape. Small elements → CPU/launch-overhead bound; large
//! elements → GPU bound (a pass takes 10ms+), showing where the replay win
//! saturates. Reports issue time (CPU cost to enqueue, before any sync) and
//! end-to-end time (including GPU execution, which both paths pay identically).

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_hip::HipRuntime;
use std::time::{Duration, Instant};

#[cube(launch)]
fn add_one_tensor(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

struct Config {
    name: &'static str,
    elems: usize,
    kernels: usize,
    iters: usize,
}

const CONFIGS: &[Config] = &[
    Config {
        name: "tiny x150",
        elems: 64,
        kernels: 150,
        iters: 100,
    },
    Config {
        name: "tiny x1000",
        elems: 64,
        kernels: 1000,
        iters: 50,
    },
    Config {
        name: "tiny x5000",
        elems: 64,
        kernels: 5000,
        iters: 10,
    },
    Config {
        name: "64k x1000",
        elems: 64 * 1024,
        kernels: 1000,
        iters: 20,
    },
    Config {
        name: "256k x1000",
        elems: 256 * 1024,
        kernels: 1000,
        iters: 10,
    },
    Config {
        name: "1m x500",
        elems: 1024 * 1024,
        kernels: 500,
        iters: 10,
    },
    Config {
        name: "1m x2000",
        elems: 1024 * 1024,
        kernels: 2000,
        iters: 5,
    },
];

fn run_pass(client: &ComputeClient<HipRuntime>, a: &Handle, b: &Handle, cfg: &Config) {
    let cube_dim = 256usize;
    let cubes = cfg.elems.div_ceil(cube_dim) as u32;
    for i in 0..cfg.kernels {
        let (src, dst) = if i % 2 == 0 { (a, b) } else { (b, a) };
        add_one_tensor::launch(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(cube_dim as u32),
            unsafe { TensorArg::from_raw_parts(src.clone(), [1].into(), [cfg.elems].into()) },
            unsafe { TensorArg::from_raw_parts(dst.clone(), [1].into(), [cfg.elems].into()) },
        );
    }
}

fn sync(client: &ComputeClient<HipRuntime>, h: &Handle) {
    let _ = client.read_one(h.clone()).unwrap();
}

struct Measure {
    issue: Duration,
    total: Duration,
}

fn per_kernel(d: Duration, cfg: &Config) -> f64 {
    d.as_secs_f64() * 1e6 / (cfg.iters * cfg.kernels) as f64
}

fn per_pass(d: Duration, cfg: &Config) -> Duration {
    d / cfg.iters as u32
}

fn bench_config(client: &ComputeClient<HipRuntime>, cfg: &Config) {
    let a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; cfg.elems]));
    let b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; cfg.elems]));

    // Warm compile, pools, and the info cache.
    run_pass(client, &a, &b, cfg);
    sync(client, &a);

    // ---- Normal launch path ----
    let start = Instant::now();
    for _ in 0..cfg.iters {
        run_pass(client, &a, &b, cfg);
    }
    let issue = start.elapsed();
    sync(client, &a);
    let normal = Measure {
        issue,
        total: start.elapsed(),
    };

    // ---- Graph replay path ----
    client.graph_prepare().expect("graph_prepare");
    run_pass(client, &a, &b, cfg);
    sync(client, &a);
    client.start_capture().expect("start_capture");
    run_pass(client, &a, &b, cfg);
    let capture_start = Instant::now();
    let graph = client.stop_capture().expect("stop_capture");
    let capture = capture_start.elapsed();

    // Warm one replay.
    unsafe { graph.replay() };
    sync(client, &a);

    let start = Instant::now();
    for _ in 0..cfg.iters {
        unsafe { graph.replay() };
    }
    let issue = start.elapsed();
    sync(client, &a);
    let replay = Measure {
        issue,
        total: start.elapsed(),
    };

    println!(
        "{:>10} | {:>9.2?} | {:>7.2} -> {:>5.2} ({:>4.1}x) | {:>7.2} -> {:>5.2} ({:>4.1}x) | {:>9.2?} -> {:>9.2?} | {:>8.2?}",
        cfg.name,
        per_pass(normal.total, cfg),
        per_kernel(normal.issue, cfg),
        per_kernel(replay.issue, cfg),
        normal.issue.as_secs_f64() / replay.issue.as_secs_f64(),
        per_kernel(normal.total, cfg),
        per_kernel(replay.total, cfg),
        normal.total.as_secs_f64() / replay.total.as_secs_f64(),
        per_pass(normal.total, cfg),
        per_pass(replay.total, cfg),
        capture,
    );
}

fn main() {
    let client = HipRuntime::client(&Default::default());

    println!(
        "{:>10} | {:>9} | {:^28} | {:^28} | {:^25} | {:>8}",
        "config",
        "pass time",
        "issue µs/kernel (speedup)",
        "e2e µs/kernel (speedup)",
        "pass: normal -> replay",
        "capture"
    );

    for cfg in CONFIGS {
        bench_config(&client, cfg);
    }
}
