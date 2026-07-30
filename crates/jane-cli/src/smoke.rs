//! Phase 0 backend gate.
//!
//! Answers one question: can this machine allocate a tensor, run a matmul and
//! read the result back? On the RTX 5070 Ti (Blackwell, sm_120) that also proves
//! CubeCL's runtime NVRTC compilation targets this compute capability — the
//! single biggest technical risk in the project (ROADMAP §6).

use anyhow::{Context, Result};
use burn::tensor::{Distribution, Tensor, backend::Backend};

use crate::BackendChoice;

pub fn run(backend: BackendChoice, size: usize) -> Result<()> {
    match backend {
        BackendChoice::Ndarray => {
            let device = <burn::backend::NdArray as Backend>::Device::default();
            check::<burn::backend::NdArray>("ndarray (CPU)", &device, size)
        }
        BackendChoice::Cuda => run_cuda(size),
    }
}

#[cfg(feature = "cuda")]
fn run_cuda(size: usize) -> Result<()> {
    let device = burn::backend::cuda::CudaDevice::default();
    check::<burn::backend::Cuda<f32, i32>>("cuda", &device, size)
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(_size: usize) -> Result<()> {
    anyhow::bail!(
        "this binary was built without CUDA support; rebuild with:\n  \
         ./scripts/x cargo run -p jane-cli --features cuda -- smoke --backend cuda"
    )
}

/// Multiply two random matrices and verify the result is finite and plausible.
///
/// A kernel that fails to compile shows up as an error or a panic here; one that
/// compiles but computes garbage shows up as a NaN or a wildly wrong magnitude,
/// which the assertions below catch.
fn check<B: Backend>(label: &str, device: &B::Device, size: usize) -> Result<()> {
    println!("backend : {label}");
    println!("device  : {device:?}");

    // Uniform [0, 1) inputs, so each output element is a sum of `size` products
    // of two such values: expected value size/4, and strictly within [0, size].
    let a = Tensor::<B, 2>::random([size, size], Distribution::Default, device);
    let b = Tensor::<B, 2>::random([size, size], Distribution::Default, device);

    // Warm-up pass, deliberately untimed. CubeCL compiles kernels through NVRTC
    // on first use, and that compile is far more expensive than the matmul —
    // timing it would report roughly the compiler's speed, not the GPU's.
    let warm = a.clone().matmul(b.clone());
    let _ = warm.mean().into_scalar();

    let started = std::time::Instant::now();
    let c = a.matmul(b);
    let dims = c.dims();
    anyhow::ensure!(dims == [size, size], "unexpected output shape {dims:?}");

    // into_scalar() forces the lazily-queued kernels to execute and come back,
    // so this timing spans the actual work rather than just the enqueue.
    let mean = c
        .mean()
        .into_scalar()
        .to_string()
        .parse::<f64>()
        .context("failed to read the result back from the device")?;
    let elapsed = started.elapsed();

    anyhow::ensure!(
        mean.is_finite(),
        "matmul produced a non-finite mean: {mean}"
    );
    anyhow::ensure!(
        mean > 0.0 && mean < size as f64,
        "matmul mean {mean} is outside the plausible range (0, {size}) — \
         the kernel compiled but is computing garbage"
    );

    let expected = size as f64 / 4.0;
    let error = (mean - expected).abs() / expected;
    anyhow::ensure!(
        error < 0.1,
        "matmul mean {mean} deviates {:.1}% from the expected {expected}",
        error * 100.0
    );

    let flops = 2.0 * (size as f64).powi(3);
    println!("matmul  : {size}x{size} in {elapsed:.2?}");
    println!("mean    : {mean:.4} (expected ~{expected:.4})");
    println!(
        "throughput: {:.1} GFLOP/s",
        flops / elapsed.as_secs_f64() / 1e9
    );
    println!("\nOK — {label} allocates, computes and reads back correctly.");

    Ok(())
}
