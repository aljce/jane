// Tests for RmsNorm.
//
// All tests use the NdArray backend, which requires no GPU and passes offline.

use burn::backend::NdArray;
use burn::module::Module;
use burn::tensor::Tensor;

use super::RmsNorm;

type B = NdArray<f32>;

fn device() -> <B as burn::prelude::Backend>::Device {
    Default::default()
}

// ── construction ────────────────────────────────────────────────────────────

#[test]
fn gamma_shape_and_values() {
    let norm = RmsNorm::<B>::new(8, 1e-5, &device());
    let g = norm.gamma.val();
    assert_eq!(g.dims(), [8], "gamma must have shape [d_model]");
    let vals: Vec<f32> = g.into_data().to_vec().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - 1.0_f32).abs() < 1e-6, "gamma[{i}] = {v}, expected 1.0");
    }
}

#[test]
fn eps_stored_correctly() {
    let eps = 1e-8;
    let norm = RmsNorm::<B>::new(4, eps, &device());
    // eps is stored as-is; compare with the value we passed in.
    assert!((norm.eps - eps).abs() < f64::EPSILON);
}

// ── forward: hand-computed reference ────────────────────────────────────────

#[test]
fn hand_computed_reference() {
    // Input: [1, 1, 4] = [[[ 1, 2, 3, 4 ]]]
    // mean(x²) = (1 + 4 + 9 + 16) / 4 = 7.5
    // eps = 1e-6 (small enough it barely moves the result)
    // rms = sqrt(7.5 + 1e-6)
    // expected[i] = x[i] / rms  (gamma = 1.0)
    let eps = 1e-6_f64;
    let norm = RmsNorm::<B>::new(4, eps, &device());

    let raw = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device());
    let x = raw.reshape([1, 1, 4]);

    let out = norm.forward(x);
    let vals: Vec<f32> = out.into_data().to_vec().unwrap();

    let mean_sq = (1.0_f64 + 4.0 + 9.0 + 16.0) / 4.0; // 7.5
    let rms = (mean_sq + eps).sqrt();
    let expected: Vec<f64> = [1.0_f64, 2.0, 3.0, 4.0].iter().map(|v| v / rms).collect();

    for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*got as f64) - exp).abs() < 1e-5,
            "output[{i}]: got {got}, expected {exp}"
        );
    }
}

// ── forward: shape preservation ─────────────────────────────────────────────

#[test]
fn output_shape_matches_input() {
    let norm = RmsNorm::<B>::new(16, 1e-5, &device());
    let x = Tensor::<B, 3>::ones([3, 7, 16], &device());
    let out = norm.forward(x);
    assert_eq!(out.dims(), [3, 7, 16]);
}

// ── forward: constant-vector normalizes to constant ─────────────────────────

#[test]
fn constant_vector_yields_constant_output() {
    // A vector of identical values: RMS(c, c, ..., c) = c, so output = 1.0 * gamma.
    let d = 8_usize;
    let norm = RmsNorm::<B>::new(d, 1e-8, &device());

    // Use constant 3.0
    let x = Tensor::<B, 3>::full([1, 1, d], 3.0_f32, &device());
    let out: Vec<f32> = norm.forward(x).into_data().to_vec().unwrap();

    // RMS of [3, 3, ..., 3] = 3, so normalized = 1.0, scaled by gamma=1 → 1.0
    let eps = 1e-8_f64;
    let rms = (9.0_f64 + eps).sqrt();
    let expected = 3.0_f64 / rms;
    for (i, v) in out.iter().enumerate() {
        assert!(
            ((*v as f64) - expected).abs() < 1e-5,
            "output[{i}] = {v}, expected ~{expected}"
        );
    }
}

// ── forward: gamma affects output ───────────────────────────────────────────

#[test]
fn output_changes_when_gamma_changes() {
    // Build one norm with gamma=1, another by hand with gamma scaled to 2.
    let d = 4_usize;
    let norm1 = RmsNorm::<B>::new(d, 1e-5, &device());

    // Manually scale gamma by 2.
    let mut norm2 = RmsNorm::<B>::new(d, 1e-5, &device());
    let new_gamma = norm2.gamma.val() * Tensor::<B, 1>::full([d], 2.0_f32, &device());
    norm2.gamma = burn::module::Param::from_tensor(new_gamma);

    let raw = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device());
    let x1 = raw.clone().reshape([1, 1, d]);
    let x2 = raw.reshape([1, 1, d]);

    let out1: Vec<f32> = norm1.forward(x1).into_data().to_vec().unwrap();
    let out2: Vec<f32> = norm2.forward(x2).into_data().to_vec().unwrap();

    for (a, b) in out1.iter().zip(out2.iter()) {
        // out2 should be 2× out1
        assert!(
            ((b - 2.0 * a).abs()) < 1e-5,
            "expected out2[i]=2*out1[i], got {b} vs 2*{a}={}",
            2.0 * a
        );
    }
}

// ── forward: zeros don't produce NaN ────────────────────────────────────────

#[test]
fn zero_input_no_nan() {
    let norm = RmsNorm::<B>::new(4, 1e-5, &device());
    let x = Tensor::<B, 3>::zeros([1, 1, 4], &device());
    let out: Vec<f32> = norm.forward(x).into_data().to_vec().unwrap();
    for (i, v) in out.iter().enumerate() {
        assert!(!v.is_nan(), "output[{i}] is NaN on zero input");
        assert!(v.is_finite(), "output[{i}] is infinite on zero input");
    }
}

// ── forward: independent verification ───────────────────────────────────────

// Cross-check the formula against an independent calculation so a single typo
// cannot satisfy both the reference value and the property.
#[test]
fn formula_agrees_with_independent_computation() {
    let eps = 1e-5_f64;
    let norm = RmsNorm::<B>::new(3, eps, &device());

    let vals = [2.0_f32, -1.0, 3.0];
    let x = Tensor::<B, 1>::from_floats(vals, &device()).reshape([1, 1, 3]);
    let out: Vec<f32> = norm.forward(x).into_data().to_vec().unwrap();

    // Independent computation in pure Rust.
    let sum_sq: f64 = vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let mean_sq = sum_sq / 3.0;
    let rms = (mean_sq + eps).sqrt();
    let expected: Vec<f64> = vals.iter().map(|v| (*v as f64) / rms).collect();

    for (i, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*got as f64) - exp).abs() < 1e-5,
            "index {i}: got {got}, expected {exp}"
        );
    }
}
