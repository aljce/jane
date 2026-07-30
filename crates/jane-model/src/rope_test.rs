use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use super::RotaryEmbedding;

type B = NdArray;

fn default_device() -> <B as burn::prelude::Backend>::Device {
    Default::default()
}

/// Helper: extract f32 values from a 4D tensor.
fn to_f32_vec(t: Tensor<B, 4>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("tensor data to_vec failed")
}

/// Helper: create a 4D tensor from a flat Vec<f32> with the given shape.
fn tensor4(vals: Vec<f32>, shape: [usize; 4]) -> Tensor<B, 4> {
    let device = default_device();
    Tensor::<B, 4>::from_data(TensorData::new(vals, shape.to_vec()), &device)
}

// ---------------------------------------------------------------------------
// Cache construction tests
// ---------------------------------------------------------------------------

#[test]
fn cache_output_shape() {
    let device = default_device();
    // Verify that the cache produces [max_seq_len, head_dim] by checking that
    // apply with seq_len == max_seq_len succeeds and the output has correct shape.
    let rope = RotaryEmbedding::<B>::new(32, 16, 10_000.0, &device);
    let x = Tensor::<B, 4>::zeros([1, 1, 32, 16], &device);
    let out = rope.apply(x, 0);
    assert_eq!(out.dims(), [1, 1, 32, 16]);

    // Also check with different dimensions.
    let rope2 = RotaryEmbedding::<B>::new(64, 8, 10_000.0, &device);
    let x2 = Tensor::<B, 4>::zeros([2, 4, 64, 8], &device);
    let out2 = rope2.apply(x2, 0);
    assert_eq!(out2.dims(), [2, 4, 64, 8]);
}

#[test]
fn position_zero_is_identity_via_cache() {
    // For position 0, angle = 0 * freq = 0 → cos(0)=1, sin(0)=0.
    // Applying RoPE at position 0 must be the identity transformation.
    let device = default_device();
    let head_dim = 8;
    let rope = RotaryEmbedding::<B>::new(16, head_dim, 10_000.0, &device);

    let vals: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let x = tensor4(vals.clone(), [1, 1, 1, head_dim]);
    let out = to_f32_vec(rope.apply(x, 0));

    for (i, (expected, got)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-5,
            "position 0 should be identity at dim {i}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn cache_values_in_range() {
    // cos and sin values are always in [-1, 1].  After applying to an all-ones
    // vector: out[2i] = cos - sin, out[2i+1] = cos + sin.
    // Both are bounded by |cos| + |sin| ≤ sqrt(2) < 1.5.
    let device = default_device();
    let head_dim = 8;
    let rope = RotaryEmbedding::<B>::new(64, head_dim, 10_000.0, &device);
    let x = Tensor::<B, 4>::ones([1, 1, 64, head_dim], &device);
    let out = to_f32_vec(rope.apply(x, 0));
    for v in &out {
        assert!(
            v.abs() <= 1.5,
            "output value {v} exceeds sqrt(2): cos/sin value likely out of [-1,1]"
        );
    }
}

#[test]
fn table_is_deterministic() {
    let device = default_device();
    let head_dim = 8usize;
    let vals: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();

    let rope1 = RotaryEmbedding::<B>::new(32, head_dim, 10_000.0, &device);
    let rope2 = RotaryEmbedding::<B>::new(32, head_dim, 10_000.0, &device);

    let out1 = to_f32_vec(rope1.apply(tensor4(vals.clone(), [1, 1, 1, head_dim]), 0));
    let out2 = to_f32_vec(rope2.apply(tensor4(vals, [1, 1, 1, head_dim]), 0));

    assert_eq!(
        out1, out2,
        "two calls with identical parameters must produce identical results"
    );
}

// ---------------------------------------------------------------------------
// Apply tests
// ---------------------------------------------------------------------------

#[test]
fn apply_zero_tensor_gives_zero() {
    let device = default_device();
    let rope = RotaryEmbedding::<B>::new(16, 8, 10_000.0, &device);
    let x = Tensor::<B, 4>::zeros([2, 4, 8, 8], &device);
    let out = to_f32_vec(rope.apply(x, 0));
    for v in &out {
        assert_eq!(*v, 0.0, "zero input must produce zero output, got {v}");
    }
}

#[test]
fn apply_position_zero_is_identity() {
    // At offset=0 (position 0), cos=1, sin=0 everywhere, so out == x.
    let device = default_device();
    let head_dim = 8;
    let vals: Vec<f32> = vec![1.0, -2.0, 3.14, 0.5, -1.0, 2.71, 0.0, 100.0];
    let rope = RotaryEmbedding::<B>::new(16, head_dim, 10_000.0, &device);
    let x = tensor4(vals.clone(), [1, 1, 1, head_dim]);
    let out = to_f32_vec(rope.apply(x, 0));

    for (i, (expected, got)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-4,
            "position 0 must be identity at dim {i}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn norm_preserved() {
    // RoPE is a block-diagonal 2D rotation, so it preserves the L2 norm of every vector.
    let device = default_device();
    let head_dim = 16;
    let rope = RotaryEmbedding::<B>::new(64, head_dim, 10_000.0, &device);

    let test_cases: &[Vec<f32>] = &[
        // unit vector in first component
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        // general vector
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ],
        // alternating signs
        vec![
            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        ],
    ];

    for (pos, vals) in test_cases.iter().enumerate() {
        let x = tensor4(vals.clone(), [1, 1, 1, head_dim]);
        // Use a non-trivial position so the rotation is non-identity.
        let out = to_f32_vec(rope.apply(x, pos + 1));

        let in_norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        let out_norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (in_norm - out_norm).abs() < 1e-3,
            "norm should be preserved at pos {pos}: |x|={in_norm:.6}, |Rx|={out_norm:.6}"
        );
    }
}

#[test]
fn output_shape_matches_input_shape() {
    let device = default_device();
    let rope = RotaryEmbedding::<B>::new(64, 32, 10_000.0, &device);
    let shape = [3, 8, 16, 32];
    let x = Tensor::<B, 4>::zeros(shape, &device);
    let out = rope.apply(x, 0);
    assert_eq!(out.dims(), shape);
}

#[test]
fn rotation_changes_with_position() {
    // The same input vector should produce different outputs at different positions.
    let device = default_device();
    let head_dim = 8;
    let rope = RotaryEmbedding::<B>::new(64, head_dim, 10_000.0, &device);

    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let out0 = to_f32_vec(rope.apply(tensor4(vals.clone(), [1, 1, 1, head_dim]), 0));
    let out10 = to_f32_vec(rope.apply(tensor4(vals, [1, 1, 1, head_dim]), 10));

    let all_equal = out0
        .iter()
        .zip(out10.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(
        !all_equal,
        "output at position 0 should differ from output at position 10"
    );
}

#[test]
fn offset_shifts_row_of_cos_sin() {
    // Two tokens with seq_len=2 at offset=5 should match independent tokens at offsets 5 and 6.
    let device = default_device();
    let head_dim = 8;
    let rope = RotaryEmbedding::<B>::new(32, head_dim, 10_000.0, &device);

    let vals_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vals_b: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];

    // Batch: [1, 1, 2, head_dim] starting at offset=5.
    let mut combined = vals_a.clone();
    combined.extend(vals_b.clone());
    let out_batch = to_f32_vec(rope.apply(tensor4(combined, [1, 1, 2, head_dim]), 5));

    // Individually.
    let out_a = to_f32_vec(rope.apply(tensor4(vals_a, [1, 1, 1, head_dim]), 5));
    let out_b = to_f32_vec(rope.apply(tensor4(vals_b, [1, 1, 1, head_dim]), 6));

    // First token in batch must match out_a.
    for (i, (batch_val, single_val)) in out_batch[..head_dim].iter().zip(out_a.iter()).enumerate() {
        assert!(
            (batch_val - single_val).abs() < 1e-4,
            "offset test: batch[0][{i}]={batch_val:.6} != single_a[{i}]={single_val:.6}"
        );
    }
    // Second token in batch must match out_b.
    for (i, (batch_val, single_val)) in out_batch[head_dim..].iter().zip(out_b.iter()).enumerate() {
        assert!(
            (batch_val - single_val).abs() < 1e-4,
            "offset test: batch[1][{i}]={batch_val:.6} != single_b[{i}]={single_val:.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// Numerical correctness against independently computed references
// ---------------------------------------------------------------------------

#[test]
fn known_angle_numerical_check() {
    // rope_theta=1.0, head_dim=2 → one frequency pair, freq[0]=1.0.
    // At offset=1 (position 1): angle = 1.0.
    // cos(1.0) ≈ 0.540302, sin(1.0) ≈ 0.841471.
    // Input x = [x0=1.0, x1=0.0]:
    //   out[0] = x0*cos - x1*sin = 1.0*cos = cos(1)
    //   out[1] = x1*cos + x0*sin = 0.0*cos + 1.0*sin = sin(1)
    let device = default_device();
    let rope = RotaryEmbedding::<B>::new(4, 2, 1.0, &device);
    let x = tensor4(vec![1.0_f32, 0.0], [1, 1, 1, 2]);
    let out = to_f32_vec(rope.apply(x, 1));

    let angle = 1.0_f32;
    let expected_0 = angle.cos(); // 1.0*cos - 0.0*sin
    let expected_1 = angle.sin(); // 0.0*cos + 1.0*sin

    assert!(
        (out[0] - expected_0).abs() < 1e-5,
        "out[0]: expected {expected_0:.6}, got {:.6}",
        out[0]
    );
    assert!(
        (out[1] - expected_1).abs() < 1e-5,
        "out[1]: expected {expected_1:.6}, got {:.6}",
        out[1]
    );
}

#[test]
fn head_dim_four_two_pairs_numerical() {
    // head_dim=4 → two pairs.
    // rope_theta=100: freq[0] = 1.0, freq[1] = 1/sqrt(100) = 0.1
    // At position 2: angle0 = 2.0, angle1 = 0.2.
    // Input: x = [x0, x1, x2, x3] = [3.0, 5.0, 7.0, 11.0]
    // Pair (0,1): out[0] = x0*cos(2) - x1*sin(2), out[1] = x1*cos(2) + x0*sin(2)
    // Pair (2,3): out[2] = x2*cos(0.2) - x3*sin(0.2), out[3] = x3*cos(0.2) + x2*sin(0.2)
    let device = default_device();
    let rope_theta = 100.0_f64;
    let rope = RotaryEmbedding::<B>::new(8, 4, rope_theta, &device);

    let x_vals = [3.0_f32, 5.0, 7.0, 11.0];
    let x = tensor4(x_vals.to_vec(), [1, 1, 1, 4]);
    let out = to_f32_vec(rope.apply(x, 2));

    let freq0 = 1.0_f32;
    let freq1 = (1.0_f64 / rope_theta.powf(0.5)) as f32; // 0.1
    let angle0 = 2.0_f32 * freq0;
    let angle1 = 2.0_f32 * freq1;

    let (c0, s0) = (angle0.cos(), angle0.sin());
    let (c1, s1) = (angle1.cos(), angle1.sin());

    let expected = [
        x_vals[0] * c0 - x_vals[1] * s0, // pair 0 even
        x_vals[1] * c0 + x_vals[0] * s0, // pair 0 odd
        x_vals[2] * c1 - x_vals[3] * s1, // pair 1 even
        x_vals[3] * c1 + x_vals[2] * s1, // pair 1 odd
    ];

    for (i, (e, g)) in expected.iter().zip(out.iter()).enumerate() {
        assert!(
            (e - g).abs() < 1e-4,
            "head_dim=4 test, dim {i}: expected {e:.6}, got {g:.6}"
        );
    }
}

#[test]
fn norm_agreement_with_independent_computation() {
    // Cross-check norm preservation with a hand-rolled rotation.
    // For head_dim=2, rope_theta=1, position p=1: angle=1.
    // Rotation matrix R = [[cos,-sin],[sin,cos]].
    // For x = [a, b]: Rx = [a*c - b*s, b*c + a*s].
    // |Rx|^2 = (a*c-b*s)^2 + (b*c+a*s)^2 = a^2(c^2+s^2) + b^2(s^2+c^2) = a^2+b^2 = |x|^2.
    let device = default_device();
    let rope = RotaryEmbedding::<B>::new(8, 2, 1.0, &device);

    // Several test vectors; verify burn computation matches formula.
    let test_vecs: &[[f32; 2]] = &[[1.0, 0.0], [0.0, 1.0], [3.0, 4.0], [-2.0, 5.0]];
    for &[a, b] in test_vecs {
        let x = tensor4(vec![a, b], [1, 1, 1, 2]);
        let out = to_f32_vec(rope.apply(x, 3)); // position 3, angle=3.0

        let angle = 3.0_f32;
        let c = angle.cos();
        let s = angle.sin();
        let expected_out0 = a * c - b * s;
        let expected_out1 = b * c + a * s;

        assert!(
            (out[0] - expected_out0).abs() < 1e-4,
            "[{a},{b}] pos=3 out[0]: expected {expected_out0:.6}, got {:.6}",
            out[0]
        );
        assert!(
            (out[1] - expected_out1).abs() < 1e-4,
            "[{a},{b}] pos=3 out[1]: expected {expected_out1:.6}, got {:.6}",
            out[1]
        );

        // Norm agreement.
        let in_norm = (a * a + b * b).sqrt();
        let out_norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "[{a},{b}] norm mismatch: |x|={in_norm:.6}, |Rx|={out_norm:.6}"
        );
    }
}
