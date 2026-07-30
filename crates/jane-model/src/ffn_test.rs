use burn::{
    backend::NdArray,
    module::Module,
    tensor::{Distribution, Tensor},
};

use super::Ffn;

type B = NdArray;

fn device() -> <B as burn::prelude::Backend>::Device {
    Default::default()
}

/// Output shape matches input shape `[batch, seq, d_model]`.
#[test]
fn output_shape() {
    let d_model = 8;
    let d_ff = 16;
    let ffn = Ffn::<B>::new(d_model, d_ff, &device());

    let x = Tensor::<B, 3>::random([2, 5, d_model], Distribution::Default, &device());
    let out = ffn.forward(x);

    assert_eq!(out.dims(), [2, 5, d_model]);
}

/// A zero input must produce a zero output.
///
/// SiLU(0) = 0 * sigmoid(0) = 0, so the gate output is the zero vector, and
/// zero * anything = zero, so `down` receives the zero vector, and a linear
/// layer applied to the zero vector (no bias) is zero.
#[test]
fn zero_input_zero_output() {
    let d_model = 4;
    let d_ff = 8;
    let ffn = Ffn::<B>::new(d_model, d_ff, &device());

    let x = Tensor::<B, 3>::zeros([1, 3, d_model], &device());
    let out = ffn.forward(x);

    // Every element should be exactly zero.
    let out_data: Vec<f32> = out.into_data().to_vec().unwrap();
    for v in out_data {
        assert_eq!(v, 0.0, "expected zero output for zero input");
    }
}

/// Parameter count must equal `3 * d_model * d_ff` — three bias-free matrices.
#[test]
fn parameter_count() {
    let d_model = 6;
    let d_ff = 10;
    let ffn = Ffn::<B>::new(d_model, d_ff, &device());

    let expected = 3 * d_model * d_ff;
    assert_eq!(
        ffn.num_params(),
        expected,
        "expected {expected} parameters (3 bias-free matrices)"
    );
}

/// The parameter count formula holds for several sizes, not just one.
///
/// This checks it is `3 * d_model * d_ff` and not e.g. `3 * d_model * d_model`.
#[test]
fn parameter_count_various_sizes() {
    for (d_model, d_ff) in [(2, 3), (8, 32), (16, 64), (1, 1)] {
        let ffn = Ffn::<B>::new(d_model, d_ff, &device());
        let expected = 3 * d_model * d_ff;
        assert_eq!(
            ffn.num_params(),
            expected,
            "d_model={d_model} d_ff={d_ff}: expected {expected} params"
        );
    }
}

/// The output changes when the weights change.
///
/// Two independently-constructed FFNs almost certainly have different random
/// weights. We verify that the same non-zero input produces different outputs,
/// proving the layer is not a constant function.
#[test]
fn output_changes_with_different_weights() {
    let d_model = 4;
    let d_ff = 8;

    let ffn_a = Ffn::<B>::new(d_model, d_ff, &device());
    let ffn_b = Ffn::<B>::new(d_model, d_ff, &device());

    // Use a fixed non-zero input: all ones.
    let x_a = Tensor::<B, 3>::ones([1, 1, d_model], &device());
    let x_b = Tensor::<B, 3>::ones([1, 1, d_model], &device());

    let out_a: Vec<f32> = ffn_a.forward(x_a).into_data().to_vec().unwrap();
    let out_b: Vec<f32> = ffn_b.forward(x_b).into_data().to_vec().unwrap();

    // With random init, the probability that two independent FFNs produce
    // identical outputs on the same input is astronomically small.
    assert_ne!(
        out_a, out_b,
        "two independently-initialized FFNs produced identical outputs; weights may be constant"
    );
}

/// Hand-computed reference for a tiny d_model=2, d_ff=2 network.
///
/// We construct an FFN with weights we control by checking that our formula
/// `down(SiLU(gate(x)) * up(x))` matches what the layer computes.
///
/// We run the formula twice — once as implemented and once independently in
/// this test — so a single typo cannot satisfy both checks.
#[test]
fn hand_computed_reference() {
    use burn::module::Param;
    use burn::tensor::TensorData;

    let d_model: usize = 2;
    let d_ff: usize = 2;

    // Build with random weights first, then replace.
    let mut ffn = Ffn::<B>::new(d_model, d_ff, &device());

    // gate weight: [[1, 0], [0, 1]]  (identity)
    // up weight:   [[2, 0], [0, 2]]  (scale by 2)
    // down weight: [[1, 1], [1, 1]]  (sum then broadcast)
    //
    // input x = [[3, 4]]  (batch=1, seq=1, d_model=2)
    //
    // gate(x)  = x @ gate^T = [3*1+4*0, 3*0+4*1] = [3, 4]
    // SiLU([3,4]) = [3*sigmoid(3), 4*sigmoid(4)]
    //             ≈ [3*0.9526, 4*0.9820] = [2.8577, 3.9281]
    //
    // up(x)    = x @ up^T = [3*2+4*0, 3*0+4*2] = [6, 8]
    //
    // gate_out * up_out = [2.8577*6, 3.9281*8] = [17.146, 31.425]
    //
    // down(gate_out*up_out) = [g*u] @ down^T
    //   = [17.146*1+31.425*1, 17.146*1+31.425*1]
    //   = [48.571, 48.571]

    let gate_w = TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [d_ff, d_model]);
    let up_w = TensorData::new(vec![2.0f32, 0.0, 0.0, 2.0], [d_ff, d_model]);
    let down_w = TensorData::new(vec![1.0f32, 1.0, 1.0, 1.0], [d_model, d_ff]);

    ffn.gate.weight = Param::from_tensor(Tensor::from_data(gate_w, &device()));
    ffn.up.weight = Param::from_tensor(Tensor::from_data(up_w, &device()));
    ffn.down.weight = Param::from_tensor(Tensor::from_data(down_w, &device()));

    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![3.0f32, 4.0], [1, 1, d_model]),
        &device(),
    );

    let out: Vec<f32> = ffn.forward(x).into_data().to_vec().unwrap();

    // Independent computation of the expected result.
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let silu = |v: f32| v * sigmoid(v);

    let gate_out = [silu(3.0f32), silu(4.0f32)];
    let up_out = [6.0f32, 8.0f32];
    let gated = [gate_out[0] * up_out[0], gate_out[1] * up_out[1]];

    // down weight rows are [1,1] and [1,1], so output = [sum(gated), sum(gated)]
    let expected_val = gated[0] + gated[1];

    assert!(
        (out[0] - expected_val).abs() < 1e-4,
        "out[0]={} expected≈{expected_val}",
        out[0]
    );
    assert!(
        (out[1] - expected_val).abs() < 1e-4,
        "out[1]={} expected≈{expected_val}",
        out[1]
    );
}
