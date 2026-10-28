module @transformer_layer {
  func.func @rmsnorm_proj(%x: tensor<4x128xf32>, %w_norm: tensor<4x128xf32>, %w_proj: tensor<128x64xf32>) -> tensor<4x64xf32> {
    // Compile-time constant math that folds into a single eps constant (0.0 + 0.00001)
    %zero_c = stablehlo.constant dense<0.0> : tensor<4x1xf32>
    %eps_raw = stablehlo.constant dense<0.00001> : tensor<4x1xf32>
    %eps = stablehlo.add %zero_c, %eps_raw : tensor<4x1xf32>

    // Algebraic identities on input %x (x + 0 and x * 1 and no-op convert)
    %z_full = stablehlo.constant dense<0.0> : tensor<4x128xf32>
    %one_full = stablehlo.constant dense<1.0> : tensor<4x128xf32>
    %x1 = stablehlo.add %x, %z_full : tensor<4x128xf32>
    %x2 = stablehlo.multiply %x1, %one_full : tensor<4x128xf32>
    %x3 = stablehlo.convert %x2 : (tensor<4x128xf32>) -> tensor<4x128xf32>

    // Square + reduce across hidden dim
    %sq = stablehlo.multiply %x3, %x3 : tensor<4x128xf32>
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%sq init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x128xf32>, tensor<f32>) -> tensor<4xf32>
    %sum_2d = stablehlo.reshape %sum : (tensor<4xf32>) -> tensor<4x1xf32>
    %var_eps = stablehlo.add %sum_2d, %eps : tensor<4x1xf32>
    %inv_rms = stablehlo.rsqrt %var_eps : tensor<4x1xf32>
    %inv_bcast = stablehlo.broadcast_in_dim %inv_rms, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x128xf32>
    %normed = stablehlo.multiply %x3, %inv_bcast : tensor<4x128xf32>
    %scaled = stablehlo.multiply %normed, %w_norm : tensor<4x128xf32>

    // Linear projection via dot_general
    %out = stablehlo.dot_general %scaled, %w_proj {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]} : (tensor<4x128xf32>, tensor<128x64xf32>) -> tensor<4x64xf32>
    return %out : tensor<4x64xf32>
  }
}
