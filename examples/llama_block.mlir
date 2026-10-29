module @llama_block {
  func.func @forward(
    %x: tensor<2x4xf32>,
    %res_in: tensor<2x4xf32>,
    %gamma: tensor<2x4xf32>,
    %wq: tensor<4x4xf32>,
    %wk: tensor<4x4xf32>,
    %wv: tensor<4x4xf32>,
    %cos: tensor<2x4xf32>,
    %sin: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %zero_half = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %zero_2x4 = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %one_2x4 = stablehlo.constant dense<1.0> : tensor<2x4xf32>
    %four = stablehlo.constant dense<4.0> : tensor<2xf32>
    %eps = stablehlo.constant dense<0.00001> : tensor<2xf32>
    %scale = stablehlo.constant dense<0.5> : tensor<2x2xf32>
    %w_gate = stablehlo.constant dense<0.25> : tensor<4x4xf32>
    %w_up = stablehlo.constant dense<0.5> : tensor<4x4xf32>

    // 1. Pre-Attention Residual + RMSNorm -> fused into castforge.fused_residual_rmsnorm
    %h = stablehlo.add %x, %res_in : tensor<2x4xf32>
    %sq = stablehlo.multiply %h, %h : tensor<2x4xf32>
    %sum_sq = stablehlo.reduce %sq, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean_sq = stablehlo.divide %sum_sq, %four : tensor<2xf32>
    %var_eps = stablehlo.add %mean_sq, %eps : tensor<2xf32>
    %rstd_1d = stablehlo.rsqrt %var_eps : tensor<2xf32>
    %rstd = stablehlo.broadcast_in_dim %rstd_1d, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %norm0 = stablehlo.multiply %h, %rstd : tensor<2x4xf32>
    %norm = stablehlo.multiply %norm0, %gamma : tensor<2x4xf32>

    // 2. Q, K, V Projections
    %q0 = stablehlo.dot_general %norm, %wq, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %k = stablehlo.dot_general %norm, %wk, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %v = stablehlo.dot_general %norm, %wv, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>

    // 3. Rotary Position Embedding (RoPE) on Q -> fused into castforge.fused_rope
    %q_lo = stablehlo.slice %q0
      {start_indices = [0, 0], limit_indices = [2, 2], strides = [1, 1]}
      : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %q_hi = stablehlo.slice %q0
      {start_indices = [0, 2], limit_indices = [2, 4], strides = [1, 1]}
      : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %neg_hi = stablehlo.subtract %zero_half, %q_hi : tensor<2x2xf32>
    %q_rot = stablehlo.concatenate %neg_hi, %q_lo {dimension = 1}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    %qc = stablehlo.multiply %q0, %cos : tensor<2x4xf32>
    %qs = stablehlo.multiply %q_rot, %sin : tensor<2x4xf32>
    %q = stablehlo.add %qc, %qs : tensor<2x4xf32>

    // 4. Scaled Dot-Product Attention -> fused into castforge.fused_attention
    %kt = stablehlo.transpose %k, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %qk = stablehlo.dot_general %q, %kt, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %scores = stablehlo.multiply %qk, %scale : tensor<2x2xf32>
    %ex = stablehlo.exponential %scores : tensor<2x2xf32>
    %sum = stablehlo.reduce %ex, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %denom = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x2xf32>
    %probs = stablehlo.divide %ex, %denom : tensor<2x2xf32>
    %attn = stablehlo.dot_general %probs, %v, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // 5. Gated-MLP (Dual Gate/Up + SwiGLU) -> fused into castforge.fused_gated_mlp
    %gate = stablehlo.dot_general %attn, %w_gate, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %up = stablehlo.dot_general %attn, %w_up, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %neg_g = stablehlo.subtract %zero_2x4, %gate : tensor<2x4xf32>
    %exp_g = stablehlo.exponential %neg_g : tensor<2x4xf32>
    %den_g = stablehlo.add %one_2x4, %exp_g : tensor<2x4xf32>
    %silu = stablehlo.divide %gate, %den_g : tensor<2x4xf32>
    %ffn = stablehlo.multiply %silu, %up : tensor<2x4xf32>
    func.return %ffn : tensor<2x4xf32>
  }
}
