module @mlp_block {
  func.func @forward(%arg0: tensor<4x8xf32>, %bias: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // Redundant double transpose (1,0 followed by 1,0 cancels out to identity 0,1)
    %t1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %t2 = stablehlo.transpose %t1, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    // Redundant chained reshapes back to original shape
    %r1 = stablehlo.reshape %t2 : (tensor<4x8xf32>) -> tensor<32xf32>
    %r2 = stablehlo.reshape %r1 : (tensor<32xf32>) -> tensor<4x8xf32>
    // Duplicate subexpression (%dup_add is identical to %sum)
    %sum = stablehlo.add %r2, %bias : tensor<4x8xf32>
    %dup_add = stablehlo.add %r2, %bias : tensor<4x8xf32>
    // Dead operation (%unused_mul is never referenced)
    %unused_mul = stablehlo.multiply %dup_add, %bias : tensor<4x8xf32>
    // ReLU pattern: add + maximum(x, 0) -> candidate for fuse-elementwise
    %relu = stablehlo.maximum %sum, %zero : tensor<4x8xf32>
    return %relu : tensor<4x8xf32>
  }
}
