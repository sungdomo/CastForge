module @general_passes_demo {
  // Private helper function that will be inlined and then eliminated by symbol-dce
  func.func private @scale_and_bias(
    %x: tensor<4x8xf32>,
    %bias: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4x8xf32> loc("scale_const")
    %scaled = stablehlo.multiply %x, %two : tensor<4x8xf32> loc("scale_mul")
    %res = stablehlo.add %scaled, %bias : tensor<4x8xf32> loc("scale_add")
    return %res : tensor<4x8xf32>
  }

  // Unused private helper function that symbol-dce removes immediately
  func.func private @dead_helper(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %x : tensor<4x8xf32>
  }

  func.func @main(%arg0: tensor<4x8xf32>, %bias: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %called = func.call @scale_and_bias(%arg0, %bias)
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32> loc("call_site")
    %c1 = stablehlo.constant dense<10.0> : tensor<4x8xf32>
    %c2 = stablehlo.constant dense<5.0> : tensor<4x8xf32>
    %pred = stablehlo.compare %c1, %c2, GT
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
    %out = stablehlo.select %pred, %called, %arg0
      : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %out : tensor<4x8xf32>
  }
}
