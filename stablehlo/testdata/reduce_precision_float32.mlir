// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<f32>
    %1 = call @expected() : () -> tensor<f32>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<f32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<f32>, tensor<f32>) -> ()
    return %2 : tensor<f32>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-0.81502068> : tensor<f32>
    return %cst : tensor<f32>
  }
  func.func private @expected() -> (tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-0.81502068> : tensor<f32>
    return %cst : tensor<f32>
  }
}
