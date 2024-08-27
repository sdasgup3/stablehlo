// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<15xi16>
    %1 = call @expected() : () -> tensor<i32>
    %2 = call @argmin(%0) : (tensor<15xi16>) -> tensor<i32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<i32>, tensor<i32>) -> ()
    return %2 : tensor<i32>
  }
  func.func private @inputs() -> (tensor<15xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[-5, 2, -4, 0, 0, 3, 0, 3, 4, -4, 7, 1, -4, -2, 0]> : tensor<15xi16>
    return %c : tensor<15xi16>
  }
  func.func private @expected() -> (tensor<i32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    return %c : tensor<i32>
  }
  func.func private @argmin(%arg0: tensor<15xi16>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<15xi32>
    %c = stablehlo.constant dense<32767> : tensor<i16>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %c), (%0 init: %c_0) across dimensions = [0] : (tensor<15xi16>, tensor<15xi32>, tensor<i16>, tensor<i32>) -> (tensor<i16>, tensor<i32>)
     reducer(%arg1: tensor<i16>, %arg3: tensor<i16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  LT, %arg1, %arg3,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i16>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<i16>, tensor<i32>
    }
    return %1#1 : tensor<i32>
  }
}