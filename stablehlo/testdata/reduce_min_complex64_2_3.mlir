// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %cst = stablehlo.constant dense<(0x7F800000,0.000000e+00)> : tensor<complex<f32>>
    %2 = stablehlo.reduce(%0 init: %cst) across dimensions = [0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
     reducer(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>)  {
      %3 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %4 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %5 = stablehlo.compare  EQ, %3, %4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %3, %4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %8 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %9 = stablehlo.compare  LT, %7, %8,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.select %5, %9, %6 : tensor<i1>, tensor<i1>
      %11 = stablehlo.select %10, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %11 : tensor<complex<f32>>
    }
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> ()
    return %2 : tensor<3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(1.45971477,-0.530904412), (3.91971016,1.38173568), (0.591264247,-0.429937601)], [(-4.28235674,6.83627414), (3.58333111,2.12905931), (0.620722293,3.66264701)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-4.28235674,6.83627414), (3.58333111,2.12905931), (0.591264247,-0.429937601)]> : tensor<3xcomplex<f32>>
    return %cst : tensor<3xcomplex<f32>>
  }
}