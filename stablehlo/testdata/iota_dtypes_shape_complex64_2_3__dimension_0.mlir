// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %1 = stablehlo.iota dim = 0 : tensor<2x3xcomplex<f32>>
    %2 = stablehlo.custom_call @check.eq(%1, %0) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
}
