// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<2x7xf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -2], high = [0, -2], interior = [0, 4] : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x7xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x7xf16>, tensor<2x7xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf16>, tensor<f16>) {
    %0 = stablehlo.constant dense<[[-1.026150e-03, 1.046660e-04, 1.714230e-04], [-1.002310e-03, -5.316730e-04, 4.243850e-04]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    return %0, %1 : tensor<2x3xf16>, tensor<f16>
  }
  func.func private @expected() -> tensor<2x7xf16> {
    %0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.046660e-04, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -5.316730e-04, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<2x7xf16>
    return %0 : tensor<2x7xf16>
  }
}
