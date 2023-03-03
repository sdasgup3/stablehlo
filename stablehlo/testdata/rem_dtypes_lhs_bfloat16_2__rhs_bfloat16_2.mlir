// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xbf16>, tensor<2xbf16>)
    %1 = call @expected() : () -> tensor<2xbf16>
    %2 = stablehlo.remainder %0#0, %0#1 : tensor<2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[8.359380e-01, -1.812500e+00]> : tensor<2xbf16>
    %1 = stablehlo.constant dense<[-1.382810e+00, 1.742190e+00]> : tensor<2xbf16>
    return %0, %1 : tensor<2xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<2xbf16> {
    %0 = stablehlo.constant dense<[8.359380e-01, -7.031250e-02]> : tensor<2xbf16>
    return %0 : tensor<2xbf16>
  }
}
