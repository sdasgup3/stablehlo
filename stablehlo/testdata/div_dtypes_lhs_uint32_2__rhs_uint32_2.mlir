// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<2xui32>)
    %1 = call @expected() : () -> tensor<2xui32>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<2xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<2xui32>) {
    %0 = stablehlo.constant dense<[9, 4]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[1, 2]> : tensor<2xui32>
    return %0, %1 : tensor<2xui32>, tensor<2xui32>
  }
  func.func private @expected() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[9, 2]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
}
