// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xui32>, tensor<4x2xui32>)
    %1 = call @expected() : () -> tensor<3x2xui32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xui32>, tensor<4x2xui32>) -> tensor<3x2xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xui32>, tensor<3x2xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xui32>, tensor<4x2xui32>) {
    %0 = stablehlo.constant dense<[[2, 4, 2, 4], [1, 2, 1, 0], [1, 1, 1, 0]]> : tensor<3x4xui32>
    %1 = stablehlo.constant dense<[[1, 9], [1, 6], [0, 0], [1, 0]]> : tensor<4x2xui32>
    return %0, %1 : tensor<3x4xui32>, tensor<4x2xui32>
  }
  func.func private @expected() -> tensor<3x2xui32> {
    %0 = stablehlo.constant dense<[[10, 42], [3, 21], [2, 15]]> : tensor<3x2xui32>
    return %0 : tensor<3x2xui32>
  }
}

