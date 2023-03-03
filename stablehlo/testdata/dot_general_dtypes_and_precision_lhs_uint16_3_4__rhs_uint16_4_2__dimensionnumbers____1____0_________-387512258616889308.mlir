// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xui16>, tensor<4x2xui16>)
    %1 = call @expected() : () -> tensor<3x2xui16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<3x4xui16>, tensor<4x2xui16>) -> tensor<3x2xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xui16>, tensor<3x2xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xui16>, tensor<4x2xui16>) {
    %0 = stablehlo.constant dense<[[1, 0, 0, 2], [2, 1, 3, 1], [1, 6, 0, 0]]> : tensor<3x4xui16>
    %1 = stablehlo.constant dense<[[2, 6], [2, 4], [0, 0], [0, 0]]> : tensor<4x2xui16>
    return %0, %1 : tensor<3x4xui16>, tensor<4x2xui16>
  }
  func.func private @expected() -> tensor<3x2xui16> {
    %0 = stablehlo.constant dense<[[2, 6], [6, 16], [14, 30]]> : tensor<3x2xui16>
    return %0 : tensor<3x2xui16>
  }
}
