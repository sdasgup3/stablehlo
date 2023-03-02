// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5xui16>, tensor<2xui16>)
    %2 = call @expected() : () -> tensor<5xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      stablehlo.return %arg1 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xui16>, tensor<2x1xi32>, tensor<2xui16>) -> tensor<5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xui16>, tensor<5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xui16>, tensor<2xui16>) {
    %0 = stablehlo.constant dense<[0, 2, 1, 0, 5]> : tensor<5xui16>
    %1 = stablehlo.constant dense<0> : tensor<2xui16>
    return %0, %1 : tensor<5xui16>, tensor<2xui16>
  }
  func.func private @expected() -> tensor<5xui16> {
    %0 = stablehlo.constant dense<[0, 2, 0, 0, 5]> : tensor<5xui16>
    return %0 : tensor<5xui16>
  }
}
