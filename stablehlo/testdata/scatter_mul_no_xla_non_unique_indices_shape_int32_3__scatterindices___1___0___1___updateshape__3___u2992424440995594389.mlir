// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3xi32>, tensor<3xi32>)
    %2 = call @expected() : () -> tensor<3xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<3xi32>, tensor<3x1xi32>, tensor<3xi32>) -> tensor<3xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xi32>, tensor<3xi32>) {
    %0 = stablehlo.constant dense<[-3, 0, -2]> : tensor<3xi32>
    %1 = stablehlo.constant dense<[2, -3, -4]> : tensor<3xi32>
    return %0, %1 : tensor<3xi32>, tensor<3xi32>
  }
  func.func private @expected() -> tensor<3xi32> {
    %0 = stablehlo.constant dense<[9, 0, -2]> : tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}

