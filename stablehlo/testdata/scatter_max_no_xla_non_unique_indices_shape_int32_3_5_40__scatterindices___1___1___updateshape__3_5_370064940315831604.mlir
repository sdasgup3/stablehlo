// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xi32>, tensor<3x5x2xi32>)
    %2 = call @expected() : () -> tensor<3x5x40xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xi32>, tensor<2x1xi32>, tensor<3x5x2xi32>) -> tensor<3x5x40xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xi32>, tensor<3x5x40xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xi32>, tensor<3x5x2xi32>) {
    %0 = stablehlo.constant dense<"0x0400000004000000FEFFFFFF00000000000000000400000003000000FFFFFFFFFCFFFFFF000000000000000000000000FDFFFFFF04000000FCFFFFFF0000000000000000030000000200000000000000020000000100000002000000000000000000000000000000FFFFFFFFFFFFFFFF00000000FBFFFFFF0000000000000000FEFFFFFF00000000FFFFFFFF0000000007000000FEFFFFFFFCFFFFFFFFFFFFFFFFFFFFFF03000000020000000200000000000000FBFFFFFFFFFFFFFF03000000000000000000000001000000FFFFFFFFFEFFFFFF0000000003000000FFFFFFFFFFFFFFFF020000000300000002000000FFFFFFFF0000000001000000000000000100000000000000FFFFFFFF0100000001000000FAFFFFFFFFFFFFFF020000000100000000000000FDFFFFFFFEFFFFFF03000000FFFFFFFF0600000005000000FEFFFFFF0100000000000000FCFFFFFF01000000FEFFFFFF00000000FCFFFFFF03000000FFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFF03000000FDFFFFFF0300000002000000000000000300000000000000FFFFFFFF00000000FEFFFFFF00000000F6FFFFFF03000000FEFFFFFF030000000300000004000000FFFFFFFFFDFFFFFF00000000FEFFFFFF01000000FDFFFFFF03000000FEFFFFFF01000000010000000000000003000000FEFFFFFF000000000000000000000000020000000000000002000000F9FFFFFF05000000FFFFFFFF0000000006000000FEFFFFFF01000000FCFFFFFF04000000FFFFFFFF00000000FFFFFFFF0900000003000000FEFFFFFF0100000001000000FFFFFFFF010000000100000003000000FFFFFFFF040000000200000000000000FEFFFFFFFFFFFFFF00000000FEFFFFFF02000000010000000000000002000000FFFFFFFF0000000001000000FAFFFFFFFFFFFFFF030000000600000000000000FEFFFFFF03000000FFFFFFFF0000000000000000FDFFFFFF0000000002000000FEFFFFFF01000000040000000400000007000000FCFFFFFF05000000000000000100000002000000FDFFFFFFFEFFFFFF0000000000000000FDFFFFFF0100000000000000060000000000000000000000FAFFFFFFFBFFFFFF01000000FFFFFFFF01000000010000000100000001000000FFFFFFFF0000000004000000FFFFFFFF02000000FDFFFFFFFEFFFFFF000000000200000000000000FFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFDFFFFFF0400000001000000000000000100000000000000060000000000000002000000070000000000000000000000FEFFFFFF0000000000000000FFFFFFFFFEFFFFFF01000000FEFFFFFF02000000000000000300000004000000FDFFFFFFFAFFFFFFFCFFFFFF01000000020000000400000002000000FCFFFFFF06000000FDFFFFFF010000000100000001000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFBFFFFFFFEFFFFFF00000000FFFFFFFFFFFFFFFF0300000000000000FFFFFFFF0200000000000000FEFFFFFF000000000000000000000000FDFFFFFF0100000000000000FDFFFFFF00000000FCFFFFFFFDFFFFFF0200000001000000FFFFFFFF0000000001000000FCFFFFFFFDFFFFFF020000000000000000000000020000000400000000000000000000000100000000000000FFFFFFFF0400000000000000FDFFFFFF03000000FDFFFFFF000000000600000001000000FFFFFFFFFEFFFFFF03000000FBFFFFFFFEFFFFFF00000000FEFFFFFF03000000FEFFFFFF0200000001000000FAFFFFFF0000000002000000000000000200000000000000FEFFFFFF00000000FEFFFFFFFDFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFF0400000004000000050000000200000002000000FFFFFFFF03000000020000000000000005000000020000000100000003000000FEFFFFFF02000000FFFFFFFF02000000FEFFFFFF010000000300000000000000FDFFFFFF03000000030000000200000000000000FBFFFFFF02000000FBFFFFFF0000000000000000FBFFFFFF03000000FEFFFFFF02000000FDFFFFFF0000000001000000FCFFFFFF00000000000000000000000001000000FFFFFFFF000000000200000000000000020000000000000000000000030000000000000004000000FEFFFFFF0600000000000000000000000000000009000000000000000100000000000000FDFFFFFF00000000FFFFFFFF0400000000000000FFFFFFFF0000000001000000FFFFFFFF06000000FFFFFFFF06000000020000000200000003000000020000000000000003000000010000000000000000000000000000000200000003000000FAFFFFFF020000000100000000000000FCFFFFFF00000000FEFFFFFF000000000000000004000000FFFFFFFF0700000000000000FEFFFFFF00000000FFFFFFFFFDFFFFFF0300000000000000FFFFFFFF00000000FDFFFFFF0200000000000000000000000100000000000000FDFFFFFFFFFFFFFFFFFFFFFF0300000000000000FFFFFFFF02000000FDFFFFFF0200000003000000000000000400000000000000FDFFFFFF00000000FEFFFFFF0200000002000000000000000100000002000000FBFFFFFF0600000001000000FCFFFFFF00000000FEFFFFFFFDFFFFFF040000000000000004000000FFFFFFFF00000000010000000100000000000000FDFFFFFF0400000000000000F9FFFFFFFDFFFFFF0000000002000000FFFFFFFF01000000FEFFFFFF01000000020000000000000003000000000000000000000001000000FFFFFFFFFFFFFFFF000000000000000005000000000000000200000004000000000000000300000004000000FCFFFFFF04000000010000000000000000000000FFFFFFFF00000000FFFFFFFF0000000000000000FEFFFFFF0200000000000000FEFFFFFF06000000FFFFFFFF01000000030000000300000001000000FDFFFFFFFEFFFFFF00000000000000000000000001000000FDFFFFFFFFFFFFFF0200000002000000FFFFFFFF02000000FDFFFFFF000000000000000000000000FFFFFFFF02000000FBFFFFFF04000000FEFFFFFFFFFFFFFF00000000000000000200000004000000FFFFFFFF02000000020000000000000002000000FDFFFFFFFFFFFFFF050000000500000001000000FFFFFFFF0000000000000000000000000000000003000000000000000200000000000000FCFFFFFFFDFFFFFF000000000400000000000000FFFFFFFF0200000002000000010000000300000007000000FCFFFFFF0100000000000000FEFFFFFFFBFFFFFF01000000020000000200000007000000"> : tensor<3x5x40xi32>
    %1 = stablehlo.constant dense<[[[-4, -3], [-3, -1], [2, -1], [0, -1], [3, 0]], [[4, 0], [-2, -2], [-1, -2], [2, -2], [-3, 1]], [[1, -5], [1, 1], [1, -1], [0, -3], [-1, 4]]]> : tensor<3x5x2xi32>
    return %0, %1 : tensor<3x5x40xi32>, tensor<3x5x2xi32>
  }
  func.func private @expected() -> tensor<3x5x40xi32> {
    %0 = stablehlo.constant dense<"0x0400000004000000FEFFFFFF00000000000000000400000003000000FFFFFFFFFCFFFFFF000000000000000000000000FDFFFFFF04000000FCFFFFFF0000000000000000030000000200000000000000020000000100000002000000000000000000000000000000FFFFFFFFFFFFFFFF00000000FBFFFFFF0000000000000000FEFFFFFF00000000FFFFFFFF0000000007000000FEFFFFFFFCFFFFFFFFFFFFFFFFFFFFFF03000000020000000200000000000000FBFFFFFFFFFFFFFF03000000000000000000000001000000FFFFFFFFFEFFFFFF0000000003000000FFFFFFFFFFFFFFFF020000000300000002000000FFFFFFFF0000000001000000000000000100000000000000FFFFFFFF0100000001000000FAFFFFFFFFFFFFFF020000000100000000000000FDFFFFFFFEFFFFFF03000000FFFFFFFF0600000005000000FEFFFFFF0200000000000000FCFFFFFF01000000FEFFFFFF00000000FCFFFFFF03000000FFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFF03000000FDFFFFFF0300000002000000000000000300000000000000FFFFFFFF00000000FEFFFFFF00000000F6FFFFFF03000000FEFFFFFF030000000300000004000000FFFFFFFFFDFFFFFF00000000FEFFFFFF01000000FDFFFFFF03000000FEFFFFFF01000000010000000000000003000000FEFFFFFF000000000000000000000000020000000000000002000000F9FFFFFF05000000FFFFFFFF0000000006000000FEFFFFFF01000000FCFFFFFF04000000FFFFFFFF00000000FFFFFFFF0900000003000000FEFFFFFF0100000001000000FFFFFFFF010000000100000003000000FFFFFFFF040000000200000000000000FEFFFFFFFFFFFFFF00000000FEFFFFFF02000000010000000000000003000000FFFFFFFF0000000001000000FAFFFFFFFFFFFFFF030000000600000000000000FEFFFFFF03000000FFFFFFFF0000000000000000FDFFFFFF0000000002000000FEFFFFFF01000000040000000400000007000000FCFFFFFF05000000000000000100000002000000FDFFFFFFFEFFFFFF0000000000000000FDFFFFFF0100000000000000060000000000000000000000FAFFFFFFFBFFFFFF010000000400000001000000010000000100000001000000FFFFFFFF0000000004000000FFFFFFFF02000000FDFFFFFFFEFFFFFF000000000200000000000000FFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFDFFFFFF0400000001000000000000000100000000000000060000000000000002000000070000000000000000000000FEFFFFFF0000000000000000FFFFFFFFFEFFFFFF01000000FEFFFFFF02000000000000000300000004000000FDFFFFFFFAFFFFFFFCFFFFFF01000000020000000400000002000000FCFFFFFF06000000FDFFFFFF010000000100000001000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFBFFFFFFFEFFFFFF00000000FFFFFFFFFFFFFFFF0300000000000000FFFFFFFF0200000000000000FEFFFFFF000000000000000000000000FDFFFFFF0100000000000000FFFFFFFF00000000FCFFFFFFFDFFFFFF0200000001000000FFFFFFFF0000000001000000FCFFFFFFFDFFFFFF020000000000000000000000020000000400000000000000000000000100000000000000FFFFFFFF0400000000000000FDFFFFFF03000000FDFFFFFF000000000600000001000000FFFFFFFFFEFFFFFF03000000FBFFFFFFFEFFFFFF00000000FEFFFFFF03000000FEFFFFFF0200000001000000020000000000000002000000000000000200000000000000FEFFFFFF00000000FEFFFFFFFDFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFF0400000004000000050000000200000002000000FFFFFFFF03000000020000000000000005000000020000000100000003000000FEFFFFFF02000000FFFFFFFF02000000FEFFFFFF010000000300000000000000FDFFFFFF03000000030000000200000000000000FBFFFFFF02000000FBFFFFFF0000000000000000FBFFFFFF03000000FEFFFFFF02000000FDFFFFFF0000000001000000FCFFFFFF00000000000000000000000001000000FFFFFFFF000000000200000000000000020000000000000000000000030000000000000004000000FEFFFFFF0600000000000000000000000000000009000000000000000100000000000000FDFFFFFF01000000FFFFFFFF0400000000000000FFFFFFFF0000000001000000FFFFFFFF06000000FFFFFFFF06000000020000000200000003000000020000000000000003000000010000000000000000000000000000000200000003000000FAFFFFFF020000000100000000000000FCFFFFFF00000000FEFFFFFF000000000000000004000000FFFFFFFF0700000000000000FEFFFFFF00000000FFFFFFFFFDFFFFFF0300000000000000FFFFFFFF00000000FDFFFFFF0200000000000000000000000100000000000000FDFFFFFFFFFFFFFFFFFFFFFF0300000000000000FFFFFFFF02000000FDFFFFFF0200000003000000000000000400000000000000FDFFFFFF00000000FEFFFFFF0200000002000000000000000100000002000000FBFFFFFF0600000001000000FCFFFFFF00000000FEFFFFFFFDFFFFFF040000000000000004000000FFFFFFFF00000000010000000100000000000000FDFFFFFF0400000000000000F9FFFFFFFDFFFFFF0000000002000000FFFFFFFF01000000FEFFFFFF01000000020000000000000003000000000000000000000001000000FFFFFFFFFFFFFFFF000000000000000005000000000000000200000004000000000000000300000004000000FCFFFFFF04000000010000000000000000000000FFFFFFFF00000000FFFFFFFF0000000000000000FEFFFFFF0200000000000000FEFFFFFF06000000FFFFFFFF01000000030000000300000001000000FDFFFFFFFEFFFFFF00000000000000000000000001000000FDFFFFFFFFFFFFFF0200000002000000FFFFFFFF02000000FDFFFFFF000000000000000000000000FFFFFFFF02000000FBFFFFFF04000000FEFFFFFFFFFFFFFF000000000000000002000000040000000400000002000000020000000000000002000000FDFFFFFFFFFFFFFF050000000500000001000000FFFFFFFF0000000000000000000000000000000003000000000000000200000000000000FCFFFFFFFDFFFFFF000000000400000000000000FFFFFFFF0200000002000000010000000300000007000000FCFFFFFF0100000000000000FEFFFFFFFBFFFFFF01000000020000000200000007000000"> : tensor<3x5x40xi32>
    return %0 : tensor<3x5x40xi32>
  }
}

