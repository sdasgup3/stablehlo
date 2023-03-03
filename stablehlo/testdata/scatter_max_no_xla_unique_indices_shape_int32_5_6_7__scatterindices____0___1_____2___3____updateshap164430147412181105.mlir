// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>)
    %2 = call @expected() : () -> tensor<5x6x7xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<2x2x1xi32>, tensor<5x2x2x7xi32>) -> tensor<5x6x7xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi32>, tensor<5x6x7xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>) {
    %0 = stablehlo.constant dense<"0xFDFFFFFF040000000400000000000000000000000000000001000000FEFFFFFFFEFFFFFFFBFFFFFF00000000FDFFFFFF01000000FFFFFFFFFEFFFFFF00000000FFFFFFFF010000000300000000000000FFFFFFFF01000000FDFFFFFF000000000000000000000000FDFFFFFF01000000FDFFFFFFFFFFFFFF02000000FDFFFFFF010000000000000000000000FFFFFFFF02000000000000000100000003000000030000000100000000000000000000000000000001000000FDFFFFFFFBFFFFFF0200000000000000020000000100000002000000FCFFFFFFFDFFFFFF02000000FEFFFFFFFFFFFFFFFDFFFFFFFEFFFFFF0000000000000000FDFFFFFF0000000000000000FAFFFFFF03000000FCFFFFFFFCFFFFFF00000000FCFFFFFFFCFFFFFF00000000010000000100000000000000FFFFFFFFFEFFFFFF03000000FFFFFFFF01000000020000000000000002000000FEFFFFFFFEFFFFFFFFFFFFFF01000000FDFFFFFF030000000100000002000000030000000000000000000000040000000000000001000000FFFFFFFFFDFFFFFF04000000F8FFFFFF00000000FFFFFFFFFFFFFFFF00000000FDFFFFFF0000000002000000FFFFFFFFF8FFFFFF00000000FBFFFFFFFEFFFFFFFBFFFFFF010000000000000000000000020000000000000000000000FFFFFFFF02000000F8FFFFFFFFFFFFFF0000000000000000F9FFFFFFFCFFFFFF0000000005000000FFFFFFFF00000000FFFFFFFF000000000300000002000000FFFFFFFFFEFFFFFF0000000000000000FEFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF060000000400000000000000000000000200000006000000000000000200000000000000FBFFFFFF0600000000000000FFFFFFFF02000000FEFFFFFF0400000001000000FFFFFFFF01000000FEFFFFFF0200000000000000020000000600000001000000000000000000000001000000010000000200000000000000050000000000000002000000000000000400000000000000FDFFFFFF02000000FAFFFFFF0400000000000000FFFFFFFF00000000040000000100000000000000FEFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF00000000FFFFFFFFFCFFFFFFFFFFFFFFFAFFFFFF0000000000000000FFFFFFFFFCFFFFFFFDFFFFFF04000000"> : tensor<5x6x7xi32>
    %1 = stablehlo.constant dense<"0x0100000000000000FEFFFFFFFEFFFFFF00000000FDFFFFFFFEFFFFFF020000000200000000000000020000000000000000000000FEFFFFFF03000000FCFFFFFFFDFFFFFFFAFFFFFF00000000FFFFFFFF00000000FFFFFFFFFFFFFFFF0500000005000000FBFFFFFF0000000000000000FFFFFFFFFDFFFFFF0100000000000000FBFFFFFF0300000000000000FFFFFFFF01000000FFFFFFFFFEFFFFFF050000000000000000000000FEFFFFFF02000000FFFFFFFF0000000002000000FFFFFFFF020000000100000002000000000000000600000000000000FFFFFFFF010000000000000002000000FBFFFFFF00000000FFFFFFFFFDFFFFFFFDFFFFFF00000000FEFFFFFFFFFFFFFFFEFFFFFF020000000300000000000000FCFFFFFFFDFFFFFFFEFFFFFF00000000FFFFFFFF01000000F9FFFFFFFEFFFFFFFFFFFFFF01000000FFFFFFFF00000000040000000200000000000000FFFFFFFFFDFFFFFF0300000001000000FDFFFFFFFFFFFFFFFEFFFFFF000000000100000000000000FFFFFFFFFDFFFFFFFEFFFFFF00000000020000000000000003000000020000000200000003000000010000000100000002000000FEFFFFFF0300000001000000020000000000000003000000040000000000000003000000F9FFFFFF0000000003000000020000000200000002000000FFFFFFFF020000000300000003000000FFFFFFFF0000000000000000FDFFFFFFFAFFFFFF0000000005000000FDFFFFFF05000000FFFFFFFF00000000FFFFFFFF00000000"> : tensor<5x2x2x7xi32>
    return %0, %1 : tensor<5x6x7xi32>, tensor<5x2x2x7xi32>
  }
  func.func private @expected() -> tensor<5x6x7xi32> {
    %0 = stablehlo.constant dense<"0x01000000040000000400000000000000000000000000000001000000020000000200000000000000020000000000000001000000FFFFFFFF0300000000000000FFFFFFFF0100000003000000000000000000000001000000FFFFFFFF0500000005000000000000000000000001000000FDFFFFFFFFFFFFFF02000000FDFFFFFF010000000000000000000000FFFFFFFF02000000000000000100000003000000030000000100000000000000000000000100000001000000FDFFFFFF030000000200000000000000020000000100000002000000050000000000000002000000FEFFFFFF02000000FFFFFFFF000000000200000000000000020000000100000002000000000000000600000000000000FFFFFFFF01000000FCFFFFFFFCFFFFFF00000000010000000100000000000000FFFFFFFFFEFFFFFF03000000FFFFFFFF010000000200000000000000020000000000000002000000FFFFFFFF01000000FFFFFFFF030000000100000002000000030000000000000000000000040000000300000001000000FFFFFFFFFDFFFFFF04000000000000000000000001000000FFFFFFFF00000000FFFFFFFF0100000002000000000000000400000002000000FBFFFFFFFEFFFFFFFBFFFFFF010000000000000000000000020000000000000000000000FFFFFFFF02000000F8FFFFFFFFFFFFFF0000000000000000FFFFFFFFFDFFFFFF0300000005000000FFFFFFFF00000000FFFFFFFF000000000300000002000000FFFFFFFFFEFFFFFF000000000000000002000000000000000300000002000000060000000400000001000000010000000200000006000000030000000200000002000000FBFFFFFF0600000000000000FFFFFFFF02000000FEFFFFFF0400000001000000FFFFFFFF01000000FEFFFFFF02000000000000000200000006000000030000000400000000000000030000000100000002000000030000000500000002000000020000000000000004000000030000000300000002000000000000000400000000000000FFFFFFFF00000000050000000100000005000000FFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF00000000FFFFFFFFFCFFFFFFFFFFFFFFFAFFFFFF0000000000000000FFFFFFFFFCFFFFFFFDFFFFFF04000000"> : tensor<5x6x7xi32>
    return %0 : tensor<5x6x7xi32>
  }
}

