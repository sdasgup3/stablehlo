// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      stablehlo.return %arg1 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi32>, tensor<2xi32>, tensor<4x3xi32>) -> tensor<4x2x3x5xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi32>, tensor<4x2x3x5xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi32>, tensor<4x3xi32>) {
    %0 = stablehlo.constant dense<"0x00000000FCFFFFFFFDFFFFFFFCFFFFFF000000000000000004000000FFFFFFFF02000000FDFFFFFFFCFFFFFF04000000FAFFFFFF00000000FDFFFFFF00000000FFFFFFFF0300000002000000050000000000000000000000000000000500000000000000FBFFFFFF01000000010000000100000003000000FDFFFFFFFDFFFFFFF9FFFFFFFCFFFFFFFFFFFFFF0000000002000000FFFFFFFFFEFFFFFF020000000100000005000000FFFFFFFF00000000FFFFFFFF0200000000000000FBFFFFFFFDFFFFFF02000000FAFFFFFF04000000FCFFFFFF0400000002000000050000000100000001000000FDFFFFFF00000000FFFFFFFF0200000004000000FFFFFFFFFEFFFFFFFEFFFFFF01000000FEFFFFFF09000000FFFFFFFFFFFFFFFFFFFFFFFF06000000FFFFFFFFFEFFFFFFFDFFFFFF0200000001000000FEFFFFFFFCFFFFFF02000000FFFFFFFF02000000FEFFFFFF05000000FFFFFFFF0000000001000000000000000400000000000000FFFFFFFF05000000FDFFFFFF0000000000000000FBFFFFFF00000000040000000100000000000000FCFFFFFF000000000100000000000000000000000000000000000000FDFFFFFFFFFFFFFF02000000FFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000"> : tensor<4x2x3x5xi32>
    %1 = stablehlo.constant dense<[[0, -1, -3], [2, -2, 3], [2, 0, 1], [8, 0, -1]]> : tensor<4x3xi32>
    return %0, %1 : tensor<4x2x3x5xi32>, tensor<4x3xi32>
  }
  func.func private @expected() -> tensor<4x2x3x5xi32> {
    %0 = stablehlo.constant dense<"0x00000000FCFFFFFFFDFFFFFFFCFFFFFF000000000000000004000000FFFFFFFF02000000FFFFFFFFFCFFFFFF04000000FAFFFFFF00000000FDFFFFFF00000000FFFFFFFF0300000002000000050000000000000000000000000000000500000000000000FBFFFFFF01000000010000000100000003000000FDFFFFFFFDFFFFFFF9FFFFFFFCFFFFFF020000000000000002000000FFFFFFFFFEFFFFFFFEFFFFFF0100000005000000FFFFFFFF00000000030000000200000000000000FBFFFFFFFDFFFFFF02000000FAFFFFFF04000000FCFFFFFF0400000002000000050000000100000001000000FDFFFFFF00000000FFFFFFFF0200000004000000FFFFFFFF02000000FEFFFFFF01000000FEFFFFFF0900000000000000FFFFFFFFFFFFFFFF06000000FFFFFFFF01000000FDFFFFFF0200000001000000FEFFFFFFFCFFFFFF02000000FFFFFFFF02000000FEFFFFFF05000000FFFFFFFF0000000001000000000000000400000000000000FFFFFFFF05000000FDFFFFFF0800000000000000FBFFFFFF00000000040000000000000000000000FCFFFFFF0000000001000000FFFFFFFF000000000000000000000000FDFFFFFFFFFFFFFF02000000FFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000"> : tensor<4x2x3x5xi32>
    return %0 : tensor<4x2x3x5xi32>
  }
}

