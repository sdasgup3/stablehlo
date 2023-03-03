// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x20xi32>, tensor<20x20xi32>)
    %1 = call @expected() : () -> tensor<20x20xi32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1] : (tensor<1x20xi32>) -> tensor<20x20xi32>
    %3 = stablehlo.xor %2, %0#1 : tensor<20x20xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<20x20xi32>, tensor<20x20xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x20xi32>, tensor<20x20xi32>) {
    %0 = stablehlo.constant dense<[[-2, 0, -4, 0, -2, 1, 0, -1, 0, 0, -1, 2, -5, 3, -4, -4, 3, -2, 2, 3]]> : tensor<1x20xi32>
    %1 = stablehlo.constant dense<"0x03000000FCFFFFFF00000000FFFFFFFF0000000002000000030000000000000000000000FEFFFFFFFFFFFFFFFDFFFFFF01000000020000000000000000000000FCFFFFFF03000000020000000200000001000000FCFFFFFF04000000000000000000000001000000FDFFFFFF01000000020000000800000002000000010000000100000001000000050000000200000002000000FEFFFFFFFDFFFFFF01000000FEFFFFFFFEFFFFFF000000000000000002000000FEFFFFFF0200000001000000FEFFFFFF0200000000000000FFFFFFFFFFFFFFFF00000000FBFFFFFF00000000FFFFFFFFFEFFFFFFFAFFFFFFFDFFFFFFFAFFFFFF0000000003000000020000000200000000000000FEFFFFFF01000000FFFFFFFF05000000000000000100000000000000FDFFFFFF02000000070000000100000000000000FBFFFFFF040000000000000000000000FFFFFFFF0400000003000000FFFFFFFF0000000000000000FAFFFFFF00000000FFFFFFFFFDFFFFFFF9FFFFFF04000000000000000000000001000000FDFFFFFFFDFFFFFF00000000FBFFFFFFFFFFFFFF040000000000000004000000FDFFFFFF01000000050000000000000001000000FFFFFFFFFCFFFFFF070000000000000000000000FFFFFFFF00000000FEFFFFFF02000000FEFFFFFF00000000FCFFFFFF00000000FEFFFFFF00000000FFFFFFFFFEFFFFFF0000000004000000000000000100000002000000020000000200000002000000FDFFFFFF020000000000000001000000000000000100000003000000010000000000000000000000FEFFFFFFFFFFFFFFFCFFFFFF0300000001000000FEFFFFFF0000000000000000FBFFFFFFFEFFFFFF02000000FEFFFFFF00000000FFFFFFFF0000000001000000FDFFFFFFFDFFFFFFFAFFFFFFFEFFFFFF000000000500000000000000030000000000000000000000FDFFFFFF0000000001000000010000000000000004000000FDFFFFFFFAFFFFFFFEFFFFFF0000000001000000FDFFFFFF000000000200000002000000FBFFFFFFFEFFFFFF0300000002000000FEFFFFFF01000000FBFFFFFF030000000000000002000000010000000100000000000000FFFFFFFFFFFFFFFFFCFFFFFF0000000000000000FFFFFFFF0000000000000000FFFFFFFF01000000FEFFFFFF000000000000000007000000000000000600000000000000000000000300000000000000FDFFFFFF05000000010000000400000003000000000000000100000000000000FFFFFFFF02000000050000000000000000000000FEFFFFFFFBFFFFFF00000000FDFFFFFF01000000FEFFFFFF01000000FEFFFFFF020000000000000002000000FFFFFFFF00000000030000000000000000000000FEFFFFFF04000000FAFFFFFF0200000000000000FBFFFFFF03000000FDFFFFFFFAFFFFFFFCFFFFFF000000000000000002000000000000000000000001000000FEFFFFFF03000000FDFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FFFFFFFFFDFFFFFFFEFFFFFFFDFFFFFFFCFFFFFFFAFFFFFFFDFFFFFFFEFFFFFF04000000FFFFFFFFFEFFFFFF0400000000000000000000000000000000000000FBFFFFFFFCFFFFFF0000000002000000FEFFFFFF00000000010000000000000000000000FCFFFFFFFCFFFFFF020000000400000000000000FEFFFFFF0100000000000000FEFFFFFFF9FFFFFF03000000FFFFFFFF0500000002000000FFFFFFFF01000000FDFFFFFF04000000000000000000000000000000000000000000000000000000FDFFFFFF03000000FDFFFFFFFFFFFFFF030000000000000003000000FFFFFFFF00000000FDFFFFFF00000000FEFFFFFF00000000FFFFFFFF04000000FFFFFFFF01000000030000000000000000000000FCFFFFFFFEFFFFFF04000000FFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFFFEFFFFFFFDFFFFFF0000000005000000000000000000000007000000FFFFFFFF00000000FFFFFFFF00000000FCFFFFFF040000000000000003000000010000000100000000000000FFFFFFFF0000000002000000FEFFFFFF00000000050000000100000002000000FFFFFFFF000000000500000002000000000000000000000001000000FDFFFFFFFAFFFFFF03000000FFFFFFFF01000000010000000000000002000000FDFFFFFF0200000001000000FCFFFFFF00000000"> : tensor<20x20xi32>
    return %0, %1 : tensor<1x20xi32>, tensor<20x20xi32>
  }
  func.func private @expected() -> tensor<20x20xi32> {
    %0 = stablehlo.constant dense<"0xFDFFFFFFFCFFFFFFFCFFFFFFFFFFFFFFFEFFFFFF0300000003000000FFFFFFFF00000000FEFFFFFF00000000FFFFFFFFFAFFFFFF01000000FCFFFFFFFCFFFFFFFFFFFFFFFDFFFFFF0000000001000000FFFFFFFFFCFFFFFFF8FFFFFF00000000FEFFFFFF00000000FDFFFFFFFEFFFFFF0200000008000000FDFFFFFF03000000FAFFFFFF02000000F9FFFFFFFEFFFFFF0100000000000000FFFFFFFF0200000000000000FEFFFFFFFCFFFFFF00000000FCFFFFFFFFFFFFFF02000000FEFFFFFFFEFFFFFF02000000FFFFFFFFFDFFFFFF040000000300000007000000FCFFFFFFFCFFFFFF00000000F8FFFFFFFEFFFFFF0400000000000000FFFFFFFF02000000FCFFFFFF01000000FEFFFFFFFEFFFFFFFFFFFFFF05000000FFFFFFFF03000000FBFFFFFFFEFFFFFFFEFFFFFFFBFFFFFF02000000FEFFFFFFF9FFFFFF07000000FEFFFFFF000000000300000004000000FDFFFFFFFEFFFFFF00000000FFFFFFFFFAFFFFFF0000000000000000FFFFFFFF0200000007000000FCFFFFFFFCFFFFFF0200000003000000FFFFFFFF0300000005000000FFFFFFFFF8FFFFFF00000000FAFFFFFFFCFFFFFF01000000FAFFFFFF000000000100000000000000FEFFFFFFFCFFFFFF03000000FCFFFFFF03000000030000000000000000000000FDFFFFFFFEFFFFFFFCFFFFFFFCFFFFFFFEFFFFFFFEFFFFFFFEFFFFFFFEFFFFFFFFFFFFFF0400000000000000FEFFFFFF00000000F9FFFFFF01000000FEFFFFFF0100000001000000FEFFFFFF0300000003000000FFFFFFFF03000000FDFFFFFF00000000FEFFFFFFFFFFFFFFFFFFFFFF0300000003000000010000000100000002000000FBFFFFFFF8FFFFFF02000000FEFFFFFFFDFFFFFFFEFFFFFFFDFFFFFF03000000FFFFFFFFFDFFFFFF01000000FAFFFFFF000000000100000005000000FFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFBFFFFFF02000000FDFFFFFFFCFFFFFF0700000003000000F8FFFFFFFDFFFFFFFEFFFFFF010000000100000000000000FCFFFFFF03000000FBFFFFFF01000000030000000200000001000000030000000000000000000000FCFFFFFFFEFFFFFF02000000FFFFFFFF02000000FCFFFFFF01000000FCFFFFFFFCFFFFFF000000000100000001000000000000000000000001000000FEFFFFFFFFFFFFFF02000000FCFFFFFF03000000FAFFFFFFFCFFFFFF03000000FDFFFFFF02000000FEFFFFFFFBFFFFFF01000000F8FFFFFF03000000FEFFFFFF0000000000000000000000000200000005000000FFFFFFFF0200000005000000F8FFFFFFFCFFFFFF01000000020000000000000003000000FDFFFFFFFCFFFFFF00000000FEFFFFFFFFFFFFFFFEFFFFFF0200000000000000FFFFFFFFFEFFFFFF040000000500000000000000FBFFFFFFF8FFFFFFFFFFFFFF01000000F9FFFFFF020000000200000003000000FCFFFFFF00000000FCFFFFFF010000000000000002000000FDFFFFFF03000000FFFFFFFFFEFFFFFF00000000FCFFFFFF0400000002000000FCFFFFFF03000000FEFFFFFF00000000FFFFFFFFFFFFFFFF04000000FDFFFFFF020000000400000001000000FFFFFFFF04000000FFFFFFFF0000000000000000FFFFFFFFF9FFFFFF0700000003000000FEFFFFFF0200000003000000FFFFFFFF020000000300000002000000FCFFFFFFFEFFFFFF04000000FEFFFFFFFFFFFFFF01000000FFFFFFFFFEFFFFFFF9FFFFFFFCFFFFFFFDFFFFFFFEFFFFFF0100000003000000FDFFFFFFFEFFFFFFFAFFFFFF0200000003000000FEFFFFFF00000000FCFFFFFF000000000300000002000000FDFFFFFF000000000300000000000000FCFFFFFFFDFFFFFFFBFFFFFFFEFFFFFFFCFFFFFF02000000030000000100000006000000FCFFFFFFFFFFFFFF03000000FCFFFFFF0000000002000000FFFFFFFF04000000000000000000000000000000FEFFFFFF0200000005000000FCFFFFFF020000000100000003000000FBFFFFFF0200000003000000F9FFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFDFFFFFF04000000FFFFFFFF0300000001000000FEFFFFFF020000000400000003000000FEFFFFFF0200000003000000FBFFFFFF03000000010000000100000000000000F9FFFFFF02000000FEFFFFFF010000000100000002000000FAFFFFFF030000000000000003000000FAFFFFFF03000000FEFFFFFF0100000001000000FFFFFFFFFEFFFFFF03000000"> : tensor<20x20xi32>
    return %0 : tensor<20x20xi32>
  }
}
