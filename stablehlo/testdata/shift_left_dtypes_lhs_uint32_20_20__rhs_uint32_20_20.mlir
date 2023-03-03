// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xui32>, tensor<20x20xui32>)
    %1 = call @expected() : () -> tensor<20x20xui32>
    %2 = stablehlo.shift_left %0#0, %0#1 : tensor<20x20xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xui32>, tensor<20x20xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xui32>, tensor<20x20xui32>) {
    %0 = stablehlo.constant dense<"0x01000000040000000300000001000000020000000000000003000000050000000100000003000000030000000600000005000000030000000400000004000000000000000200000003000000020000000500000001000000000000000400000002000000010000000200000001000000010000000200000002000000000000000100000002000000060000000200000001000000000000000400000001000000000000000100000002000000000000000400000001000000010000000300000003000000010000000000000001000000030000000000000000000000010000000400000003000000010000000200000001000000000000000300000001000000030000000000000002000000000000000300000001000000000000000500000002000000020000000200000006000000000000000600000002000000020000000500000002000000050000000200000006000000050000000300000003000000040000000100000002000000010000000000000000000000020000000400000000000000050000000200000002000000000000000400000001000000020000000000000001000000050000000100000000000000000000000000000004000000020000000000000001000000030000000500000002000000040000000600000003000000000000000000000000000000010000000300000002000000020000000500000003000000060000000300000004000000040000000200000001000000020000000200000001000000040000000600000001000000040000000000000001000000030000000100000003000000000000000300000002000000000000000100000000000000030000000400000002000000000000000000000002000000060000000200000002000000040000000200000003000000030000000300000002000000010000000000000000000000000000000100000000000000020000000300000000000000070000000100000003000000000000000400000002000000010000000100000002000000010000000200000002000000000000000100000000000000010000000200000001000000020000000000000001000000010000000100000007000000000000000200000000000000000000000100000000000000010000000200000000000000000000000100000002000000010000000200000002000000030000000000000006000000000000000100000002000000000000000000000000000000020000000200000005000000020000000100000008000000010000000200000000000000000000000100000006000000060000000100000003000000000000000200000006000000020000000000000005000000030000000100000006000000000000000200000004000000000000000100000003000000020000000100000005000000020000000400000000000000000000000400000001000000000000000100000000000000010000000300000003000000000000000500000009000000010000000500000000000000000000000000000004000000010000000100000001000000040000000200000004000000000000000000000002000000020000000900000001000000010000000200000002000000010000000200000000000000030000000400000002000000010000000300000002000000030000000500000003000000010000000000000002000000020000000100000003000000010000000100000003000000010000000300000000000000010000000000000001000000020000000200000001000000010000000300000001000000010000000200000001000000040000000200000004000000060000000000000000000000000000000000000003000000010000000000000000000000000000000300000001000000020000000000000001000000020000000300000001000000000000000200000004000000020000000400000000000000000000000100000001000000010000000200000002000000040000000400000002000000030000000200000001000000000000000000000004000000070000000600000008000000010000000100000003000000010000000400000000000000030000000000000000000000000000000000000005000000000000000100000002000000020000000200000000000000020000000300000002000000010000000300000000000000"> : tensor<20x20xui32>
    %1 = stablehlo.constant dense<"0x00000000010000000300000003000000010000000300000005000000020000000000000000000000030000000000000005000000040000000000000001000000030000000200000003000000050000000300000003000000020000000300000000000000030000000200000000000000020000000200000002000000020000000100000007000000010000000500000002000000030000000500000000000000000000000200000000000000000000000100000001000000000000000000000001000000030000000100000000000000020000000100000000000000000000000300000004000000050000000500000001000000020000000500000002000000000000000400000000000000010000000200000000000000020000000200000000000000020000000000000005000000000000000000000001000000000000000300000001000000060000000000000000000000010000000000000000000000030000000000000002000000020000000100000002000000040000000000000003000000030000000000000004000000060000000300000002000000010000000100000002000000060000000200000002000000010000000000000002000000020000000800000002000000010000000000000000000000000000000300000001000000010000000100000002000000000000000300000000000000010000000000000000000000010000000100000001000000000000000100000000000000000000000500000002000000000000000300000001000000050000000000000002000000020000000100000003000000000000000000000005000000000000000000000002000000010000000100000003000000010000000500000001000000020000000000000002000000010000000000000001000000000000000400000004000000010000000100000002000000000000000000000004000000020000000400000002000000010000000100000000000000000000000000000004000000000000000400000001000000000000000900000001000000010000000000000002000000010000000000000003000000010000000200000000000000020000000100000003000000010000000000000004000000030000000200000002000000030000000100000000000000030000000000000002000000010000000100000000000000000000000300000000000000020000000200000001000000030000000400000002000000030000000000000004000000060000000200000003000000010000000000000000000000010000000700000000000000010000000200000004000000030000000000000005000000030000000200000001000000000000000000000003000000050000000000000004000000020000000200000000000000020000000200000000000000020000000200000000000000060000000000000000000000010000000000000002000000040000000300000001000000020000000500000000000000030000000000000007000000040000000200000001000000010000000000000005000000020000000300000007000000020000000100000000000000010000000300000003000000000000000000000001000000040000000300000002000000000000000200000001000000040000000100000001000000010000000100000000000000000000000200000006000000010000000200000000000000000000000000000000000000020000000300000002000000000000000100000003000000000000000000000002000000020000000600000002000000000000000300000000000000000000000000000000000000040000000200000007000000020000000300000001000000030000000500000001000000030000000400000005000000020000000500000004000000000000000100000000000000020000000100000004000000010000000400000000000000000000000100000000000000070000000200000001000000010000000100000000000000030000000100000000000000010000000200000001000000030000000000000000000000000000000500000000000000000000000000000002000000000000000300000007000000000000000000000000000000020000000100000000000000040000000200000002000000030000000300000002000000000000000400000002000000"> : tensor<20x20xui32>
    return %0, %1 : tensor<20x20xui32>, tensor<20x20xui32>
  }
  func.func private @expected() -> tensor<20x20xui32> {
    %0 = stablehlo.constant dense<"0x010000000800000018000000080000000400000000000000600000001400000001000000030000001800000006000000A00000003000000004000000080000000000000008000000180000004000000028000000080000000000000020000000020000000800000008000000010000000400000008000000080000000000000002000000000100000C00000040000000040000000000000080000000010000000000000004000000020000000000000008000000020000000100000003000000060000000800000000000000010000000C0000000000000000000000010000002000000030000000200000004000000002000000000000006000000004000000030000000000000002000000000000000C000000010000000000000014000000020000000800000002000000C00000000000000006000000040000000200000028000000040000004001000002000000060000000A00000003000000030000002000000001000000080000000400000000000000000000002000000004000000000000002800000002000000200000000000000020000000040000000400000000000000040000004001000004000000000000000000000000000000100000000800000000000000040000000600000005000000020000000400000030000000060000000000000000000000000000000100000018000000020000000400000005000000030000000C00000006000000080000000400000004000000010000000200000040000000040000000400000030000000020000008000000000000000040000000C00000002000000180000000000000003000000400000000000000001000000000000000600000008000000100000000000000000000000040000001800000002000000080000000800000002000000060000000300000030000000200000000200000000000000000000000000000001000000000000000800000030000000000000000E00000002000000030000000000000004000000200000000100000010000000040000000100000000040000040000000000000001000000000000000200000002000000080000000400000000000000010000000400000002000000380000000000000002000000000000000000000004000000000000000800000004000000000000000000000001000000080000000200000004000000020000000300000000000000060000000000000004000000040000000000000000000000000000001000000002000000500000008000000004000000400000000200000002000000000000000000000080000000060000000C00000004000000300000000000000002000000C000000010000000000000000A00000003000000010000003000000000000000020000004000000000000000040000000300000008000000040000000500000008000000100000000000000000000000040000000100000000000000010000000000000010000000180000000600000000000000A000000009000000080000000500000000000000000000000000000008000000020000000100000020000000100000001000000000020000000000000000000002000000040000004800000008000000010000000200000004000000100000001000000000000000030000001000000004000000100000000600000004000000060000000A0000000300000001000000000000008000000004000000040000000300000001000000010000000300000004000000180000000000000001000000000000000800000002000000020000000400000004000000C000000004000000010000001000000001000000040000000200000004000000600000000000000000000000000000000000000006000000080000000000000000000000000000003000000020000000080000000000000010000000020000000600000001000000000000000400000040000000040000004000000000000000000000000200000001000000800000000800000004000000080000000800000002000000180000000400000001000000000000000000000008000000380000000600000008000000010000002000000003000000010000000400000000000000030000000000000000000000000000000000000005000000000000000200000002000000200000000800000000000000100000001800000008000000010000003000000000000000"> : tensor<20x20xui32>
    return %0 : tensor<20x20xui32>
  }
}
