// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xi16>
    %1 = call @expected() : () -> tensor<20x20xi16>
    %2 = stablehlo.abs %0 : tensor<20x20xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xi16>, tensor<20x20xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xi16> {
    %0 = stablehlo.constant dense<"0x0400020000000200FFFFFEFF0000FFFF0100FFFF050001000000FDFFFFFF0000FCFF0200030000000100FDFF00000000FBFF00000000FDFF0000FEFF010003000000030000000000FAFF0200FFFF05000400FEFF04000400FCFFFBFF02000300FDFF000000000100FDFFFDFF000003000300FFFFFDFF0000FFFF0000FCFFFFFF010002000200020003000400030003000200FEFFFFFFFFFFFAFF00000100FDFF000001000500040005000200FFFF00000200000004000000FCFF020001000000FDFF0000020001000100FCFF0000FDFF0300000001000300FFFFFCFF0000FFFFFFFF000001000000FEFF0000FEFF0000FEFF00000200010000000500050005000100FFFF00000000010004000000FAFFFDFFFBFFFEFF0000FCFF0200000001000000FCFF00000000FFFF00000000FEFFFBFFFDFFFFFF0000FFFF000002000000030000000000FFFF01000000FCFFFCFF0900FEFFFEFF030000000000FDFFFFFF03000200FDFF00000200FEFF03000000FCFFFCFF0100FAFF00000000FEFFFDFF0200FDFF0000FEFF0300010001000000010001000000010005000100020005000000000005000100050005000500FFFFFFFF03000100FFFFFBFF0600FDFF0000FEFF0000FEFF0000000000000000FBFFFEFFFFFF000001000200FFFF0300FEFFFCFFFFFF000000000100000000000200010000000100FBFF0300FFFF010002000000000000000000030006000000FEFF000000000200FBFF01000000FEFFFFFFFEFF0400FFFF0100FFFF03000000FFFF02000300FFFF0200F9FFFCFFFCFF010000000500FDFF02000300FCFF010000000900F9FFFFFFFDFFFFFF010000000000FEFF0100010000000000000002000000FDFFFFFF00000100FFFFFDFFFDFF00000000FBFF0100FDFF00000400FFFF04000000FDFFFDFF000004000100030000000300FEFF01000100FFFF040000000000FCFFFCFF01000000010000000000FDFF04000000FCFF0000FDFF06000000000007000200FEFFFBFF0400010002000300FEFFFFFF0400FFFFFCFF02000200FEFF000001000100FFFFFEFF010000000200000001000100FDFF0200FCFF04000200FBFFFDFF000002000100FFFFFFFFFFFF"> : tensor<20x20xi16>
    return %0 : tensor<20x20xi16>
  }
  func.func private @expected() -> tensor<20x20xi16> {
    %0 = stablehlo.constant dense<"0x0400020000000200010002000000010001000100050001000000030001000000040002000300000001000300000000000500000000000300000002000100030000000300000000000600020001000500040002000400040004000500020003000300000000000100030003000000030003000100030000000100000004000100010002000200020003000400030003000200020001000100060000000100030000000100050004000500020001000000020000000400000004000200010000000300000002000100010004000000030003000000010003000100040000000100010000000100000002000000020000000200000002000100000005000500050001000100000000000100040000000600030005000200000004000200000001000000040000000000010000000000020005000300010000000100000002000000030000000000010001000000040004000900020002000300000000000300010003000200030000000200020003000000040004000100060000000000020003000200030000000200030001000100000001000100000001000500010002000500000000000500010005000500050001000100030001000100050006000300000002000000020000000000000000000500020001000000010002000100030002000400010000000000010000000000020001000000010005000300010001000200000000000000000003000600000002000000000002000500010000000200010002000400010001000100030000000100020003000100020007000400040001000000050003000200030004000100000009000700010003000100010000000000020001000100000000000000020000000300010000000100010003000300000000000500010003000000040001000400000003000300000004000100030000000300020001000100010004000000000004000400010000000100000000000300040000000400000003000600000000000700020002000500040001000200030002000100040001000400020002000200000001000100010002000100000002000000010001000300020004000400020005000300000002000100010001000100"> : tensor<20x20xi16>
    return %0 : tensor<20x20xi16>
  }
}
