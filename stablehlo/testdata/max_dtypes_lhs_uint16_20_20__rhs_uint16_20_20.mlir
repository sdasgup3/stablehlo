// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xui16>, tensor<20x20xui16>)
    %1 = call @expected() : () -> tensor<20x20xui16>
    %2 = stablehlo.maximum %0#0, %0#1 : tensor<20x20xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xui16>, tensor<20x20xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xui16>, tensor<20x20xui16>) {
    %0 = stablehlo.constant dense<"0x0000040004000400050000000300020001000100010002000200040003000300010000000200020001000200000003000200010001000000000002000500010006000000020003000400020000000600000001000200050001000200000000000300030002000200000003000100020000000200030000000500050000000100030005000300020003000400010002000000030002000100020004000000020004000100000001000100000003000300000006000000060000000500020002000000030001000200000001000100030001000000000004000300000001000400010001000200010003000000020002000100020002000100020001000100010001000500020000000400010002000200010002000400040002000200020002000100030002000500000001000000000000000700000007000000070003000400000003000100000001000200000002000100030000000100010005000100030001000800020001000400000000000000000004000100060006000300020003000000010002000400010007000100000001000200030000000000050000000300020001000000030004000000060002000000000001000200000000000000010002000100030002000000090001000300000001000100020000000200020004000000020000000500010005000100020000000000030000000100010004000500040001000000020004000100010000000200000003000000050006000400040001000500000001000400000006000100030002000100000001000100020003000400040002000500010002000000030002000200010002000000000001000200030000000100010002000400010000000400000000000100010002000600030000000100020004000400000004000400020000000000030004000300020001000200010002000200010002000200050005000700060003000200020002000000030005000600000000000400000003000200040001000000000002000000030006000300040000000200010003000300000000000100010006000800000000000300020000000200000006000400040005000000040002000300040004000000"> : tensor<20x20xui16>
    %1 = stablehlo.constant dense<"0x0300000001000000010000000300020000000300000002000100000002000400020001000400020000000400020000000400020000000100000000000200010000000300040001000300010000000200010000000200030000000600030002000100060002000300030000000000010002000400020005000000000000000400000003000000010002000000000000000000000000000000000000000400020002000500010000000400000005000000020000000300000001000200050000000200030004000000000000000200000004000300020002000100000002000400000008000400020001000100020001000100010002000300000000000200030003000300010001000100030002000000010002000600000007000000020004000100000005000000010000000100070001000500010000000000030001000200030002000600010006000000050001000000000000000400010004000100060001000300010001000100010000000000020002000400000000000200030001000300020006000600000000000500000001000200010003000300010002000400000001000200010004000100030001000200040002000200010003000000050002000000040003000000000000000200010002000400040006000200020000000000010000000000030000000000020000000300030005000100000002000300000001000100020002000200040001000000050001000200000001000300020002000300030000000400030000000000000005000100020005000000000000000200010000000000020003000000050001000000000000000000010000000200010000000300010000000000000002000000040001000000010000000000000002000500000001000000000005000000020003000000020001000000020003000100010001000000020006000200010000000400020001000300010000000200050005000400000000000100000000000100070004000400020002000000020005000800000000000200030001000700000000000200000000000300010000000000000002000200000002000100010002000200030005000300050002000500"> : tensor<20x20xui16>
    return %0, %1 : tensor<20x20xui16>, tensor<20x20xui16>
  }
  func.func private @expected() -> tensor<20x20xui16> {
    %0 = stablehlo.constant dense<"0x0300040004000400050000000300020001000300010002000200040003000400020001000400020001000400020003000400020001000100000002000500010006000300040003000400020000000600010001000200050001000600030002000300060002000300030003000100020002000400030005000500050000000400030005000300020003000400010002000000030002000100020004000400020004000500010001000400000005000300020006000300060001000500050002000200030004000200000001000200030004000300020004000300000002000400010008000400020003000100020002000100020002000300020001000200030003000500020001000400030002000200010002000600040007000200020004000100030005000500010001000100070001000700010007000000070003000400030003000600010006000200050002000100030000000400010005000100060001000800020001000400010000000000020004000400060006000300030003000300020006000600010007000500000001000200030003000300050002000400020001000200030004000100060002000200040002000200010003000000050002000100040003000000090001000300010002000400040006000200020004000000020000000500030005000100020000000300030005000100010004000500040001000100020004000200040001000200050003000200050006000400040002000500030001000400030006000100030005000100020005000100020003000400040002000500020003000000050002000200010002000000010001000200030000000300010002000400010002000400040001000100010002000600030002000500020004000400000005000400020003000000030004000300020003000200010002000200020006000200050005000700060003000300020002000200050005000600000000000400000003000200070004000400020002000000030006000800040000000200030003000700000000000200010006000800010000000300020002000200000006000400040005000200040005000300050004000500"> : tensor<20x20xui16>
    return %0 : tensor<20x20xui16>
  }
}
