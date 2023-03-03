// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui16>, tensor<5x2x2x7xui16>)
    %2 = call @expected() : () -> tensor<5x6x7xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui16>, tensor<2x2x1xi32>, tensor<5x2x2x7xui16>) -> tensor<5x6x7xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui16>, tensor<5x2x2x7xui16>) {
    %0 = stablehlo.constant dense<"0x010003000200020000000100020002000000040002000000010002000300020000000100000001000300010000000000030005000100000005000200010001000200040002000200000001000500000002000100000002000100020000000000040003000100010000000300010005000200020003000100000001000100030002000400030000000000020001000100030004000300000002000400020000000100000003000000030000000200000001000500070001000100020003000000010002000000010004000000020002000200010002000100000002000100000002000500030002000000020000000000020002000200040002000000020000000000070005000000000003000000010005000100000002000300040001000400020001000400000002000000010006000400020001000300010001000000000003000300010003000100020005000000010003000200030001000000000003000300010001000000000000000400020003000200040001000100030002000400010001000000010004000400030001000100030000000000000000000300010003000200"> : tensor<5x6x7xui16>
    %1 = stablehlo.constant dense<"0x02000000000001000100000002000200020002000200030002000200000000000400020001000200040003000100040000000400010000000100030001000100000003000100030001000400000003000000010002000100000000000100020006000000020002000000010002000200020000000000000003000100000002000500040001000000020000000100040000000600000001000300020002000300010001000200030003000100040002000100010000000000050000000000070001000500000003000600010006000100030003000400020003000000030000000800040000000200000001000000000003000000010000000000050002000000000004000100020002000100020001000100000004000000"> : tensor<5x2x2x7xui16>
    return %0, %1 : tensor<5x6x7xui16>, tensor<5x2x2x7xui16>
  }
  func.func private @expected() -> tensor<5x6x7xui16> {
    %0 = stablehlo.constant dense<"0x030003000200030001000100040004000200060004000300030004000300020004000300010003000700040001000400030009000200000005000200010001000200040002000200000001000500000002000100010005000200030000000300050006000200050000000600010006000400030003000100010003000700030004000600030001000200040001000100030004000300000002000400020000000100000003000000050000000200000004000600070003000600060004000000030002000100050004000600020003000500030004000400010003000300030002000500030002000000020000000000020002000200040002000000050001000400090006000100000003000500010005000800010007000300070007000500080002000700030006000200040006000700020001000300010001000000000003000300010003000100020005000000090007000200050001000100000003000600010002000000000005000600020003000600050003000300040004000500020001000400010004000400030001000100030000000000000000000300010003000200"> : tensor<5x6x7xui16>
    return %0 : tensor<5x6x7xui16>
  }
}

