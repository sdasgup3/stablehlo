// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %2 = call @expected() : () -> tensor<1x50x3xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xui16>, tensor<1xi32>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xui16>, tensor<1x3xui16>) {
    %0 = stablehlo.constant dense<"0x000001000000020002000300010002000200010000000300010001000200040002000100060002000300000002000400080002000400000002000500010004000200000005000000040001000100030005000300020004000000000004000100000000000100030000000000040002000400020000000100010005000000020000000800010000000000010004000600020003000100000001000100040005000300010001000100010003000400010000000400000000000300040000000300000003000000050004000300040005000500000000000500020002000200020004000200030000000000010000000100030007000000010004000400030003000300000001000400010005000600030001000200010002000300030003000100020001000100020001000200"> : tensor<1x50x3xui16>
    %1 = stablehlo.constant dense<[[4, 1, 1]]> : tensor<1x3xui16>
    return %0, %1 : tensor<1x50x3xui16>, tensor<1x3xui16>
  }
  func.func private @expected() -> tensor<1x50x3xui16> {
    %0 = stablehlo.constant dense<"0x000001000000020002000300010002000200010000000300010001000200040002000100060002000300000002000400080002000400000002000500010004000200000005000000040001000100030005000300020004000000000004000100000000000100030000000000040002000400020000000100010005000000020000000800010000000000010004000600020003000100000001000100040005000300010001000100010003000400010000000400000000000300040000000300000001000000050004000300040005000500000000000500020002000200020004000200030000000000010000000100030007000000010004000400030003000300000001000400010005000600030001000200010002000300030003000100020001000100020001000200"> : tensor<1x50x3xui16>
    return %0 : tensor<1x50x3xui16>
  }
}

