// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %2 = call @expected() : () -> tensor<5x6x7xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x1xi32>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) {
    %0 = stablehlo.constant dense<"0x050202040301010106010601010502010403010300000201000302030200030101030201000000030004000100050001030008030301000204010005000100030600030102030204050303040004050003040100010000060201030000030101020102020203000804010102010100050103010200020301010300000901030100030004000201030002000101040303000400000102040401030301070200040203040104010004000309030400000401040207000001020000040003050201020100000202000101030004040002030600"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0504030602000102000000070107020307010000020101020402030601010103020200020103000200000200040100010001020102000103040002020100010001010000000504000002060106010000030002010302020101010202000604020104010004000001010800010002030002000202040400010000000702010002010403020301010000000302"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x050202040200010100000001010502010401000000000101000202030200030101030201000000030004000100030001000001030001000002000001000100010200020001030204050303040004050003040100010000020100010000010000000102000002000104010000010000010103010200020301010300000901030100010001000200030002000101000300000100000001000201000301070200040203040104010004000002020400000100000007000000020000030003010100000000000202000101030004040002030600"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

