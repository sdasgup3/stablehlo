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
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x1xi32>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) {
    %0 = stablehlo.constant dense<"0x000101030600000104040101030201040003010404000003050200060303030401010306030102010803030201000100020003030103020001000303050404010202010201010004000300000001020100020002010301020003020001060103000101050200030002020200000202020100010100010302060602010002020203050002030200020400020004020403000001000501000303020301030002010002020101020005030502020303000203010001030100010401010102000101000200000100030801010000000202050102"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0101030202020100010200010100000503010100050000040002000104030001000300000500000202000100010401000002000200060000000206010201010006000100010300030101040302010200000104020201000301010103050302040403030201020602000002010205000101010100020002010501020100000001040103040100000104030403"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010103020202010001020001010000050301010005000004000200010303030401010306030102010803040300010003000005000002020001000104010000020002000600000004000300000001020100020002000206010201010006000100010300030101040302010200000104020100010100010302060602010002020100030101010305030204040303020102060200000201020500010301030002010002020101020005010101000200020105010201000000010401030401000001040304030100030801010000000202050102"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

