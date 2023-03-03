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
    %0 = stablehlo.constant dense<"0x0203000300010002040000010302030102010401000301040401070402000503030A0004030001030203020203010300050001010101010007020001030000010101060103040202010301050103020000050202020004020100000106010201020001010603030001020101040002000201020102030102010304040201020100010602010400010304040104020201040001000001000200000100030001010201010703000201030203000002010001000105070201000001010101050202010502030202030100000201000300010402"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0201030401010101060405020301010002040103020101030300020100030300010004030402010302020200030406000003010102000401010402010200010003030001020201020000010001030000000105010303020003040000000106020302030500040203040103030103010500010100000800010401010200000004020002030604020400010601"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x0201000300010001040000010301010002010101000101030300020102000503030A0004030001030203000203000100040001010101010002000001030000010101020003010202010301050103020000050202010002010100000003010001020001010000010001020000000002000201020102030102010304040201020100000302000000010302030103020001020001000001000200000100030001010201010703000201000101000002000001000102000000000000010101040202000102010202030100000201000300010402"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

