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
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x1xi32>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) {
    %0 = stablehlo.constant dense<"0x040002000000010103020101050003060002010403010002010002010101020100030100010006000000000804050001000403010002010202050005050000000300020201010100000202000205010404010001000101010300010004000200030206020503000100020200010501030004020202020600050101040203000102010103010203030003000002050104050100030100040103040304030000000505000100010002030005040303030701000101030200020001010003010401020105010500030100050204010102000001"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0106060104010004030101000101010302010003010201020000010201020001000102010103010200010206020102040100020502050101050008010101000300030004030104000301020202010102010503010106020001020201020704010101000000010101000303020200030202030508040001000000000200000100030003030101010003020100"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x050608010401010506030201060104090203010704030104010003030101020100030100010006000000010A040600020205040401040103040B020607040100050504070202010000020200020501040401000105010902040101030403020406030A020804020302030302020A040400040202020206000501010402030107040102050303050A040401010205010506020006040206010606030403000000050500010001000205030A0C0703040701000103030201020301040304020501050306010500030100050204010102000001"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

