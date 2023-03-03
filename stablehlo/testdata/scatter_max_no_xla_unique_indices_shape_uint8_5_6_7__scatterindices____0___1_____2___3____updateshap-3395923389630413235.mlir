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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x1xi32>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) {
    %0 = stablehlo.constant dense<"0x010200000007010002020000040305030203020201010500040200020502000004080200010202010300020101000001000202000300040007000303040102010204010002000000000000000401000303070100000201000601040100010103020001010505000600060101040000010000020500020400030401000004030000000106040000000205000001010701020401010002010201020105040102040002030303000404020103000305010102030201010206020301020100000102020300000301030101020001030004040002"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0002000702080001000000030001010303020103010202020300010101040102020501000001020301000202030000040003000000030006020102030300030200010403010008020103020500010504020002050200000100000200010300000000010106000000000000000002030005050104010303010004030002000503060600020100030302040100"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010200070208010102020003040305030303020301020502040201020502000004080200010202010300020401020205010202010303040007020303040402030204010302060000000000000401000303070100020202030601040200010403020008020505020600060504040002050000020500020400030401000004030000010106040001030205000001010701020401010002010203020105040102040002030303000404050503040305030102040301020206030606020201000303020401000301030101020001030004040002"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

