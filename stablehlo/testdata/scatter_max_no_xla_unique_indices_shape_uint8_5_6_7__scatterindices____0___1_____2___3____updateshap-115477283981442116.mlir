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
    %0 = stablehlo.constant dense<"0x010200010101000000030302050101020000010104010104030103030203010103010304020202010302030100000000000100000402020101040205050101060201070109010102030607030001000400050301000101020004060301010004040102000100020206040102030100030201040000000300070305010305010105040302030403020300020102000401020100020301030202040302020002020100000004010400000107000401010002010205000005000202020302040302000100010001030000040101010000010102"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0002020500000004010200050204040301000102000100010104000101040202000004020000050205010004020402060500010001000103010601040301000000050301000106020100000201020200030300020101020004000501000003000006070401060303000401040402010105000100000308020105020400060000000302030201060103000108"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010202050101000401030305050404030100010204010104030403030203010103010304020202010302030402020000040200000502050101040205050605060201070109030102030607030001000400050301010601040304060301050304040106020100020206040202030300030201040000000300070305010305010105040402050403020300020607040406030300040304040202040302020002020100000004010400050107000403080202050205000605000203020302040602030101080001030000040101010000010102"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

