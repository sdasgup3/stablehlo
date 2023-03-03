// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>)
    %2 = call @expected() : () -> tensor<3x5x40xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xi16>, tensor<2x1xi32>, tensor<3x5x2xi16>) -> tensor<3x5x40xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xi16>, tensor<3x5x40xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>) {
    %0 = stablehlo.constant dense<"0x01000100FBFFFFFF0100000001000000FEFFFFFFFEFF000004000000FEFF0200FBFFFEFF0100000001000000FCFFFFFF0000FFFFFAFF0100FFFF000001000100020000000100FEFF07000100000005000100FFFF0300FDFF01000500FDFFFFFF00000200030003000300040003000000FFFF0200FEFF000000000500050008000100FBFF050005000200FFFF000000000000000002000100FFFF0300FFFFFFFF0100000003000100FEFFFDFFFCFFFCFF010000000000FFFFFFFFFFFF000003000200FEFF0400FEFFFEFF0100FDFF05000100F9FF000002000100000000000000000000000100FEFFFBFFFEFFFEFF000002000400FEFFFEFFFEFFFDFFFFFFFEFF000003000000FCFFFFFF0100FBFF0200FFFFFEFF0400FFFFFFFFFFFF02000500FFFFFFFFFFFFFEFF00000000FEFFFEFF0100030000000000FCFF000004000000030000000000FFFF000004000500040003000300FFFFFFFF0100FCFFFDFF0100000001000000FEFF000000000000FBFFFCFFFCFFF9FFFAFF0000FFFF02000000FEFFFCFFFDFFFCFF0000FBFF020000000200FEFFFCFF0000FFFFFEFFFFFFFFFFFFFF0300FFFF0200000000000000FFFFFFFF010001000000FEFFFCFFFDFF0100FFFF0300FDFF0000FEFFFCFFFDFF00000400FDFFFEFF00000000000003000100010003000000FEFFFFFFFFFF00000400010001000000FFFF0000FEFFFEFF00000300FFFF0100FBFFFDFFFDFF0200FDFF0200FDFFFEFFFCFFFDFF0400FDFF0300FAFFF9FF03000200FBFF010001000000FEFF0000020000000200FEFF010000000000FCFFFBFFF9FFFFFFFBFF0100FDFF020001000000000000000500020000000300FFFFFCFF00000000FDFF000000000100FCFF00000000020001000200020001000000FCFFFEFF040000000000FDFF0300000002000200FEFF00000100FFFFFFFF0400FFFF0400FEFF0100000004000400FFFF0200FFFF03000200FEFFFDFFFEFFFEFFFFFF01000200FBFFFFFFFCFF0500FBFF01000000FEFF01000400FEFF0100FEFF0500FCFF0400FCFF02000400000000000000FAFF0000FCFFFEFFFDFF0400070004000100FFFF01000100FDFFFEFFFEFF00000000FDFF0300FEFFFEFF0000000003000100020002000100FFFF0100FBFF000003000100FDFFFDFF010001000100FBFF010000000300FEFFFDFF0300FFFF0100F9FF00000100FFFF00000100FCFFFBFFFBFFFFFF020001000100010004000000FFFF0100FFFF030000000100FFFF0400030001000000FCFFFEFF0700020002000000000000000100FEFF03000400FEFF00000200FAFF0100FEFF02000200FEFF01000100FFFF0000000000000000FBFFFEFF00000000010000000100FCFFFDFF0200FFFFFFFF0000F9FF01000300FDFF0200FDFF000002000000FDFF03000200030000000600FFFFFBFF00000400FAFF040001000100FEFF0100FDFF0000FEFF0600000003000000FEFFFCFFFDFF0300FFFF000000000000FFFF0000010000000000000004000300FEFF0100FEFF020003000100FFFFFEFFFFFFFEFF0200FBFF000000000200000001000000FFFF0200000000000200000004000500FFFF00000000020001000000FEFF0000FEFF03000300040004000000050003000100FFFF0100FFFFFFFF04000000FFFFFFFF000000000300FDFF02000100"> : tensor<3x5x40xi16>
    %1 = stablehlo.constant dense<[[[1, -2], [0, -3], [0, 0], [-6, 0], [0, -2]], [[0, -6], [-2, 0], [3, 1], [4, 0], [-3, -1]], [[3, -2], [4, 5], [0, 3], [-5, 1], [-2, -1]]]> : tensor<3x5x2xi16>
    return %0, %1 : tensor<3x5x40xi16>, tensor<3x5x2xi16>
  }
  func.func private @expected() -> tensor<3x5x40xi16> {
    %0 = stablehlo.constant dense<"0x0100FEFFFBFFFFFF0100000001000000FEFFFFFFFEFF000004000000FEFF0200FBFFFEFF0100000001000000FCFFFFFF0000FFFFFAFF0100FFFF000001000100020000000100FEFF07000100000005000100FDFF0300FDFF01000500FDFFFFFF00000200030003000300040003000000FFFF0200FEFF000000000500050008000100FBFF050005000200FFFF000000000000000002000100FFFF0300FFFFFFFF0100000003000100FEFFFDFFFCFFFCFF010000000000FFFFFFFFFFFF000003000200FEFF0400FEFFFEFF0100FDFF05000100F9FF000002000100000000000000000000000100FEFFFBFFFEFFFEFF00000200FAFFFEFFFEFFFEFFFDFFFFFFFEFF000003000000FCFFFFFF0100FBFF0200FFFFFEFF0400FFFFFFFFFFFF02000500FFFFFFFFFFFFFEFF00000000FEFFFEFF0100030000000000FCFF0000040000000300FEFF0000FFFF000004000500040003000300FFFFFFFF0100FCFFFDFF0100000001000000FEFF000000000000FBFFFCFFFCFFF9FFFAFF0000FFFF02000000FEFFFCFFFDFFFCFF0000FBFF020000000200FAFFFCFF0000FFFFFEFFFFFFFFFFFFFF0300FFFF0200000000000000FFFFFFFF010001000000FEFFFCFFFDFF0100FFFF0300FDFF0000FEFFFCFFFDFF00000400FDFFFEFF000000000000030001000100FEFF0000FEFFFFFFFFFF00000400010001000000FFFF0000FEFFFEFF00000300FFFF0100FBFFFDFFFDFF0200FDFF0200FDFFFEFFFCFFFDFF0400FDFF0300FAFFF9FF03000200FBFF010001000000FEFF0000020000000200FEFF010000000000FCFFFBFFF9FFFFFFFBFF0100FDFF020001000000000000000500020000000300FFFFFCFF00000000FDFF000000000100FCFF00000000020001000200020001000000FCFFFEFF040000000000FDFF0300000002000200FEFF00000100FFFFFFFF0400FFFF0400FEFF0100000004000400FFFF0200FFFF03000200FEFFFDFFFEFFFEFFFFFF01000200FBFFFFFFFCFF0500FBFF01000000FEFF01000400FEFF0100FEFF0500FCFF0400FCFF02000400000000000000FAFF0000FCFFFEFFFDFF0400070004000100FFFF01000100FDFFFEFFFEFF00000000FDFF0300FEFFFEFF0000FEFF03000100020002000100FFFF0100FBFF000003000100FDFFFDFF010001000100FBFF010000000300FEFFFDFF0300FFFF0100F9FF00000100FFFF00000100FCFFFBFFFBFFFFFF020001000100010004000000FFFF0100FFFF030000000100FFFF0400030001000000FCFFFEFF0700020002000000000000000100FEFF03000400FEFF00000200FAFF0100FEFF02000200FEFF01000100FFFF0000000000000000FBFFFEFF00000000010000000100FCFFFDFF0200FFFFFFFF0000F9FF01000300FDFF0200FDFF000002000000FDFF03000200030000000600FFFFFBFF00000400FAFF040001000100FEFF0100FDFFFBFFFEFF0600000003000000FEFFFCFFFDFF0300FFFF000000000000FFFF0000010000000000000004000300FEFF0100FEFF020003000100FFFFFEFFFFFFFEFF0200FBFF000000000200000001000000FEFF0200000000000200000004000500FFFF00000000020001000000FEFF0000FEFF03000300040004000000050003000100FFFF0100FFFFFFFF04000000FFFFFFFF000000000300FDFF02000100"> : tensor<3x5x40xi16>
    return %0 : tensor<3x5x40xi16>
  }
}

