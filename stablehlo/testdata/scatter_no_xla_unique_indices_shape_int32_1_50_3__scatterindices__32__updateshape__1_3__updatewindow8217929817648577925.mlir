// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi32>, tensor<1x3xi32>)
    %2 = call @expected() : () -> tensor<1x50x3xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      stablehlo.return %arg1 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi32>, tensor<1xi32>, tensor<1x3xi32>) -> tensor<1x50x3xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi32>, tensor<1x50x3xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi32>, tensor<1x3xi32>) {
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFF05000000FDFFFFFF0500000003000000FFFFFFFFFFFFFFFFFFFFFFFF00000000FEFFFFFFFDFFFFFFFEFFFFFFFEFFFFFFFDFFFFFFF9FFFFFFFCFFFFFFFFFFFFFF0300000001000000FEFFFFFFFCFFFFFFFCFFFFFFFBFFFFFF00000000000000000000000000000000000000000000000000000000FFFFFFFF0100000000000000FDFFFFFF08000000FCFFFFFF00000000FDFFFFFF0400000000000000FFFFFFFFFDFFFFFF00000000000000000000000000000000FFFFFFFF040000000200000003000000030000000500000000000000070000000000000005000000020000000100000003000000FFFFFFFF000000000000000000000000FDFFFFFF01000000FEFFFFFF000000000000000002000000FFFFFFFF01000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFF000000000000000002000000010000000200000002000000FEFFFFFFFEFFFFFF000000000000000000000000FEFFFFFFFEFFFFFF010000000100000002000000000000000000000001000000FFFFFFFF010000000400000003000000FDFFFFFF010000000300000001000000FEFFFFFF0400000000000000FFFFFFFF000000000000000000000000000000000100000002000000FCFFFFFF000000000100000001000000000000000200000001000000FDFFFFFF01000000FFFFFFFFFDFFFFFFFEFFFFFFFCFFFFFF0100000000000000FFFFFFFFFEFFFFFF050000000300000001000000FBFFFFFF03000000FDFFFFFFFEFFFFFF020000000000000001000000FEFFFFFF01000000FFFFFFFFFEFFFFFF00000000FFFFFFFFFEFFFFFF09000000FFFFFFFF"> : tensor<1x50x3xi32>
    %1 = stablehlo.constant dense<[[0, -1, -1]]> : tensor<1x3xi32>
    return %0, %1 : tensor<1x50x3xi32>, tensor<1x3xi32>
  }
  func.func private @expected() -> tensor<1x50x3xi32> {
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFF05000000FDFFFFFF0500000003000000FFFFFFFFFFFFFFFFFFFFFFFF00000000FEFFFFFFFDFFFFFFFEFFFFFFFEFFFFFFFDFFFFFFF9FFFFFFFCFFFFFFFFFFFFFF0300000001000000FEFFFFFFFCFFFFFFFCFFFFFFFBFFFFFF00000000000000000000000000000000000000000000000000000000FFFFFFFF0100000000000000FDFFFFFF08000000FCFFFFFF00000000FDFFFFFF0400000000000000FFFFFFFFFDFFFFFF00000000000000000000000000000000FFFFFFFF040000000200000003000000030000000500000000000000070000000000000005000000020000000100000003000000FFFFFFFF000000000000000000000000FDFFFFFF01000000FEFFFFFF000000000000000002000000FFFFFFFF01000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFF000000000000000002000000010000000200000002000000FEFFFFFFFEFFFFFF000000000000000000000000FEFFFFFFFEFFFFFF01000000010000000200000000000000000000000100000000000000FFFFFFFFFFFFFFFF03000000FDFFFFFF010000000300000001000000FEFFFFFF0400000000000000FFFFFFFF000000000000000000000000000000000100000002000000FCFFFFFF000000000100000001000000000000000200000001000000FDFFFFFF01000000FFFFFFFFFDFFFFFFFEFFFFFFFCFFFFFF0100000000000000FFFFFFFFFEFFFFFF050000000300000001000000FBFFFFFF03000000FDFFFFFFFEFFFFFF020000000000000001000000FEFFFFFF01000000FFFFFFFFFEFFFFFF00000000FFFFFFFFFEFFFFFF09000000FFFFFFFF"> : tensor<1x50x3xi32>
    return %0 : tensor<1x50x3xi32>
  }
}

