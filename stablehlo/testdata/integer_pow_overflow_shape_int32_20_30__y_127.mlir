// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xi32>
    %1 = call @expected() : () -> tensor<20x30xi32>
    %2 = call @integer_pow(%0) : (tensor<20x30xi32>) -> tensor<20x30xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x30xi32>, tensor<20x30xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xi32> {
    %0 = stablehlo.constant dense<"0xFFFFFFFF04000000FDFFFFFFFCFFFFFF0200000000000000FCFFFFFFFEFFFFFF02000000020000000000000002000000FEFFFFFFFEFFFFFF03000000000000000200000000000000FFFFFFFFFCFFFFFF02000000FEFFFFFFFFFFFFFF060000000200000004000000FBFFFFFFFEFFFFFF0200000000000000FDFFFFFF05000000020000000000000002000000FEFFFFFF010000000000000000000000010000000300000002000000FBFFFFFF0300000002000000FEFFFFFFFFFFFFFFFCFFFFFFFCFFFFFFFCFFFFFF0200000001000000020000000300000003000000FFFFFFFFFFFFFFFFFDFFFFFF0300000000000000FFFFFFFFFFFFFFFFFEFFFFFF01000000000000000100000003000000000000000000000000000000FDFFFFFFFFFFFFFF0000000000000000020000000000000000000000FFFFFFFFFEFFFFFF0200000002000000FFFFFFFF02000000FEFFFFFF000000000000000000000000FDFFFFFF00000000FEFFFFFF0000000002000000000000000000000000000000FEFFFFFF0000000003000000FEFFFFFFFFFFFFFF00000000FAFFFFFFFBFFFFFFFDFFFFFFFDFFFFFF0200000001000000010000000400000003000000FFFFFFFF05000000FEFFFFFFFCFFFFFF010000000000000000000000FFFFFFFFFFFFFFFF0200000005000000000000000600000001000000060000000100000005000000FFFFFFFF01000000FDFFFFFFFCFFFFFF08000000FFFFFFFF000000000000000000000000FEFFFFFF01000000030000000100000005000000FFFFFFFF0200000001000000010000000400000000000000FEFFFFFF03000000020000000000000003000000010000000000000000000000FEFFFFFFFEFFFFFF00000000FDFFFFFF00000000FCFFFFFF040000000300000000000000FDFFFFFF01000000FFFFFFFF00000000F9FFFFFF0500000000000000FEFFFFFFFFFFFFFFFFFFFFFF0300000001000000FFFFFFFF010000000100000002000000FEFFFFFFFEFFFFFF030000000000000000000000000000000000000005000000FFFFFFFFFAFFFFFF00000000FBFFFFFF01000000010000000100000003000000FCFFFFFF00000000000000000100000002000000FDFFFFFFFFFFFFFF0400000000000000FEFFFFFF0300000006000000F8FFFFFFFDFFFFFF000000000000000000000000FCFFFFFF01000000010000000100000002000000FEFFFFFF0000000001000000FBFFFFFF00000000FDFFFFFF00000000FEFFFFFF00000000FEFFFFFF0100000002000000000000000500000002000000FDFFFFFF0500000000000000FEFFFFFF000000000000000002000000FDFFFFFF0400000001000000FFFFFFFF040000000000000002000000050000000600000000000000FDFFFFFFFCFFFFFF000000000000000002000000FDFFFFFF0000000001000000FEFFFFFF000000000200000000000000000000000000000002000000FBFFFFFFFDFFFFFF00000000020000000000000005000000FFFFFFFFFAFFFFFFFEFFFFFF00000000FDFFFFFF01000000FFFFFFFF000000000100000001000000FEFFFFFF000000000100000000000000FEFFFFFF00000000FEFFFFFF0100000001000000010000000100000002000000FFFFFFFFFBFFFFFFFFFFFFFF02000000FFFFFFFFFFFFFFFFFDFFFFFF0000000002000000040000000200000003000000FFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF0200000000000000FFFFFFFFFEFFFFFFFFFFFFFF00000000FFFFFFFF00000000FFFFFFFF02000000FCFFFFFF00000000FFFFFFFF0400000001000000010000000000000004000000020000000500000000000000FDFFFFFFFDFFFFFFFFFFFFFF04000000FFFFFFFF0000000006000000FEFFFFFFFEFFFFFF00000000FFFFFFFF03000000FEFFFFFF03000000FEFFFFFFFEFFFFFF03000000FBFFFFFF0000000001000000FDFFFFFF04000000FFFFFFFF0000000006000000FDFFFFFFFFFFFFFF0600000002000000FDFFFFFFFDFFFFFF020000000000000000000000FCFFFFFFFEFFFFFFFFFFFFFFFCFFFFFF000000000000000001000000FCFFFFFFFAFFFFFFFEFFFFFF0000000000000000FCFFFFFF0200000001000000020000000100000001000000FEFFFFFF000000000400000000000000000000000300000003000000020000000000000000000000FFFFFFFF0000000005000000FFFFFFFF02000000FEFFFFFF0100000001000000FEFFFFFF030000000000000000000000010000000000000000000000FEFFFFFFFFFFFFFF000000000000000000000000FAFFFFFF00000000FDFFFFFF0000000003000000FEFFFFFF00000000FEFFFFFF010000000000000001000000FDFFFFFF00000000000000000100000001000000FFFFFFFFFDFFFFFF0300000005000000FFFFFFFF01000000000000000100000003000000FAFFFFFFFFFFFFFFFEFFFFFF010000000200000005000000FFFFFFFFFAFFFFFFFFFFFFFFFEFFFFFF0000000001000000010000000000000000000000FEFFFFFF00000000FEFFFFFF0100000001000000FDFFFFFF02000000FEFFFFFFFFFFFFFFFEFFFFFFF8FFFFFFFCFFFFFF000000000000000005000000FEFFFFFF0000000006000000FEFFFFFF0000000005000000FBFFFFFF0000000000000000000000000200000001000000FDFFFFFF00000000030000000100000000000000FCFFFFFF00000000FDFFFFFF020000000200000003000000FAFFFFFF000000000500000003000000FEFFFFFF06000000FCFFFFFF020000000000000000000000FCFFFFFF040000000000000000000000000000000300000003000000F9FFFFFF02000000FFFFFFFF000000000300000000000000FDFFFFFF00000000FDFFFFFFFEFFFFFF0100000003000000FFFFFFFF010000000000000002000000FFFFFFFF00000000030000000100000000000000020000000200000005000000FFFFFFFFFFFFFFFFFEFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF000000000200000000000000000000000000000000000000030000000100000008000000FDFFFFFF0200000000000000FDFFFFFF0000000002000000FCFFFFFF00000000FFFFFFFF05000000010000000000000000000000FAFFFFFF0000000000000000FBFFFFFFFEFFFFFF03000000FBFFFFFF04000000F9FFFFFF00000000000000000200000001000000FEFFFFFF000000000500000002000000FFFFFFFF00000000FEFFFFFF000000000200000007000000FAFFFFFF00000000FDFFFFFFFFFFFFFF03000000FFFFFFFFFEFFFFFF0400000001000000FEFFFFFFFFFFFFFF010000000200000000000000"> : tensor<20x30xi32>
    return %0 : tensor<20x30xi32>
  }
  func.func private @expected() -> tensor<20x30xi32> {
    %0 = stablehlo.constant dense<"0xFFFFFFFF0000000055D77D7C0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000AB288283000000000000000000000000FFFFFFFF000000000000000000000000FFFFFFFF000000000000000000000000338D489000000000000000000000000055D77D7CCD72B76F0000000000000000000000000000000001000000000000000000000001000000AB28828300000000338D4890AB2882830000000000000000FFFFFFFF000000000000000000000000000000000100000000000000AB288283AB288283FFFFFFFFFFFFFFFF55D77D7CAB28828300000000FFFFFFFFFFFFFFFF00000000010000000000000001000000AB28828300000000000000000000000055D77D7CFFFFFFFF0000000000000000000000000000000000000000FFFFFFFF000000000000000000000000FFFFFFFF000000000000000000000000000000000000000055D77D7C000000000000000000000000000000000000000000000000000000000000000000000000AB28828300000000FFFFFFFF0000000000000000338D489055D77D7C55D77D7C00000000010000000100000000000000AB288283FFFFFFFFCD72B76F0000000000000000010000000000000000000000FFFFFFFFFFFFFFFF00000000CD72B76F0000000000000000010000000000000001000000CD72B76FFFFFFFFF0100000055D77D7C0000000000000000FFFFFFFF0000000000000000000000000000000001000000AB28828301000000CD72B76FFFFFFFFF000000000100000001000000000000000000000000000000AB2882830000000000000000AB28828301000000000000000000000000000000000000000000000055D77D7C000000000000000000000000AB2882830000000055D77D7C01000000FFFFFFFF0000000049DE2945CD72B76F0000000000000000FFFFFFFFFFFFFFFFAB28828301000000FFFFFFFF0100000001000000000000000000000000000000AB28828300000000000000000000000000000000CD72B76FFFFFFFFF0000000000000000338D4890010000000100000001000000AB288283000000000000000000000000010000000000000055D77D7CFFFFFFFF000000000000000000000000AB288283000000000000000055D77D7C0000000000000000000000000000000001000000010000000100000000000000000000000000000001000000338D48900000000055D77D7C00000000000000000000000000000000010000000000000000000000CD72B76F0000000055D77D7CCD72B76F000000000000000000000000000000000000000055D77D7C0000000001000000FFFFFFFF000000000000000000000000CD72B76F000000000000000055D77D7C0000000000000000000000000000000055D77D7C000000000100000000000000000000000000000000000000000000000000000000000000338D489055D77D7C000000000000000000000000CD72B76FFFFFFFFF00000000000000000000000055D77D7C01000000FFFFFFFF000000000100000001000000000000000000000001000000000000000000000000000000000000000100000001000000010000000100000000000000FFFFFFFF338D4890FFFFFFFF00000000FFFFFFFFFFFFFFFF55D77D7C00000000000000000000000000000000AB288283FFFFFFFFFFFFFFFFFFFFFFFF000000000000000055D77D7CFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF000000000000000000000000FFFFFFFF000000000100000001000000000000000000000000000000CD72B76F0000000055D77D7C55D77D7CFFFFFFFF00000000FFFFFFFF0000000000000000000000000000000000000000FFFFFFFFAB28828300000000AB2882830000000000000000AB288283338D4890000000000100000055D77D7C00000000FFFFFFFF000000000000000055D77D7CFFFFFFFF000000000000000055D77D7C55D77D7C0000000000000000000000000000000000000000FFFFFFFF0000000000000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000001000000010000000000000000000000000000000000000000000000AB288283AB288283000000000000000000000000FFFFFFFF00000000CD72B76FFFFFFFFF0000000000000000010000000100000000000000AB288283000000000000000001000000000000000000000000000000FFFFFFFF000000000000000000000000000000000000000055D77D7C00000000AB28828300000000000000000000000001000000000000000100000055D77D7C00000000000000000100000001000000FFFFFFFF55D77D7CAB288283CD72B76FFFFFFFFF010000000000000001000000AB28828300000000FFFFFFFF000000000100000000000000CD72B76FFFFFFFFF00000000FFFFFFFF000000000000000001000000010000000000000000000000000000000000000000000000010000000100000055D77D7C0000000000000000FFFFFFFF0000000000000000000000000000000000000000CD72B76F0000000000000000000000000000000000000000CD72B76F338D4890000000000000000000000000000000000100000055D77D7C00000000AB2882830100000000000000000000000000000055D77D7C0000000000000000AB2882830000000000000000CD72B76FAB2882830000000000000000000000000000000000000000000000000000000000000000000000000000000000000000AB288283AB28828349DE294500000000FFFFFFFF00000000AB2882830000000055D77D7C0000000055D77D7C0000000001000000AB288283FFFFFFFF010000000000000000000000FFFFFFFF00000000AB28828301000000000000000000000000000000CD72B76FFFFFFFFFFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFF000000000000000000000000000000000000000000000000AB288283010000000000000055D77D7C000000000000000055D77D7C00000000000000000000000000000000FFFFFFFFCD72B76F010000000000000000000000000000000000000000000000338D489000000000AB288283338D48900000000049DE2945000000000000000000000000010000000000000000000000CD72B76F00000000FFFFFFFF00000000000000000000000000000000B721D6BA000000000000000055D77D7CFFFFFFFFAB288283FFFFFFFF00000000000000000100000000000000FFFFFFFF010000000000000000000000"> : tensor<20x30xi32>
    return %0 : tensor<20x30xi32>
  }
  func.func private @integer_pow(%arg0: tensor<20x30xi32>) -> tensor<20x30xi32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<20x30xi32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<20x30xi32>
    %2 = stablehlo.multiply %0, %0 : tensor<20x30xi32>
    %3 = stablehlo.multiply %1, %2 : tensor<20x30xi32>
    %4 = stablehlo.multiply %2, %2 : tensor<20x30xi32>
    %5 = stablehlo.multiply %3, %4 : tensor<20x30xi32>
    %6 = stablehlo.multiply %4, %4 : tensor<20x30xi32>
    %7 = stablehlo.multiply %5, %6 : tensor<20x30xi32>
    %8 = stablehlo.multiply %6, %6 : tensor<20x30xi32>
    %9 = stablehlo.multiply %7, %8 : tensor<20x30xi32>
    %10 = stablehlo.multiply %8, %8 : tensor<20x30xi32>
    %11 = stablehlo.multiply %9, %10 : tensor<20x30xi32>
    return %11 : tensor<20x30xi32>
  }
}
