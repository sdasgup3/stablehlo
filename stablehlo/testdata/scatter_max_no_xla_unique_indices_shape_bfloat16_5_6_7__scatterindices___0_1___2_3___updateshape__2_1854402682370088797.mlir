// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %2 = call @expected() : () -> tensor<5x6x7xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xbf16>, tensor<2x2xi32>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>) {
    %0 = stablehlo.constant dense<"0x50BF2E4042BF25C0BAC0DC3E703E873E124039BF443F5FC0153F2E4005C0D0C0013FB83F9DBF0C406D3F97BF32C0DDBF0BC0A0C022C0C4BF3040A2C04540953FC9BF85BF8A40403F194090C08C408AC041C000C0C5BF06C01340923F28C05A403EC063C0D6BF1D4000C01BC09D3FCDC04EC0833F3B4030C07DBF0FBF9440873F12401340D83F16C08FC07C40693F88BF8DC0FABF09405F40C33F0C3FBD3CC33FDC3F6CC034C02D3D0AC0013F7AC0053E12C092C06DBE3D40B03FA2BFB4BF05BF034011BC9CC0633F3BBF6A3F1ABE0B3F534044BF8A400A3F913FF73ED03F9ABD32C08540824087408FBE2FC032BF42BFEDBF4B409CC0FA3E1440B93FCA3E2EC0FC3E2AC0A63EC2BDC83FBF3F4AC0D83F9240833F35C0FDBFD63F314052BF80C07E3FC53E15BDFFBF0240F1BF47C0C4C0653FFC3F2440D3BF374074400B40363F6040CEBF0341A73E3440F5BF9B3E8540B63F72C0F43FFF408D40784017C044BE29C03840A33D2DC0FE3F69C081C022C079BF833EBABF5B3E873F0640D5C08B3E7B40E9C09A40C1BFB83FA9C0A53FC9C054BF83406B3E30400CC0803FEA3E3E40EFBCC6BF"> : tensor<5x6x7xbf16>
    %1 = stablehlo.constant dense<[[-6.218750e+00, -6.210940e-01, 3.468750e+00, -2.781250e+00, -3.062500e+00, 2.597660e-01, -1.289060e+00], [-3.906250e-03, 2.285160e-01, 7.851560e-01, -2.515630e+00, -3.093750e+00, 1.367190e+00, 5.585940e-01]]> : tensor<2x7xbf16>
    return %0, %1 : tensor<5x6x7xbf16>, tensor<2x7xbf16>
  }
  func.func private @expected() -> tensor<5x6x7xbf16> {
    %0 = stablehlo.constant dense<"0x50BF2E4042BF25C0BAC0DC3E703E873E12405E40443F44C0153F2E4005C0D0C0013FB83F9DBF0C406D3F97BF32C0DDBF0BC0A0C022C0C4BF3040A2C04540953FC9BF85BF8A40403F194090C08C408AC041C000C0C5BF06C01340923F28C05A403EC063C0D6BF1D4000C01BC09D3FCDC04EC0833F3B4030C07DBF0FBF9440873F12401340D83F16C08FC07C40693F88BF8DC0FABF09405F40C33F0C3FBD3CC33FDC3F6CC034C02D3D0AC0013F7AC0053E12C092C06DBE3D40B03FA2BFB4BF05BF034011BC9CC0633F3BBF6A3F1ABE0B3F534080BB8A40493F913FF73ED03F0F3F32C08540824087408FBE2FC032BF42BFEDBF4B409CC0FA3E1440B93FCA3E2EC0FC3E2AC0A63EC2BDC83FBF3F4AC0D83F9240833F35C0FDBFD63F314052BF80C07E3FC53E15BDFFBF0240F1BF47C0C4C0653FFC3F2440D3BF374074400B40363F6040CEBF0341A73E3440F5BF9B3E8540B63F72C0F43FFF408D40784017C044BE29C03840A33D2DC0FE3F69C081C022C079BF833EBABF5B3E873F0640D5C08B3E7B40E9C09A40C1BFB83FA9C0A53FC9C054BF83406B3E30400CC0803FEA3E3E40EFBCC6BF"> : tensor<5x6x7xbf16>
    return %0 : tensor<5x6x7xbf16>
  }
}

