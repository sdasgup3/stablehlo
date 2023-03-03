// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %2 = call @expected() : () -> tensor<5x6x7xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf16>, tensor<2x2x2xi32>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>) {
    %0 = stablehlo.constant dense<"0x6DC0BA403538DCBDDF40B8A9A12EBF40A2C0E84281C2C241D530EA3CE73D8CC40EC62B386BBC5EB90836A3BD0348F338ED424FBD054349C1E84426C4904001C672BFCDBC2DB3FE477F3DCA43A1B855C1FAC21EC154BFF344C8BF1D4098B6603B60BF133E61C5DF3DDEC64EC01CC3724407A48FC25CC5A3BCDD45E0BEA2C14230913AC23E3743473B5CC765431642AFBA05BC7CBF813D8FC066457FC4DFC0CD41983FF235ADB8C6404743CD418CC1123F81C05AA8B1BF34C179C4F3C00EBCA041BFC347BE864061C22EBEFDC181BCABC775343E4132B39B3DAD3C3A45BA40883D53399AC24540F8C04DBCD33C64C4B6C5C6C091403446FCBAC0BC7A40DF3EDAC090448D37073256C2774041C5383EC7C03A415C457DC1BCC042C40E3D604128C8F3BF153EFD35BD3218C89B1EE9C061C05DBD87C12B400D3C4F3F6CBA6FC29EB5E6C10D3D1EC2D7C7BBAB873F73B8B2BAB2BF8E442E348B342EB6C74013435C43BAC4C0401FC5CCC09F307EC266369AC241BF69C064B83EB8B5B0CB416B416C4572C35641B4BE1CB9CABECCC3EF421046673B6EC0CD39663D3F42A9BAA53EC03872B138C4"> : tensor<5x6x7xf16>
    %1 = stablehlo.constant dense<[[[2.666020e-01, -2.009770e+00], [1.698240e+00, -1.306640e+00]], [[4.468750e+00, 3.207030e+00], [-3.082030e+00, 2.953130e+00]], [[-1.972660e+00, 2.525390e+00], [3.203130e+00, -2.410160e+00]], [[4.475100e-01, -5.613280e+00], [8.062500e+00, 2.296880e+00]], [[1.724610e+00, -2.894530e+00], [-2.324220e+00, 7.211910e-01]]]> : tensor<5x2x2xf16>
    return %0, %1 : tensor<5x6x7xf16>, tensor<5x2x2xf16>
  }
  func.func private @expected() -> tensor<5x6x7xf16> {
    %0 = stablehlo.constant dense<"0x6DC00A393538DCBDDF40B8A9A12EBF40A2C083C481C2C241D530EA3CE73D8CC40EC630BC6BBC5EB90836A3BD0348F338ED424FBD054349C12A4826C4904001C672BFCDBC2DB3FE477F3DCA43A1B855C1FAC21EC154BF874DC8BF1D4098B6603B60BF133E61C55644DEC64EC01CC3724407A48FC25CC56FC3DD45E0BEA2C14230913AC23E3743473B5CC76543B0C8AFBA05BC7CBF813D8FC066457FC4DFC0CD41983FF235ADB8C6404743B9C58CC1123F81C05AA8B1BF34C179C4F7450EBCA041BFC347BE864061C22EBE90C781BCABC775343E4132B39B3DAD3C3A45BA40883D43409AC24540F8C04DBCD33C64C4B6C5C6C091403446FCBAC0BC7A40DF3E58BC90448D37073256C2774041C5383E7DC53A415C457DC1BCC042C40E3D6041D551F3BF153EFD35BD3218C89B1EE9C061C05DBD87C1334C0D3C4F3F6CBA6FC29EB5E6C10D3D1EC2D7C7BBAB873F73B8B2BAB2BFDB472E348B342EB6C74013435C43BAC4DA3E1FC5CCC09F307EC266369AC241BF624664B83EB8B5B0CB416B416C4572C35641B4BE1CB9E443CCC3EF421046673B6EC0CD39663D3F42A9BAA53EC03872B138C4"> : tensor<5x6x7xf16>
    return %0 : tensor<5x6x7xf16>
  }
}

