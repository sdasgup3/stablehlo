// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %2 = call @expected() : () -> tensor<5x6x7xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xf16>, tensor<2x2xi32>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16>, tensor<2x7xf16>) {
    %0 = stablehlo.constant dense<"0x7E3E2C473BBFB2C4F6C1D53B55C06FC54445A5412D4267BEC93D3942D7400A4448C1BFC23C3F5242943847BBCC4070B189244C3C3042C1437B3B0BBEC7C00E434AC46DC2FA46683D3CBC2C2D92C1DFBFAD3E2A4633BD144119B8A23854380A3D12C69EB46BC505BEE0BDF84596BE7B41F4C4CC42A542DC45B03B47347F4253C34C383B3D7CC0654274C3903601BEBD3F63C3F1413D2FDB424C435640FF3DD6B17E3E06C2CABCEBC401410D44AFBB3DB2B942CFC405442FBC2039D13FE9BEA9BB40C699B90746E9C29C3C222DCCBF24C42EBD473AE3C49DB07E3D9B40D8BFB63C0FC033C4D938CCC2C43751BF0BC53DC418AFAD309C3F11C14BB13E40293C10C788C0863E0443BC41DBBFE9B4B6C1BDC5A9BCA83D53B87E43A5C18542F242C6BDD141C03CEDC32ABB4AB0A3BD633721349B3AE141E84004C657317E4139C0DAC5C032FEBE2F4459C7A13DC3B497B88F441243DD3D68BB9D3CF846762E0641404408ACF3C264382B3E953E2C3EB3C16533FB4546B474BD26C42D41D7B7A3C0FD40ACB6AC3589B8F8C56AC5C53E074146C1FB3FF53CEB3EB2C132C34E432FBC6A3AFD3FF938"> : tensor<5x6x7xf16>
    %1 = stablehlo.constant dense<[[1.750980e+00, -1.882810e+00, 2.378910e+00, -3.033200e+00, -6.772460e-01, 1.436770e-01, 3.437500e+00], [9.794920e-01, 5.574210e+00, 1.020510e+00, 2.433590e+00, -2.348630e-01, -3.287110e+00, -2.832030e+00]]> : tensor<2x7xf16>
    return %0, %1 : tensor<5x6x7xf16>, tensor<2x7xf16>
  }
  func.func private @expected() -> tensor<5x6x7xf16> {
    %0 = stablehlo.constant dense<"0x7E3E2C473BBFB2C4F6C1D53B55C0C2C8F5C8B746AFC8563CA6325949D7400A4448C1BFC23C3F5242943847BBCC4070B189244C3C3042C1437B3B0BBEC7C00E434AC46DC2FA46683D3CBC2C2D92C1DFBFAD3E2A4633BD144119B8A23854380A3D12C69EB46BC505BEE0BDF84596BE7B41F4C4CC42A542DC45B03B47347F4253C34C383B3D7CC0654274C3903601BEBD3F63C3F1413D2FDB424C435640FF3DD6B17E3E06C2CABCEBC401410D44AFBB3DB2B942CFC405442FBC2039D13FE9BEA9BB40C699B90746E9C29C3C222DCCBF24C42EBD263ACFCEB5B0AF4254B87246ACC20FC033C4D938CCC2C43751BF0BC53DC418AFAD309C3F11C14BB13E40293C10C788C0863E0443BC41DBBFE9B4B6C1BDC5A9BCA83D53B87E43A5C18542F242C6BDD141C03CEDC32ABB4AB0A3BD633721349B3AE141E84004C657317E4139C0DAC5C032FEBE2F4459C7A13DC3B497B88F441243DD3D68BB9D3CF846762E0641404408ACF3C264382B3E953E2C3EB3C16533FB4546B474BD26C42D41D7B7A3C0FD40ACB6AC3589B8F8C56AC5C53E074146C1FB3FF53CEB3EB2C132C34E432FBC6A3AFD3FF938"> : tensor<5x6x7xf16>
    return %0 : tensor<5x6x7xf16>
  }
}

