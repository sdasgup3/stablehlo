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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf16>, tensor<2x2x2xi32>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>) {
    %0 = stablehlo.constant dense<"0xADBC2D405229653DF0432A366BC3D7B597332BC300C062B8F741F0B974BE24C48BBC4BB7B8B259C57546CD3EFCC4D543523D72C585385EC04C3CC33724BAE6B89AB5B8BBEDC3724252C171C4693F481D76C5FE3DDCC3FE42F33C6E457B3B75B65EBC36BBBBB6DEAE2EBDC5C32744ED3025C543420F3B973EEB396CBFF54646BF3732FDBFB8C4FA3E6441723FF2B3D841A93C2BC474C05EC12F44B93ED9440F409F43B1442A3CDE3E0B41FFBFA83C80BC18BE40BD74BF9B4583BD6A38D5BB9638BEC4FEC311C43D3C64C234C1EF3EBFC1BEBA4FC4B0C5EA34E6C0D3462E3A23C6C7380B405D3A4C412AC2E9418AC35A3C3A3D372FD6BDBD45333DBC345A4024B772C022C65FB8733D40402B3CB1BFFAC21E3EF341AEC285B475424544663EFD41D0C70E437339603C0AC6953EA9C11BC2C4C21CA846C013C41CB8693DF824B9B65B398B2823C3A7C40B47744458BC49C171446742AD3FE240C3C41944B6B80EBB5443953A5D3D66C1C7478BB7723E11BCA540D6B0FF3FD1BDB4BDBCB94FC00348F1C2CE33B345D0C45EC51641222E6C3DCC41E63B8DB71EC0A4C0134529408B4014436045"> : tensor<5x6x7xf16>
    %1 = stablehlo.constant dense<[[[-9.565420e-01, 4.976560e+00], [-2.486330e+00, -4.078130e+00]], [[1.174800e+00, 2.197270e+00], [-1.819340e+00, -2.488280e+00]], [[-3.632810e+00, -2.675780e-01], [-1.559570e+00, 1.933590e+00]], [[-7.104490e-01, 2.781250e+00], [-7.548830e-01, 4.304690e+00]], [[1.765140e-01, -1.285160e+00], [1.727540e+00, 1.448240e+00]]]> : tensor<5x2x2xf16>
    return %0, %1 : tensor<5x6x7xf16>, tensor<5x2x2xf16>
  }
  func.func private @expected() -> tensor<5x6x7xf16> {
    %0 = stablehlo.constant dense<"0xADBC863C5229653DF0432A366BC3D7B59733AAC700C062B8F741F0B974BE24C48BBC8544B8B259C57546CD3EFCC4D543523D72C585385EC0A6BDC33724BAE6B89AB5B8BBEDC3724252C171C4693F481D76C5FE3DDCC3AC44F33C6E457B3B75B65EBC36BBBBB631C12EBDC5C32744ED3025C543420F3BB043EB396CBFF54646BF3732FDBFB8C4FA3E6441723F23C0D841A93C2BC474C05EC12F44B93ED9440F409F43B1442A3CDE3E0B41A2C5A83C80BC18BE40BD74BF9B4583BDF840D5BB9638BEC4FEC311C43D3C64C2BDC1EF3EBFC1BEBA4FC4B0C5EA34E6C0D3462E3A23C6B3BB0B405D3A4C412AC2E9418AC35A3C3A3D372FD6BDBD45333DBC345A40A0BC72C022C65FB8733D40402B3CB1BF883A1E3EF341AEC285B475424544663EC645D0C70E437339603C0AC6953EA9C11BC2C4C21CA8C8C113C41CB8693DF824B9B65B398B2823C3A7C40B47744458BC49C17144C142AD3FE240C3C41944B6B80EBB54438B405D3D66C1C7478BB7723E11BCA540BFBDFF3FD1BDB4BDBCB94FC00348F1C2CE33B345D0C448C31641222E6C3DCC41E63B8DB71EC0A4C0134529408B4014436045"> : tensor<5x6x7xf16>
    return %0 : tensor<5x6x7xf16>
  }
}

