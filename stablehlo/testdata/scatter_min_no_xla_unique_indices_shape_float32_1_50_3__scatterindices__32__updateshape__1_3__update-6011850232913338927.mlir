// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xf32>, tensor<1x3xf32>)
    %2 = call @expected() : () -> tensor<1x50x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf32>, tensor<1xi32>, tensor<1x3xf32>) -> tensor<1x50x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf32>, tensor<1x50x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf32>, tensor<1x3xf32>) {
    %0 = stablehlo.constant dense<"0x26FE9E3E2F069DC05BE09340490BC4403CCDD63F82F09BBFC943803FFDF6B4BF53C6BC408DAC5DC04E857D40BE7D283FE751B7C096BC023EAC111EC1746DA03F29C22AC04157DD3F746C01BF43C59B3F02A3343FD0E10641FB5725C02CC6B2C00224673FBBFF65406D8715C0D6992EC0D60416BFEEF6D5BFBDF5B840FA941FC090098E3F961EC5BF6C942B4094F08CC0853028BE183D3DC02DE26F40AC0A93C03E2E9D3FFC5CB8BF39B5353F5C9E503FB43409BE77C49DBFFBDCE7409D6E25C0D570AD3FC56BD6BF59786DC069DCABBE038181BE39A38CBFA1768FBFF9B18A3FB3BF1C4090D63E40C60581C093D5E2BEE42E81BF5E14A43FAF1A84C001AA8E3FB005244018119440D1DF8DC0029AE93F2EB2B5C03C1EAB3EE81004C089531DC0F0D8D43FB5ACB8C0EA147F3EAA47A140CD645DBF8CB90CBFC1E468409B34C8C05C68C83FF5C12CBFDB685CBF9FEA8CC05657844079E15B40DC4206C011CA5E3F4C662D4020BB2F4035415740AF5917C06EE69A3F4710B0403A37A240F643D0BFB6878E40BA49C2C0CB82CEBFFF8E11406186A93ED5B6AABF9447443FF31833C0941AB6C08C3249401D92C2BEA3C050C0BB4981C0761CDEBFF404AB403EDFF5BF7CDFA4BD2240973FB83608C03E48FAC082B32A406D481C3F3FC6D7C0392BE9BF508E43400A5782400398EABD1F1F463F4CF958C023FF19C06316B43EF8462D403800A1BF2BF031C0230A32401E80A4C0420622C0C17AB13F20B0D4BE01A6C63E739391401D799F3F4A2A644017FBAFBFDFD9B6C053585D404A2D68BF159458C0E89F6040FDDA973F5BAEE0BFF3F2813F39712C40CCA2F0BF"> : tensor<1x50x3xf32>
    %1 = stablehlo.constant dense<[[3.13880539, -0.327942282, 4.19906425]]> : tensor<1x3xf32>
    return %0, %1 : tensor<1x50x3xf32>, tensor<1x3xf32>
  }
  func.func private @expected() -> tensor<1x50x3xf32> {
    %0 = stablehlo.constant dense<"0x26FE9E3E2F069DC05BE09340490BC4403CCDD63F82F09BBFC943803FFDF6B4BF53C6BC408DAC5DC04E857D40BE7D283FE751B7C096BC023EAC111EC1746DA03F29C22AC04157DD3F746C01BF43C59B3F02A3343FD0E10641FB5725C02CC6B2C00224673FBBFF65406D8715C0D6992EC0D60416BFEEF6D5BFBDF5B840FA941FC090098E3F961EC5BF6C942B4094F08CC0853028BE183D3DC02DE26F40AC0A93C03E2E9D3FFC5CB8BF39B5353F5C9E503FB43409BE77C49DBFFBDCE7409D6E25C0D570AD3FC56BD6BF59786DC069DCABBE038181BE39A38CBFA1768FBFF9B18A3FB3BF1C4090D63E40C60581C093D5E2BEE42E81BF5E14A43FAF1A84C001AA8E3FB005244018119440D1DF8DC0029AE93F2EB2B5C03C1EAB3EE81004C089531DC0F0D8D43FB5ACB8C0EA147F3EAA47A140CD645DBF8CB90CBFC1E468409B34C8C05C68C83FF5C12CBFDB685CBF9FEA8CC05657844079E15B40DC4206C011CA5E3F4C662D4020BB2F4035415740AF5917C06EE69A3F4710B0403A37A240F643D0BF30E24840BA49C2C0CB82CEBFFF8E11406186A93ED5B6AABF9447443FF31833C0941AB6C08C3249401D92C2BEA3C050C0BB4981C0761CDEBFF404AB403EDFF5BF7CDFA4BD2240973FB83608C03E48FAC082B32A406D481C3F3FC6D7C0392BE9BF508E43400A5782400398EABD1F1F463F4CF958C023FF19C06316B43EF8462D403800A1BF2BF031C0230A32401E80A4C0420622C0C17AB13F20B0D4BE01A6C63E739391401D799F3F4A2A644017FBAFBFDFD9B6C053585D404A2D68BF159458C0E89F6040FDDA973F5BAEE0BFF3F2813F39712C40CCA2F0BF"> : tensor<1x50x3xf32>
    return %0 : tensor<1x50x3xf32>
  }
}

