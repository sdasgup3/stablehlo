// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>)
    %2 = call @expected() : () -> tensor<3x5x40xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xf16>, tensor<2x1xi32>, tensor<3x5x2xf16>) -> tensor<3x5x40xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xf16>, tensor<3x5x40xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>) {
    %0 = stablehlo.constant dense<"0xAFC11F3F044100C279B63CBDF94086B887C07B3E33C583450D4682C2A0394FB9263EC53FDFC633BC113FA4411FC2C5B6B62C71BF6EBAA1C0BD3F82C22DC26EB92FC524C05D3C54BFC13759B6AF3764C01DC1ED4324BABABFE0C35E3775411538F03B6940E0A9C1C43545693435415DBF3B3E12C501BFCAC55E3C57C0F9C53EBDCE3DB044A13F67C5F94072BA5645D03EAFC28F42C0BCEB3D8A42F93DE3AFCC40B2BA543A73BD19C533444BBD343C81C04A3C30C04637743546C1E1BA0BB9B2BEF5BEE63DDCBCBE445939B74786461B3A963EBB3C57B065B5D0BF2EC026C1D5B3FE4185BB2C48953C25332B4040C5CA3FDF3D1EBE9439CF42AEC0F43E4FBEA132914685B3A5C2353E72319B41184257AF2538CCB6CBC4133F11BC2CBDC74136461D41B3BE66C5B232CFADE44297B8313EE02C223E41BA6EBD0330B13F66C0DFBC0BC13FC196C225BE183FCE402539353ABBC4EA41933E1B3F7B3E9837D8425839A7C454BBC0AE00C6983840414D41D4B73BB604C40AC26F41B3423440E53BE0C31444154829B8BFC3B3B5BA427D44B4BC4AC07D3D40ACA3414937F5276B405AC32DC1B239B342E040BEC009C10840D6A859C26D3B65426E42DC298EBE0DC2AA304244AABF55C237C4A540C1C54DC581AC3A3FD744A1314EBE9DB0A1C4A5C249B86BBCF0C0ED43D344D9BF89C1D03F3943C84647C285B94FB870C4074319412FB7A53A5641D9B958C075C17EB2EF39AFB77730CDC0144644C25645FFB6C9B87F425E445B3F2CB84E3CF13D90C0933F86B6AF3B373966C0B9BEF8BC58417B3F9342393DA3425EC06E40304440BCAEC11AB894BFD540EAC6824463B9CC2F843E1945DAC465390A440D3C4C436D3D14C0819A66BCFBC2B8BF58426ABB453F9F3CD6ADA3448A3BC02DDD45F9321ABE713C6941E03FEBC145BF76C34D44963D7AC01A404D437C3C2F3BC3C5053C3C471B3A86B7183B78461E4301B04A2C1AC3C4C69ABBE5C05B43643D953FD345A3B985AD0BBFE04270B5C9BF5A4111445FBA653E2645E3B8B32F02C5FF408B4382C49FC443BA1EC0673FD3C10BBC20C021439EBD20C6793FB7B2E93BCEC4B73CB946B336F53D46C6F69064396EC24CC7C6441A41B8B72DC25C3DACC4CEB5BE3FA731ABC29F43ACC474BCEF39C03603BFA941D2384543523F78BC63BFF13935B56EC1F9BCF23D7FC0CCC1733978288D41E5BCDC42C0444D413A3FAFAB4240DC44A3452EC000458DC6D5ACC7BAD9C31638CAC2163EF63CBF4353BBB538FE40B5C2ECC59A45A942D340533DBBBD814004BBC23F6DB6C73C04B485C2053AD8447843153C98BF47C11B3A42C4BB3A40B89FBE03427C3EE1BA12BC42413F46DC40CDC1AF41FA3D65C5B94690BF5832E5B109C7E84422432A42973D7CBD4D41A1BE90427CB6AE4515402A36D8BD4641F84482402339A6C41ABB53B678C4144430BB59C05138F6B527441B3E4FC6B1BF78B08A4052B9C5C68141DC3EFFB836C0E63B0EC1F13149C169C4EB295A436B416D35A4422F4525413CC48DC65D440B41F0BD393EE7B73026134157447045FABEF92E123BCBB86C468F43CDC2DDB47A3E1F41E4C3FEC5043C684656BEFAC1563DF9BDE8402B3301C7CB4069BC092E9043DEB513C5B2C4EA3E3B44E53893C11DAA3840B2C314C2D3BC803C23B5613E12C4D94484C09740D142BCC4"> : tensor<3x5x40xf16>
    %1 = stablehlo.constant dense<[[[1.479490e+00, 3.388670e+00], [-2.470700e+00, 5.585940e-01], [-1.370120e+00, 5.097660e+00], [-3.861330e+00, -1.189450e+00], [4.515630e+00, 3.525390e-01]], [[5.083010e-01, -4.769530e+00], [-1.918950e+00, -2.056640e+00], [-2.794920e+00, 3.371090e+00], [2.884770e+00, 3.164060e+00], [-6.805420e-02, 4.449220e+00]], [[2.933590e+00, 1.962890e+00], [-4.667970e+00, 2.113280e+00], [1.066410e+00, -2.115230e+00], [5.781250e+00, 2.263670e+00], [-5.786130e-01, -2.300780e+00]]]> : tensor<3x5x2xf16>
    return %0, %1 : tensor<3x5x40xf16>, tensor<3x5x2xf16>
  }
  func.func private @expected() -> tensor<3x5x40xf16> {
    %0 = stablehlo.constant dense<"0xAFC1C742044100C279B63CBDF94086B887C07B3E33C583450D4682C2A0394FB9263EC53FDFC633BC113FA4411FC2C5B6B62C71BF6EBAA1C0BD3F82C22DC26EB92FC524C05D3C54BFC13759B6AF3764C01DC1ED4324BABABFE0C35E3775411538F03B6940E0A9C1C43545693435415DBF3B3E12C501BFCAC55E3C57C0F9C53EBDCE3DB044A13F67C5F94072BA5645D03EAFC28F42C0BCEB3D8A42F93DE3AFCC40B2BA194573BD19C533444BBD343C81C04A3C30C04637743546C1E1BA0BB9B2BEF5BEE63DDCBCBE445939B74786461B3A963EBB3C57B065B5D0BF2EC026C1D5B3FE4185BB2C48953C25332B4040C5CA3FDF3DC2BC9439CF42AEC0F43E4FBEA132914685B3A5C2353E72319B41184257AF2538CCB6CBC4133F11BC2CBDC74136461D41B3BE66C5B232CFADE44297B8313EE02C223E41BA6EBD0330B13F66C0DFBC0BC1844496C225BE183FCE402539353ABBC4EA41933E1B3F7B3E9837D8425839A7C454BBC0AE00C6983840414D41D4B73BB604C40AC26F41B3423440E53BE0C31444154829B8BFC3B3B5BA427D44B4BC4AC07D3D40ACA3414937F5276B405AC32DC1B239B342E040BEC009C10840D6A859C26D3B65426E42DC298EBE0DC2AA304244AABF55C237C4A540C1C54DC581AC3A3FD744A1314EBE9DB0A1C4A5C249B86BBCADBFED43D344D9BF89C1D03F3943C84647C285B94FB870C4074319412FB7A53A5641D9B958C075C17EB2EF39AFB77730CDC0144644C25645FFB6C9B87F425E445B3F2CB84E3CF13D90C0933F86B6AF3BBE4266C0B9BEF8BC58417B3F9342393DA3425EC06E40304440BCAEC11AB894BFD540EAC6824463B9CC2F843E1945DAC465390A440D3C4C436D3D14C0819A66BCFBC2B8BF58426ABB453F9F3CD6ADA3445442C02DDD45F9321ABE713C6941E03FEBC145BF76C34D44963D7AC01A404D437C3C2F3BC3C5053C3C471B3A86B7183B78461E4301B04A2C1AC3C4C69ABBE5C05B43643D953FD345A3B985AD0BBFE0427344C9BF5A4111445FBA653E2645E3B8B32F02C5FF408B4382C49FC443BA1EC0673FD3C10BBC20C021439EBD20C6793FB7B2E93BCEC4B73CB946B336F53D46C6F69064396EC24CC7C6441A41B8B72DC2DE41ACC4CEB5BE3FA731ABC29F43ACC474BCEF39C03603BFA941D2384543523F78BC63BFF13935B56EC1F9BCF23D7FC0CCC1733978288D41E5BCDC42C0444D413A3FAFAB4240DC44A3452EC000458DC63A40C7BAD9C31638CAC2163EF63CBF4353BBB538FE40B5C2ECC59A45A942D340533DBBBD814004BBC23F6DB6C73C04B485C2053AD8447843153C98BF47C11B3A42C4BB3A40B89FBE03427C3EE1BA12BC42413F46DC40CDC1AF41FA3D65C5B94690BF5832E5B109C7E84422432A42973D7CBD4D41A1BE90427CB6AE4515402A36D8BD4641F84482402339A6C41ABB53B678C4144430BB59C05138F6B527441B3EC845B1BF78B08A4052B9C5C68141DC3EFFB836C0E63B0EC1F13149C169C4EB295A436B416D35A4422F4525413CC48DC65D440B41F0BD393EE7B73026134157447045FABEF92E123BCBB86C468F43CDC2DDB47A3E1F41E4C3FEC5043C684656BEFAC1563DF9BDE8402B3301C7CB4069BC092E9043DEB513C5B2C4EA3E3B44E53893C11DAA3840B2C314C2D3BC803C23B5613E12C4D94484C09740D142BCC4"> : tensor<3x5x40xf16>
    return %0 : tensor<3x5x40xf16>
  }
}

