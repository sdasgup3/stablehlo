// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>)
    %2 = call @expected() : () -> tensor<5x6x7xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf32>, tensor<2x2x1xi32>, tensor<5x2x2x7xf32>) -> tensor<5x6x7xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>) {
    %0 = stablehlo.constant dense<"0xEF36373EF1929A40A72B81C0713735C022FD38C0E5101D3E26BE4540FF75ABBF6295F9BFDF674940C3C313C04981454054869E3F8FEA38405425D9BFF8EBE3BEF1CD8F3E29D7573FE238BE3F34DD88C0CF65914048874340EF7A7B3E1F3E70BFB27D06406CD828C0D8BE95408B8193C0A5B3DBBE4AA082408D1A51BD604F2B3F3718F4BE259EC53DA3A7F93FD256A5BDA6343EBF5C3E36C06E6147BED5B90EBD67C5F4BF85FE23403AE74E3E818A363E23FAB3C0661FBFC00BC58ABE707B213FCA92B0C02575803FFE3323409E422BC0A1CBA140F25024BFC08EBEC05611B73E5D1FE5BF355607C04E4ADA3E4403413F7B71E2BF3DAC5D403693B43FBE8269C05A7F2C4014ED1F3FC28CA0C02FA513413B8772C0109A15C031378B3FB30B96409A7B4C3E633086C04121463EACF49740EC6C93BEF705BE3EFE87D0BF74ACB7C02BB248BF6CD9D7BD5E55D0BE5A5BA63F7E30CC3FACA07FC0CDD988BF864561402CA2AB3D35EC863EA751353F87E761C01CB68F40B02B18403B594240E3E89AC03B8D493E39A9B840BB23FDBF964D17C0F6227240FC2FA63AA13321C018DAA43FD5093340616FB2BF5915D340CB88B23F8405634094BB0DC0B1901A40F9CBD7BFDC9F2E3ECE03D6BFF83296BE926B9BBFCE194C3F15F93D40A707313FEC2066C0B7458C3FBF1C7B40DC9A6F408EA6EB401EBA0B40A6E97DBFBDF301404E21D9BE36E860401CC5D9BEBB54E73EEA95A6BCA0673B408FA00B40683FDDC06A8795C0B9D00BC040A31D400BACB83F3BDE0BBDA7070FBFF08C9F3F005142C04A11FABFBBF33FC0506A653FD75A0D3F2D17C63FC64FD7C02AF2AFC069FCA840FC8F683E5B2783C0E383E2BE4FA700C049B1D94044B54BBF7F5E4DC053C292C02BEC5B400A48C03F143B0ABFE3E58140EF2F4040BB0B86C061819E3E8CDB0340012A41BFD29D8FBE8E2D1DBF8F360BBFC28C56400DEB99C03DE13740685F19406A3537C05B4FB94027ACFF3F8E0E5C3CBE84F3C02AA38C3D61FA09BEC68C6240EAB0C5BFCA920DC04BFE0F40AD96B1BFDCBB5040CDE570C0898C9EBF130F383F0FDB6D3FB201B8BF7717E9BF73F998408F4C4BBE3BB096406F5826401A29E13D270C5FC061DFDC3FEC3CAFBF969586C04F87FBBF51DF24408F5B3CC0F18DE93F14D80B409B46A540D4E2CC40"> : tensor<5x6x7xf32>
    %1 = stablehlo.constant dense<"0x8C2E09C07CE3E93F43D48CC04B94FCBC6D5649C0422A27BDDE72CC3F09AF163F3D581C400F8588C0A18393BF6A74B1403F5F85401C0D613F548084406C321BC00EEE6D3F0C2FF1BEA9B3003F596AD0BF063E1240021E76407DB358BEED10C9BF00500E41B8C4B3BFC8E4B9BF6F3FD4BE668F3BC0AE01864029E6B53F6533B23F694EDABDC833A0BFAB04A0BF2A9F4A3EA66877C0AABF0BBEAB3080BF9F7E153FA06B8F40BE08A9BF59BA0BC0E41109C00E12673F68C3AFBF0BD2D8BFD6ED75C0F31E6F40C3FFB64032F41741D4FECA3F726BF23F06C27F40B0D1B5BFABEE32400C9748402935F03E71C862C04EA9C43D6A26A6C02AEF2D409F340FBF1A5BD2BF6A25B53FE9BE08C02D7B9DC00225DC3F53540BBFEA930EC0281B8CC02C6CC13FE153DAC02B2EAB40235055C0567F82BDD34201C0D1245E3F8DFA51BF6FAD4ABD323E9F3F74C5B73FAB44E5BD1DA26EC05B502E3F92BE8F40E9AB1CC0C48084406199F4BE966E16BFC80745402911F43FEC0B11C0C025F7BE44EDE13EC9AB263FFF67D53EBB1FA2403773E53FC0A97040FC6C99C02BFA7540C28790402BC193404D3582BFBCE01C3FF45993C0CE518640A24B4B40D4CC01BF400E173F6B5015BE51EC20C0134985BF9379914060E11EBF699F0E40613AF13F458E8B40C8EB0BBE01C0173FBEBE6A3FE67B4CC0706D923FE5D405BFAACAF53F90BF5FC0DAE13BBFD4948A402180D2BF6276894046D8AC40E485D6BF4BB49340A722B83F0FB8043E244136BF3CE5B9C079AFDCBEF1976940"> : tensor<5x2x2x7xf32>
    return %0, %1 : tensor<5x6x7xf32>, tensor<5x2x2x7xf32>
  }
  func.func private @expected() -> tensor<5x6x7xf32> {
    %0 = stablehlo.constant dense<"0x8C2E09C07CE3E93F43D48CC0713735C06D5649C0422A27BDDE72CC3FFF75ABBF6295F9BF0F8588C0C3C313C04981454054869E3F1C0D613F5425D9BF6C321BC0F1CD8F3E0C2FF1BEA9B3003F34DD88C0063E1240488743407DB358BEED10C9BFB27D06406CD828C0C8E4B9BF8B8193C0A5B3DBBE4AA082408D1A51BD604F2B3F3718F4BE259EC53DA3A7F93FD256A5BDA6343EBF5C3E36C06E6147BED5B90EBD67C5F4BF85FE2340668F3BC0818A363E23FAB3C0661FBFC00BC58ABEC833A0BFCA92B0C02A9F4A3EA66877C09E422BC0AB3080BFF25024BFC08EBEC0BE08A9BF59BA0BC0E41109C04E4ADA3E68C3AFBF7B71E2BFD6ED75C03693B43FBE8269C05A7F2C4014ED1F3FC28CA0C006C27F403B8772C0109A15C031378B3FB30B96409A7B4C3E633086C04121463EACF49740EC6C93BEF705BE3EFE87D0BF74ACB7C02BB248BF6CD9D7BD5E55D0BE5A5BA63F7E30CC3FACA07FC071C862C04EA9C43D6A26A6C035EC863E9F340FBF87E761C06A25B53FE9BE08C02D7B9DC0E3E89AC053540BBFEA930EC0281B8CC0964D17C0E153DAC0FC2FA63A235055C0567F82BDD34201C0616FB2BF8DFA51BF6FAD4ABD323E9F3F94BB0DC0AB44E5BD1DA26EC0DC9F2E3ECE03D6BFF83296BE926B9BBFCE194C3F15F93D40A707313FEC2066C0B7458C3FBF1C7B40DC9A6F408EA6EB401EBA0B40A6E97DBF5B502E3F4E21D9BEE9AB1CC01CC5D9BE6199F4BE966E16BFA0673B402911F43F683FDDC06A8795C0B9D00BC0C9AB263FFF67D53E3BDE0BBDA7070FBFF08C9F3FFC6C99C04A11FABFBBF33FC0506A653F4D3582BFBCE01C3FC64FD7C02AF2AFC0A24B4B40D4CC01BF5B2783C0E383E2BE4FA700C049B1D94044B54BBF7F5E4DC053C292C02BEC5B400A48C03F143B0ABFE3E58140EF2F4040BB0B86C061819E3E8CDB0340012A41BF51EC20C0134985BF8F360BBF60E11EBF0DEB99C0613AF13F685F19406A3537C001C0173FBEBE6A3FE67B4CC0BE84F3C0E5D405BF61FA09BE90BF5FC0EAB0C5BFCA920DC02180D2BFAD96B1BFDCBB5040CDE570C0898C9EBF130F383F0FB8043EB201B8BF3CE5B9C079AFDCBE8F4C4BBE3BB096406F5826401A29E13D270C5FC061DFDC3FEC3CAFBF969586C04F87FBBF51DF24408F5B3CC0F18DE93F14D80B409B46A540D4E2CC40"> : tensor<5x6x7xf32>
    return %0 : tensor<5x6x7xf32>
  }
}

