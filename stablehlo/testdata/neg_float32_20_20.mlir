// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.negate %0 : tensor<20x20xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %2 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x26492DC0DF2CDABEE13F3F401573603F37C8F4C05EE24840560291403FF2193F8744EB3E7BF00041B77B9140E08E02C0809183C0C37D5E3F0F604740E77D83404DA53FC0107AC5BF25725D4067205340ED2B05408E8EE03F8B4EE43F3378B7BEAE1E35BFA35701400A43593F6AB79E402CE0D9C0F387213F3934E7BFAFB187C08D3F5BBE3473C3BFBB8B5AC09D1B70BFB027A23ED5D912BF7E8862BEC420BB3FF0C7E7BF040C93BEAB4597BDD069813F9638D2BFE9B7783F7C5D6040350264BF19F6C8BE14692940766A14404FD0C2C0B0C655C0B0ED763F62B179BF3185173E4E913640A45F7E40E53E0FC09DDC943FF962BC3F62F60E406CCC7DBF40B7CDC0DCC33B402C37E1BFEE1369BD57EA71BF5FD668BF0D93BC4037D73E405BCE7E40EAD7C23F516195BDBA45ADC084F3783DAD84BDBF66F5554029A595406DE3ED3EB8AC0040D2808BBDE80F823F5D695F3F676D28C098F8353FF70E8D405270143F02031840E1DC05BF8D0102C074D105C09C7C2EC09DA4DB3FDA14C3C02CEEE63F0A381ABFCD7733C0A995C5BE86A7C1BD77F226404B2B7DC0F51AD2C0E284ACBF476B8AC0681888402656883F7A5A1740E9E2E5BFD6DDE1BF89893A400A64E7BF88C6F4BFBC31C8BF8A1DA3BE432DA4407CB3294055AD92BF26D85F4000A1AD3F76953CC02306C3BF0E37164094DCE0BF7C4B64C0E8A2283FB19587BE1AF4024028091AC0B3D5D8C0D403BB40B5B82A40E986CCBE7C879FC05CEC014014FC2340E4DC22BF4E9567C0C6C666C0FF16E0BE22E4F83E5FC65E3FA235A63FD6C437C175378140D6320AC05766C93FB74389407BC3D9BD42DB1CC0A49B30404B9F37C02DC98D40AB5CA93F54181640EDDE373FE9A723BF7B32A33FA984DCBF5ECB82C0A323BDC014C3D6402E30B43F591797C0ABDC99BF066D953E308A863EB3A66EC0CF78B94019CAA4409A1820BE47D6D3BFA94D24C0DE4B3B40577D3C3FF1D397BF63D874BF77CF0CC0401571BD607CB53EC6381F3FD7C62FC01A456E3FDC24B93F919C3040A9BC59C0F602154040C34040D8562FC018E2ABBE8FD705C01D33393FB7DAB1C0A8455BBF4842374094BDE8BF71417540D7F133402EECBCBF3F2E92C086D82540170ED3BF95D8944080B53A40BD619D3F04BE2A4019C930C09ABE713FC99A3F403BF7A240AB5068BF4BE578C0B4A41E405B868FC0C4184ABFB48037BF94EBAEBF9BF902401FC129BE0F398F3E92A830404007A2404895AEBF6BF9DE3F493B1AC0B395BBBF80CEB540DC52DABE0A3E7FC0048880C0FE378D40555592C0A88A48402896EB3FCE7D7C40C1CBBC40293F03BF3ECBF33EE9302840AA226F3C31E4C23F5B6D8EC0C48A15BF58E926BD9258AABD82CA39408840334065B7B53FF77AD0BFEEB690BF85B6E73FC5B891C052F8AB3FEE258040928448BF2DC667C03E9CC1BE7026FE3F038C3B40177CCEBF74C137BFA85FC03F4C624A4074091340AD7A4BBF7E0998BEB155F23EC774BD3FA7DD1AC090344DC067C40CC005E38340E4D61C4031186F404EFB3BBF1E73E4BEA3053EC0BBB9D93F45E794C0CF7C2140065436408A458A3FE9324640BD655340AD967440408EA5408DE88240A547A8BF8A1A07C06F05E8BFCB156E40A72D743F224271BF2DA30E3F60C5EB3FF98551C0EF230FC003462A3F2EC9BFC0C4B7734011A75B40C274B7BE8A99AA401FD3DBC085431340B7304BBF866A74C05E03DEBCF1C5FAC096C914C0DFB65CC00ACEC7404D70DBC0CE8E61BF2D07DBBF7DD69DC0FCF1D4BE522D67C093052AC0EC1C063F08EAD7BFF94D654033DB8940FAF7BD3FF87A2DC083C28540F10185BF49DB2740AE62E5C0DD57B3BD8391A140BFAAD0C098E2B7BFB2761B40550FB0BEC9A3E43F46D4133F95272C40AD94613FF4BF87409E177FBF4FAEC63F2DF7D53F001086C0D38BF4BF9C6253C0FD9D4C4020AEABC00E5ECB40B54740C0A01B803F98DFFC3E87C064C00C09EE3D27BB4CBF002DDF3E3737A8BFF9A456C09F1B8EC0CF20833F164A45C0625837C0917A66C059B5214003E028406144BB40B78780BF3BF54AC0CD4EA83F93BB3A4056F50F40981D0E3F618E32C02DFA9BC0B0310041CB7DE73F1C147CC0FA737B3F09D94C40D45AE5BF054F494022820DC01373DEBF6743E13FF19EE6BFB74E07C05C4358C0ABC1CEBDA0C4B1402957A3BF9F9E56BF0E205F409ED93E41D2C30DBE903B06BE331758BF712100BF43A81EC0073B63BFC504583F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x26492D40DF2CDA3EE13F3FC0157360BF37C8F4405EE248C0560291C03FF219BF8744EBBE7BF000C1B77B91C0E08E024080918340C37D5EBF0F6047C0E77D83C04DA53F40107AC53F25725DC0672053C0ED2B05C08E8EE0BF8B4EE4BF3378B73EAE1E353FA35701C00A4359BF6AB79EC02CE0D940F38721BF3934E73FAFB187408D3F5B3E3473C33FBB8B5A409D1B703FB027A2BED5D9123F7E88623EC420BBBFF0C7E73F040C933EAB45973DD06981BF9638D23FE9B778BF7C5D60C03502643F19F6C83E146929C0766A14C04FD0C240B0C65540B0ED76BF62B1793F318517BE4E9136C0A45F7EC0E53E0F409DDC94BFF962BCBF62F60EC06CCC7D3F40B7CD40DCC33BC02C37E13FEE13693D57EA713F5FD6683F0D93BCC037D73EC05BCE7EC0EAD7C2BF5161953DBA45AD4084F378BDAD84BD3F66F555C029A595C06DE3EDBEB8AC00C0D2808B3DE80F82BF5D695FBF676D284098F835BFF70E8DC0527014BF020318C0E1DC053F8D01024074D105409C7C2E409DA4DBBFDA14C3402CEEE6BF0A381A3FCD773340A995C53E86A7C13D77F226C04B2B7D40F51AD240E284AC3F476B8A40681888C0265688BF7A5A17C0E9E2E53FD6DDE13F89893AC00A64E73F88C6F43FBC31C83F8A1DA33E432DA4C07CB329C055AD923F26D85FC000A1ADBF76953C402306C33F0E3716C094DCE03F7C4B6440E8A228BFB195873E1AF402C028091A40B3D5D840D403BBC0B5B82AC0E986CC3E7C879F405CEC01C014FC23C0E4DC223F4E956740C6C66640FF16E03E22E4F8BE5FC65EBFA235A6BFD6C43741753781C0D6320A405766C9BFB74389C07BC3D93D42DB1C40A49B30C04B9F37402DC98DC0AB5CA9BF541816C0EDDE37BFE9A7233F7B32A3BFA984DC3F5ECB8240A323BD4014C3D6C02E30B4BF59179740ABDC993F066D95BE308A86BEB3A66E40CF78B9C019CAA4C09A18203E47D6D33FA94D2440DE4B3BC0577D3CBFF1D3973F63D8743F77CF0C404015713D607CB5BEC6381FBFD7C62F401A456EBFDC24B9BF919C30C0A9BC5940F60215C040C340C0D8562F4018E2AB3E8FD705401D3339BFB7DAB140A8455B3F484237C094BDE83F714175C0D7F133C02EECBC3F3F2E924086D825C0170ED33F95D894C080B53AC0BD619DBF04BE2AC019C930409ABE71BFC99A3FC03BF7A2C0AB50683F4BE57840B4A41EC05B868F40C4184A3FB480373F94EBAE3F9BF902C01FC1293E0F398FBE92A830C04007A2C04895AE3F6BF9DEBF493B1A40B395BB3F80CEB5C0DC52DA3E0A3E7F4004888040FE378DC055559240A88A48C02896EBBFCE7D7CC0C1CBBCC0293F033F3ECBF3BEE93028C0AA226FBC31E4C2BF5B6D8E40C48A153F58E9263D9258AA3D82CA39C0884033C065B7B5BFF77AD03FEEB6903F85B6E7BFC5B8914052F8ABBFEE2580C09284483F2DC667403E9CC13E7026FEBF038C3BC0177CCE3F74C1373FA85FC0BF4C624AC0740913C0AD7A4B3F7E09983EB155F2BEC774BDBFA7DD1A4090344D4067C40C4005E383C0E4D61CC031186FC04EFB3B3F1E73E43EA3053E40BBB9D9BF45E79440CF7C21C0065436C08A458ABFE93246C0BD6553C0AD9674C0408EA5C08DE882C0A547A83F8A1A07406F05E83FCB156EC0A72D74BF2242713F2DA30EBF60C5EBBFF9855140EF230F4003462ABF2EC9BF40C4B773C011A75BC0C274B73E8A99AAC01FD3DB40854313C0B7304B3F866A74405E03DE3CF1C5FA4096C91440DFB65C400ACEC7C04D70DB40CE8E613F2D07DB3F7DD69D40FCF1D43E522D674093052A40EC1C06BF08EAD73FF94D65C033DB89C0FAF7BDBFF87A2D4083C285C0F101853F49DB27C0AE62E540DD57B33D8391A1C0BFAAD04098E2B73FB2761BC0550FB03EC9A3E4BF46D413BF95272CC0AD9461BFF4BF87C09E177F3F4FAEC6BF2DF7D5BF00108640D38BF43F9C625340FD9D4CC020AEAB400E5ECBC0B5474040A01B80BF98DFFCBE87C064400C09EEBD27BB4C3F002DDFBE3737A83FF9A456409F1B8E40CF2083BF164A454062583740917A664059B521C003E028C06144BBC0B787803F3BF54A40CD4EA8BF93BB3AC056F50FC0981D0EBF618E32402DFA9B40B03100C1CB7DE7BF1C147C40FA737BBF09D94CC0D45AE53F054F49C022820D401373DE3F6743E1BFF19EE63FB74E07405C435840ABC1CE3DA0C4B1C02957A33F9F9E563F0E205FC09ED93EC1D2C30D3E903B063E3317583F7121003F43A81E40073B633FC50458BF"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
}