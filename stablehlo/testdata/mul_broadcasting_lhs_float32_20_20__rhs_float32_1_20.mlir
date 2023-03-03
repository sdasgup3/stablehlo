// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1] : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.multiply %0#0, %2 : tensor<20x20xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf32>, tensor<1x20xf32>) {
    %0 = stablehlo.constant dense<"0x718DE640D4A06740A41805C0CF6F02C027F60F3F3A798CBF777D82C08C8CDFBFB31684C002642E3FD77104BEEC823CBF0B84D3BEB32C72BF18B99DC0E3A68D40A704A7BF449DC7C062338E407BDCA73F1651BFBE487F18C0C2260D404B078AC0FEFF1CC0A90D5CC02A2921BF6765C73C0E4080BE213D95C0E7402D3F55A36EBF5025D8C09A4E09408D590240781E343FDC5BACC00E122D40476DDEC0FD7B483F0747CDBF51C722BF02953440C6384BBF6B0C4EBF06574ABED085B8C08AC30EC0388AD43D8B0433C0B8DD97BF671E3B3E6DA2CE3F5DA187BF849494BF72C42CBF1E268AC05880834090C93040A823324021747A40E42B6FC07E74A9BF8385D73FBDFD8A409AEEDE403DA4454025AB774021DB313F84725140DB84833F656809416548E43EDE5726C003C39B409DE9913F46BFBCBFF4B9DCC038C0CDBF739D354081F0263F605020BE69FDD1409BBC703FA03C2B3FCB6814C05A81CCC05A7EB8BFB0DA8CC030FDE83F84094AC0F3FB5BBF37A64F40341CDEBFAD609BBFBB7C14C0A7F917BF31CDDBBF7F6318403D4EDABF4AF9AC4090A2C7BFB09438C074C754C0C0DA3140EB87A6404E9191C09C7C3E401741D43F3F096DC0028A003CC64852BC4469D1C04C730F408E372FC06E04CF3F651E893F869D513F8E0028BFEBB005C00EEC07BFD8FB673F1E06F23F5569B6BE051E0040E73D9C40E431713E76E5AD406769063E646587C005202AC0721FB24041E73F40F9DAAD3F68D39CC0CB5F3E40D26CB640EA242EC00B6EED3F502CF63F630B883FD4E83E40D3A10BBFBAD276BF136EC9C0F66E89BF6A6C2740E72B2AC0819C82C04E6A25BF1FCEA8BF52B8C43F0B0E87BEA6E08CBFC6349EBFBB2A894026FDD53F196A38C0464E0C40B14E6040E4B78B40E3A8814008A3BDC002B271BDF939A0BF1C140E412DEF084016EB0440F2463D3F1141F1BF74076A402072E53F3F3385C0195489C07C523840CE7D9F406C5456C064E32FC0F8E24CBF1FBCBCC094405E4078F2F1BE8CA0C7BF9871403F65A53A40DD32763FB373EC3ECE53FB3C7020D5BF18F03540700DBEBEEA07114017C51040F3B164BFD07F22BF7E3480BFA48A45C0AD949A3FA78A86BEB78584C02F193BBDBAFB8F3EADA36C40AF6C0B40A0FE4F3E9316EF3F5622B64059D606C0BD626EC06BDE9ABF4F4298BFEBF772BFF425993E260A71C0DB37F83F88E7164004E1803EC798A1BF27A4283D76A275BF8D457EC0094A5A403BD115400239BFBEDC9E4EBE591532C0267A8F40485581BFFCA65CC0D2C816C016AFBCC0E5C915C0DBED7AC0174AD23E77AA1FBF647B833E26B0C9C0A384CF3E0B0D42BD1F4369BF6B2C4D40BDBD0040299CA0C0166DABBF1CC7514086E86F40827FF7BF93147AC040BDBF3FE1AE963FD23E2DC0AE7ADF3F90E50040A790793F3392A3BFD803D73FEA8EBFBFCACE6740EF3A164011D41E403A5A9ABEDA181E40BC73A03F16DB073FE7B73E3F4C1595BFCB6D8B3F08F0ABC0B16ADFC06FBC62C0C0F464C0A93A923F10CE99BF0B56AE3F1FF42E40362E74406F6FB53F852B47BF1CAC4EBFAF41D23F75AC07C081891240CF0F6CC079B49640D2FA8140DC1B62C0DCDB9EBFBFC207C0DF9B78C09104AA403A5B82BF37F13E40609780404EC78D3FBFD5A5403857AA408C8688405ABBEABFB2E5944020549AC0CE7770C0EB220E400F9FB73FB5C3FDBEC6851EBE7A3301BFA309E33F026C5B3EA1EEF73FC27C2B3EA67BAEBFD50CB640D3A7F43F819C00BFC7B3B73FA889533F11265F3F82917E40D401D33FD50CC43FC98D8BC0326048BE8B8322C0686E79BEA17447C0CF04163FDC0E4940FE8332C0C5A0DA3FEB38573F7553A63FAE2C19BF10C83140CAD8DBBF7E832B40D3A60EBE02903F3DA0CB243FCE628E40D0E8223E77A0B73F3CA663BD753D0241ECCC2AC0E19C833DF3CE74BF24873B3F324EE4BE1F1A99409D88C1C00E9A4FBE5BD26540AAAC8B3F34A6543E84935B3F73E122C13064FCBF654E24407E337ABFD53D0B40D3C877BFF21085BE510F07C05D49B4404F09C5BF8717B3C06CBF5CC0D55DFD3FE2A6863FFF4369C0DE796BC062E74D402B8383BF7A771BC09D261C40F25A8F401B58DF40DAD3904089D7C83F2FFB523F9AFD6E3F6D52EC3F9E9291406D9147C05E23B6BF073E484095C33EC0BB212C4017B282BDA0DE5C40221BB3408082DCBE0589E9BEDE65AD3FB6F6DEBDE41B2DBF924C42BF743709BEC4A200C0B1341CC0"> : tensor<20x20xf32>
    %1 = stablehlo.constant dense<[[-0.105458081, -2.77254343, 2.15986228, -0.595729291, 2.00408602, 8.59326171, 2.7623651, 3.69609976, 2.03827119, -2.85471582, 3.62267494, -3.30146885, -2.41679049, 1.53583598, 1.08981121, 2.85020018, -1.50110793, 3.37812281, -2.0805912, -0.245028853]]> : tensor<1x20xf32>
    return %0, %1 : tensor<20x20xf32>, tensor<1x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x4E8242BFC28C20C11BBC8FC0F5689B3F7241903F07E416C10C3B34C19B90CEC0C39D06C1FAEAF8BFEBE6EFBE55971B403D987F3F61F8B9BF69E3ABC03EDE49415AB6FA3F9694A8C147EE13C1FE85A4BE3C68213D0D67D3400D6F98408A7424401B529DC03C5FECC1AD97DEBF3A3FB83D51B402BF6504554109E91C40C8F64440419882418FE15240830E0E4004580040565D0141D3291241C36367416F7F44BE722F2D3EC4A7E13F2504C3405D21F23E2F78CEBF5258D9BF0BDC7EC1D6EA03C1649B583EBC85FF40568A89C013711ABF35B279C0524ED0BF9FECA1BFF635F6BF5C60CF40251D5E413EE9B7C0D9982EBF834CD3BE3FC72541F2FF36C07A6480BF6F460B41D4766F42527D084116DA64416542B53F687A15C1A9396E40E3D2E2C18CED89BFD6797FC03CC0A940A3F04F4039AA0D400369BAC1AD0A5640F20032BF47D78CBD1C3DDE3E4DC66241F7690FBF2F96AB3F406A9FC1C53A8DC1067AAAC0B18C0FC18447A6C08FFA36C15791354022ECFAC0F98F2AC01155A9BFF29BD3C09521643F0FA1B9C07D879EC0F7F6D53E8FEE11BFCB5F8A40A655C7C07B84FD3FC537B24067E13242300E49C1B7033041DD505840EE2A2941E3D3E83CCE8F2D3D2E0D7D41F6505C4016F43EC09C8293407CD4CDBFCE0631409CC5AE3F5F08033F6758653DD8CB20C040AF8240F055593E076180400ED427422C91263F3FAFA041D9FB883E36424141B4131AC1490493C124E5E7C0B481054018E9AAC0B1A607417CEB08C1F31113C14BFF76C02E47F1BE458DE5BD825304C107CB96BF2F0A133F6DD749C10DA013C10E3EE740223E1DC1551C05C1551BEC3FAEE198C0C25DA2C01C33233F625DD8BF326AACBF017A4341359C20C060BE1BC19FF591C0DDD85BBF36C0EBBE5CBE33C173CB4CC129FC0F3DC68D20C0629D98429921BD40C1A3F54029E6C03F932DAC40EEF353418660BDC05EF5204100EAD2C05CE048408F4A6341B6DDA040048B14C18324D53F60FBB83FA281BBBEC6B3A73F649557C0EA49E5BE0307BB40653A0441AA4AA33F963BE83D7C3459C061D801C1E61FACBF5368EFC072F0AEC07D9EAFBFF11731BF7CB436C0FF439440578C82408AF60B3F1EE3813F29D99D3B999947BFE18DFF405B1EA6BF696BD03EC26880417B8F7B41A12FF9C085F2F2C0A10D5D406FE589C0C68948404A1039BF4319B9C067410740CC0DD7401476C1BE347988C0CD6FAFBD12C0703E2185D63ED84D17C1D9CAA1407BD5633EED0ACFBE264ABFC10A2B46419A036FC0E5DFE0C03039D74098E2AAC1E342F7405C9C17412B7C213F73012EBF00603B3FB76017416341AF3FCFDEC93DB69F643EFF18ADBE6178B2C09F722DC1433F4C3FD334D24084D9004286EBAAC09B1467C187684340291457C027E71CC1D273B8C019C29BC03FA5BF3FF942B2BF7E3599405AC60F40CFC44341A9489CC08FAB1BBFC238023D5F2ADBC0F7462D40D7DDA1BEA61BBF3F8C2320C1A5934040E9DF9EC125B163C117D12141BC5B4FC1AF6271C06EDB394037E0054099AA3E40A7FD2D414E2D08C0873428C01300D73F6213CEBE18ED643EE923CBC031EEFEC0098F33C0CD3E024165E0F2C1BD695BC081E4FAC0BC5DFDC022AD72C18E1E6CC0F4981DC19E631BC1A3BFD93F91BAB440A7C072418BF0CCC0FE3CC6C0ACE51AC18642973FD5DFCA3E280AC5C0634C4640C82C973EAFD89EBE47C88AC04CCA9C4056C04A3F2DAD7C4018C6F4BE18069EC01A4296C100D293C0A38645BF6633C83F31BB1640327CA7BFA6FD564182825BC0D926C0BE2A79EB3E3BE30A3FF680AFC0F197143EF2DCC7C0D224A14054D90A41EFF324C1C4CF5E40839919C0E2A29640B8D9FC3F8FD4D6C008D328C0E1EA3A40FB4ACBBE2CC78FBDBF2C0B409D1F14C188AB1FBD61EB1ABEBCCA1D3E7AA68C416F80CB3FB7E1033E3E7B03C150810140BEF5D2BF20081C41F31E8A41B8043CBFF3AF3DC12FC828C0134CA33EF14B6F3F0A1FE8C1EF6E3D40FCC20A413E240240087908BF080CD13D4577383FF6DA91C0D2CD56C05C7045C07A5F40C24C7218C1F31DEA40813A09400D7A2641674355C13FF229C13FEB1E4078C56EC0C82C2A40AC4B4C41C1A127C1649F74415BEF50C032C94EBEC1A0C9BDA1CDA3C061351D41EFC6ED3FA18236C08117D7416DBD03C1D50D1F41553205BE48A11DC1F035A2416400B63FF7198D3FC727054005FDF2BD94B2F6BFFBD4913F6DC4E7BEBBD185408819193F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
