// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xbf16>
    %1 = call @expected() : () -> tensor<20x30xbf16>
    %2 = stablehlo.multiply %0, %0 : tensor<20x30xbf16>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<bf16>) -> tensor<20x30xbf16>
    %5 = stablehlo.divide %4, %2 : tensor<20x30xbf16>
    %6 = stablehlo.custom_call @check.eq(%5, %1) : (tensor<20x30xbf16>, tensor<20x30xbf16>) -> tensor<i1>
    return %6 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xbf16> {
    %0 = stablehlo.constant dense<"0xA13FD2BF4EC0A9BF3B409B3F8ABF94BE213F16C0CAC06C3F50402A401D3FE1C0B2BDCBC0E9BF2040AE3E4FBD1BC005C03840193F2CBFD9BF2B4005C07540234003401AC09740743F12C031C08A3F10BFA7405AC01840ADC08C3F883F95400F4005C0A24073C040C05A408B40AC3E5F4089C0CB3F89BF0DC0064095BF6EC085BFA13F8B406E3F96C0C3BF3DC071BF01C0B040333F3740D8BF66409AC0D13F8FC04DC0B73FF4C0D73E5AC097C0283FA93F8FC089BE46C0C8409BBF89C0DE3FBBBF87408C408F40F33F07C0963E4BC0404017C03340BFBEADC0ABBEC2BF0B3F9C40EB3FCF40E2BE73BFB3BE5ABF434083C07BBD6740C74000C02FC003BF0540B63E1CC041C0223F2D3EEC3E014073C011C0BABFCBC0BDBEC9C0D73F24C03F3F01BD2AC0DFBE1340D63F73C058C085401BC012BF4140AEBF223FFEBF373E10BF983FA54027C07540FABED13FDBBF14C0ABBF2AC088BF1ABF1DBFB0BFC63FA6401B4088C0BC3FA3C060C08AC09E406A3FB4C0204021C04A40D7BE343F29C019400F40F93EABBFA040C6BEF6C0A2C0873F9AC04AC089BFA7BE3040BA4070408540CC3F423F13C0D0BFDFC07FC0EEBF2AC03DC0AB40283FE74039BF8E3FC0BEA4BE384012C00B3FD53EF1BFF140AB3E0440293F4B3E2640F74064BF99BE7AC049C09CC05CC03BBFD4BF55C0354066BF72C0D23FE7C0D640183F2FC08DC016C08FC022BF71C0FF3E90BF024002C1813FF93F8740F8BF4B40A2BF6CBF2240CC3F53C08D40D93F753F4A3E0FC088400F40413F6B40C6BFDA3E9A4037409CBFEF3D883D243F47C0A140513FF7C060BF00406040843FA93D944032BD4540BC3F8EC010404B40F6BFABBE30BFADBEA340D33FA8BF824072C054C03DC08F3CEC3E913FBA3F9AC00540214003BF16BFBEBFC5BFE7BF80BF853F8A3E7BC0ACBF8FBF093F494020BF4A3F304018C1A2C020C0E13F0140614018BFDA3FBF3FD5BF8F409D3E43BF0B4088BFBAC0914088BF56C07C3C14C07C4001BE17404CC03C3E2B407EC0154026C047409BC047C027BFBDB7763F1DC008413C4006C0EFC03DC0C2BF04C0344044C0A9C0B54041BF19407D4067C0823FDEBFE1BF9BC0E43D80401740BBBFA6BEE7BF04401C401E408C3F0BBED8BF3840A23F0940C5BFF3BF723FF23E28C0A73FADC07840463F6B40173F5340133F94C085402AC07B3F923F98BF19C0463FB1BF99BE9B40BB3F8340AD3F42C09240DB3FF33F27402740A54004C0CF3F92406B40CEBFE13F78C0863BFEBFE33F31BF2040673FC2BFA2BF1D3FC7BE574011403A4077C08A40BE40B24041C0ED3FA6BF83C05EC05CC037C09FC0B23D0B40B240D13F0BC0AF3EA9C0134056BFF1BFF33E59BE2CBF6440303F724092BF4640C2BF913F844086407D40B2C039C0843F15400E4085C011C0B1BF93408DBFA13F434049BFF03FCEC07C40F63DABC05AC084C0294092BFB040AB4025C0963F843FF7BE91C0543F2140E3C0BCBE7B3FBEBF4B4088C0E1BE49C0D73F64403BC0CD3FD8BDA7BF093FA43E693F9BC02D403A40F93E2A3DD9BF463F8B3D89BF9FBE31405FBF0B40353FE8BF804054C072405CC0D83F98C090C025409AC0833E55C0D0BDDFBF53C049C092C02DC0E440D23E0FC011409740B5C098BFA03CD1C0CFC0E4BF21C0B03F9E3F254000C0AB3E87C0854089BF8ABFC63F"> : tensor<20x30xbf16>
    return %0 : tensor<20x30xbf16>
  }
  func.func private @expected() -> tensor<20x30xbf16> {
    %0 = stablehlo.constant dense<"0x213FBF3EC53D133FEF3D2E3F5C3F404121403A3ECE3C963FC23D113E2A40A53C0443CC3C9B3E243E0A41C4432E3E6D3EF83D33400E40B23E103E6D3E8C3D1E3E753E313E383D8D3F443E063E5C3F4A40163DB03D363E0C3D563F643F3D3D4D3E6D3E203D8E3DE43DB03D593D0E41A93D5F3DCC3E5F3F533E6A3E3D3F943D6D3F213F593D943F3A3DDC3EEA3D903F7C3E073D0340FA3DB43E9E3D313DC03E4D3DC83DFA3E8D3CB540B03D383D1540133F4D3D5F41D63DD23C2E3F5F3DAA3EEF3E673D563D4D3D8E3E673E3A41CC3DE43D383E033EE5400C3D1041DF3E59402C3D983EC43CA4408E3F0341B03FDC3D753D85439E3DD33C803E093E75406D3EFE402C3EE03D20400C4296407C3E8E3D483EF33ECC3CEA40CF3CB53E1C3EE53F7C44113EA940423EB73E8E3DB43D6D3D2E3E4440E03D0A3F2040823EFA414A40363F1A3D163E8C3D8640C03EAF3E403E103F113E643F31402A40073FD63E183D2E3E643DED3E1E3DA73D5C3D283D993F023D243E213ECE3DB5400240133E333E4D3E8740103F243DD6408B3C203D673F313DCE3D5F3F1641073EF33C923D6D3DC93EDF3F423EC23EA93C813D943E113EEA3D103D15409E3CF53F4F3FE4401C41F83D443E5940B940903E903C1041713E1340CC41183E8A3CA13F3341863DCF3D2C3DAD3DEF3FBA3EB93D003E9E3F8F3DBF3E9E3CB73C3640093E533D3A3E4D3D2040903D81404A3F783E783C7C3F873E673D893ECC3D203F963F203EC93EBC3D533DB23E8C3FCE414D3E643D4D3EE03F983DD63EB040313DFA3D2C3F934264431C40D33D213DC03F8A3CA73F803EA73D713F1343403D0444D83DED3E4F3D4A3ECC3D8B3E104107400C411E3DBC3E153F783D8F3DBA3DEA3D4D459640483FF33E313D6D3E213E75403A40E83ED83E9E3E803F6D3F5C41853D0E3F4D3F5F40CF3D2440CE3F073E363C203D243EA53E7C3EA53D3640B03EE53EB93E4D3D2A41DC3F593E643FF33C483D643FB73D8445403E843D7C42383EC93DED41103E823D3D3E183ED33D2E3DD33D1640EA4E8B3F2A3E643CED3D6A3E933CEA3DDF3E713E023EDA3D133D003DE03F333E833D9E3D783FAA3EA53E2E3DA142803D383EEF3E18419E3E713E2C3E283E563F5942B43EF83D203F5F3ED83E8E3E8F3F8F40153E163F0C3D893DD63F983D3840BC3D4240403D6D3D113E853F443F363F333ED63F063F33412E3DEF3E753D0C3FDF3D443DAF3E8E3E163E163E1A3D713EC43E443D983DC53EA53E893D6A47823EA33E0640243E9E3FDF3E203F2A40D340B53D483EF33D8A3D5C3DE83C043DE03D963E183F753DAA3DAD3DFA3D253D0443593E043DC03E593E0941133D423EB73F903E8E40B2410E40A13D07408F3D443FD63DDF3E483F713D6A3D833D043DF53D713F3D3E4F3E6D3D483E063F423D533F213FDC3DCF3F923EC53C843D8B42103DB03D713D133E443F073D103D1A3E3A3F713F8A40483DBA3F213EA33CED40853FE83ECC3D643DA540CF3DB53EA13DEF3DC83EB442163F5F401C419B3F2E3D0C3EF33D87401144B23ED63F59435F3F2541063EA93F593E00409C3E803DBA3D8F3DAD3DB43E363D4A3D1A3E313D7541B93DC242A93EBC3DCF3D443D0C3EA13CBF404D3E483E383D003D363F2445C03CC43CA13E213E073F283F1A3E803E1041673D6D3D5F3F5C3FD63E"> : tensor<20x30xbf16>
    return %0 : tensor<20x30xbf16>
  }
}
