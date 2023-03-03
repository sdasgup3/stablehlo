// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x6x9x10xf32>, tensor<6x3x4x5xf32>)
    %1 = call @expected() : () -> tensor<2x6x6x6xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 2 : i64} : (tensor<2x6x9x10xf32>, tensor<6x3x4x5xf32>) -> tensor<2x6x6x6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x6x6x6xf32>, tensor<2x6x6x6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x6x9x10xf32>, tensor<6x3x4x5xf32>) {
    %0 = stablehlo.constant dense<"0x7A3D83BF98D6D3BF4F8C89C01289BCBF394B434025BEA43F018070402E791040C71BB3BF96309B4031299240B152EFBF44610BC005C563BFC3A900C1387D89BFC0C49E3FE42E8CBFA41DF0BEBC8AA3C0096F84BD9843CDBF4BD50AC0BABF4D3FB01984403A0EDF40FE1D113F1AD9D43EA1EAEEC007C7A13FA60B8B3FF50C32BF238E9FBF075D3EBF713FBA408A7F2CC01A50D7BF9D701ABFDC246B3F706DD43F4ACEBF3FA1882D3F6ADB02C0E7F5E73E0F124EBFEB69F2BFF20BF9BFE7D6EC3F1B960240AEFB63BF0583C83A6F60DF3EDC743C408486B93F5E4FAC3F77833740E93C7FC0648DFABEEFB5163FA4CA64C0A1F96940F846893F8D1103C03FDA1CC0733D8CBFA76DBD3FDD58C540D8057B405247D9BEE6DD95C0F5C1EDBF55B9A9BF8D17F640F2EECE3F0CA092409B2003C08E69AA4084CB4FBFD17A6BBF601CAABF1E047BC031F9C9C0BE694DC0FF4EE03FEC2D7140EBEA183F143E90BF7D0E2A4045D9CC3D83290940DE35EA3F1A2C94C06DFBA33EC215003F2AF7B13F79070B40508E5940FC0E58C01852E23F63F5A0BFF382D64008C7BE3FBB7DDE3F127A97404F67A6BFB85D24C076B4BBC0243B3B40526424BF9CE8B9BFE45E5C4031C5C43F88660D3FB5DC6BBF9D630DC06AE3A1BFFD7714C0EC64BC3EC3FC423F79AEE0BF84C8E4405DC693C012BD05C08D839040983D3D40080A0CC072E130C0FB178AC0C6B60440B47E7E402B7471C08E4500C0EF330A40BC8167408A4437C07C673CBF8A5F0A3F32779A40939467C07C1538C0A43D14C08DBE9C3D341C103F422F34BF4D11CF3FD8BEA53FC903123F695BF13FB2ECF6BDE265C23FFD02B140292B91BF4BDB6C3F4010F43F4D5BA73F2B9341407A7A193FB544593F8A9401BF97C56540F137224077F7E2405FA50F3F055E334046D484C0D5A994BFAB477EC005F4093F72DAC3C0381C07C066380E4074081EBE7EF168C0F8E0003EC8D8B1BE9CFC0BC0982030BF8FA28640E4E550C006768D40E2FB2040B6D582C0928831C06596B0405FCE03C1837650BFCE5389BF32953F40C5072C40497A8340B8A2ACBF2A21BD3FD217564008549A40E27227C061978CBF87338640CC122EC0C02B0E40B286EC3EF56B82C0FD391AC09C646EBF64527AC0B2A9F7BF71FAD23F2FA659BF8FE2B53E4D97A8C0815CC3C0B7025B3FCEBC76C0621AA1C040C1A3C02EB8053F71E4103F359F273F996634409B109F3FEB9EBABF4C4B8840C85128BF235589C05906ADC003FF6140EE2FF33E2F7F033FC224A4BF2708AFC0A0750BC0CD460440CB81A53FA7A1ED3F497E0DC0E5222CC01306974038B056404D53A53FC152363FC9861040B83321408F691EC04C8B7B404B5FD13F4B28F33F3B7F5CBE526FE9BFD72F01C059A0D1C00CB42EBF142D3CC0E16554C0EBC0A1BF48483A40F938B0C07263DD4041EC9C407A875C408D82EDBF5457A9401EEBC13FEF266DC07CC3FFBDAD4487C011EA2C404360B4BFCD5BD03FA3B4F6401B49A04034C448C0578A7E3F049BB83FC9B9193F65A7793F8E7CE03F9B998D40F30A813F198107402A07C9BF3CFA1440EA2071BF4B529CC0DB9237404D527F4075327EBFB6648A408CC963C0B5118740A952AD3FA5899DC0D1ECA4C0593486C0DD657EC0A33323C0FA9F68402A27DA3F09CBF6BE158C9B3F6B593AC0B4C5A73FE1B006C068C762C06F8F97BF75CB1F4021047540A06B124042D452C0E3FC4FC0745F7E40D408DC3F0F6331C071C8BDBF658D2BC046529D3F53C249BE19548FC0AEFCFCBF0CBE3940277B004124C893BFBDFD5CC0CED0A93EEE9E3D3E2B1B9EC064850FC07BA61440808A6F3FAF740DC01B7DF93F6DB708404DA0643F188C03401E2B0FC0D14F5740002DBCBDA01C34BE3D7360C0FAE83C3F888ACC4012636FBF90E57AC02CA9DBBFE24D94BE1458ADC03168853EEB0D9FBF706DB73F4B17B1C08EFBE33F5F8F8ABE5EF50CC09890CE40BE9080BF7BB6AEBE17378CBFAF7C6EC0F97C8E3F85CB2C403FAD40C0A19027C0152BB6C0E8257B3FDBCDBABFDD0E8EBF05244BBF41B82840C9EC68C0B18A593F1C1E4B3EFC042440B8F788C0F043D43E52C57D3FEDBF81C0D72CA1BFC3A9E73E373E7A407B4E55C08370893F5EAD85C0C601C5BD5CB0C93F4D105B4076E7C5BFCC78F4BEF8DE7EBF05F60BC0AE531F40237B0FBCC44943C014E2C640EB4E0040717CA5BFB4A38540D64424400EDFF2BFE41CA33C24D10540CE0D173FD78D95C055C25D3F50FD3C4044A0ABC09927B53EBFED4E40C39CA63F2946A4C00FF0B63FEEC96C404FA87CBF1D5C0D40201CDD3DD7B23B3F3FE82A3F035B98C07F49A0BFD2E458C02A1E223F08FAA23F295600C088EC454014B40BC05072A4BF63F622C05C45ADBF2F0A03BF9FC6C33F8EA9833FB6D8B3C0E88D2BBEE40B31402B11EC3F818B42408F3DA440B8049FC0FF253BC0DC8F54BF59FFA0C0A4E38FBEAF096BBF48065BBFA14786407C6AAEC0C51A373FF939204034E04F403D76543F09EC98BFE9F19E3F97A8F43FD308C03E4762D73FB68B3C4080A2AABEACB8A0402B0D2AC017051B3F2C2E81BF3B5A63BF9B1135C024F112C0821626404F2351BF8C408540011651C02EBE90C0A4401BC01D0B71BFA0204F3F8287B540080C734088E897BF612C28C090004A40E0A6C0C07FF7253F14D888409DA687C031AD203F19FF12C0A660204067118DC0762EBAC0DAC42FC07F5616400FD7BF3C517C04C08F64F93F6C8CB840C432373F2DE9E0BE00565ABF0874913E87B671C04B228740182CEDC0724EB9BFBE3B29BF0F275C3F8692B43FB6CC01409DF9EF4037470A40C86ABEBF69AE8CC07C5DE8BFA67994BFB050F1BF6134823ED9AF16C095E457C07FE872403A82F9BF21F6A83F5BC72740137132C0381A07C0E4FF3740B5DAE23E199303406B046E3F091C21BE1432C6C08F2459BFA55C193FC951BAC076A3E43FB58C8FBFB62228BEEF8A0A40FEE3C5C03FE44CC0E558F03EC34C613E6ADF174004776540046773C03071BEBF56A793C085E687C07E251CC09704693E368FE1BEF235FB3EB10697C0AEA087407E1415C08F9DA13F179EBD406CF0044066BD7C3C68785FC03738DE3D8AC7D1BFB323BA3FDD7A933FE895A3BF80DA2C40AE1E294020938F406C06B1C0F34AA34014B2A73F761D0040474FB2BFE9F872404D4431C0F3FFCDBE05A115BF49CC82BF4963D8BF2B2E1640D272B43EF81203BF3D99C8BF1B7335C0A0CCBCBFDA06483FFC6809C04BE07E3F22E448BF1099A63FA630CEBF1A8900C0EE0D02C0CB6CA13EE14286C01232703F2F7224C0290858BD1EFE8740A31126C0511044BCF14D8040771EFE3E1E83DCBF14DB03BFCD974B40B47992BF8A21A840C0723AC0F9CE89BF200C96C04C636B3F10C8C8BFFECAD13FF77E16401F654F4017DD88BFAF5FC2BF45251840D0BE6B403ADDACC015C46A3EAE5FB2C0E281A1BEC590EF3FB0D4BF3E3CB98A40237761BF2B024BC043D701C1DDDB48C04C8525C0EFBFA8BF571187405E0B87C09AF05A3E62CA5DBF38B687BFD6ED05408974F6BF9C51993FB3C06FC04E4B9CC0E1DC0AC0313F503FFE510640A6091340EDAE743F609644401411B94069014A404D51A53F2E4D5F3F2580C93FA8804740920A3E40279F5C408650833F927124C076C06DBFCF733C3F779DB74028AA23C0ABCC074003BEBD3F2052013FEFFC05C041888EC05AB70940F8DCF13C33F51940C74483403345B6BF494E283E0D09D9BF712D5640B1F6CCBEC64E7ABE8D570CC0E76D15C0DA84EC3F7B588FC0D3BE00C068D6883F5F70ED3D49A1163F8B064040191462BF1C2A7CC006669BC0F687913EDF0866C04B2BB4BF72FD4AC0984E07402AD88040441332C0A6C479404EA8783F139EA9C003364EBFBD5347C0D9DBC1BF71D5D53FF23F02C0688C3640E1DC39BF1EA66840DFD0804015D8523FC793FCC0A7C244BF01B0D33FEE7480BF1E2C8D405637A33FBF4B263FB7529040F893B540ABB10A3F54557E3F8E166D3F26122BC08043D6BFE5C77140CFA70140C535B4C008A22340365F52C0D83A4240FE09033E3A34AEBF092DF9BF845B373E4ADC5CC06FC19F40863ABBBE87B53DC075A67C3FB8139140E32611409ED2D6BF342D473E5705A5405D25BABFA53576C09ED9FD3FF3F89E3FEDE55BBE00349C4090A214BFE88E6D3F4B971FBEF804AD3E7F6B9D404E7BCC4065E3EE3FC91D04C02A317A401F59BFBF0165523F756C6CC06C7747C00A8AD23F17D9834003EA8A40E9E852C0AA338BBDE8C88540241E9BC0807679408267FABFF36E1BC0CEB1E0BFEE88043E46CB8AC09F0211C0CA66C73FAFA8B2BFDA76C0BFCA973DC023E7AA3F31B249409D93C4BFBE22BA405002B8BFA32EE6BFBDD5D2BFA90E5CBFA1C802C0229F274021ED5BBFB032A0409C068A3F5876A7C05063C93F5FE560C0109E2040A2E3A73E46E7E3BF78D7483E995F5FBF5057C63F868224C07092ACC08700E7BFA6D49DC0BC4C66C09A0FC8C0BE2A71402184C63F0F693FC0292648C01C96CAC000D2F23DDF02653F2FE52AC0C0BE3CC0BBDC9DBFDC88283ED8CC3D40A198B44081D32B3F93F103C0B33C57C030D09ABF445E4EC0AAA75940420358C0399EC4BFAB9843C09E86EC40264BECBEF37F913E045329C0D48927BF3DAC7A3FDC545A405A34AFBF348B2B40BED566C094ECD6BD63D201C0BCDCC8BF6CEFB6BD65792D40029B0940A0D3A83F703BC1BFBA6E1540F8E6194045499FBFF0A82DC0A8E172BFB268084073C22DBED591CEBE3BA084C029737EBF234D33C0D1E95ABF164C07C0A32FEE404CC0D43FEE6D8A400A220CC03F3824C0A46B5C3C15938440DA0384403C13713F9895853DF0192E40ADD31D3F19300AC0901B1D3FE854993F191A11BFF9DEA9C05BF511C0ECF70DC08FCCDE40C5BD34405186E23F8292F4BF96C2BB40617F883F20C091C0653F17C01CEDAEBFF25CC33F8C1E18C0783835BFC33183C007619ABE709922BFFD5D8EC0E4CDF63FFECF3B405CD23B3EB244ECBF04DCBB3E1821ACC02D947E4095F4B3408CE2A0BEA2DFDCBF2E6F11C09FCC93402A2ABC3D9510184097B0613F30E8D13FFAC6943FECC869BFC310E63FDE263D403522373F869681C0E8C7F5BE9EFD48C01F3A7240AC757B3F78A03AC0645948C0E81C7FBFC373ABC03C821F4037358DC0E576F3BE4F9AC8BF3C13E4BD0FE5B0BEBA96DCBFAE9EE13F6408603FDBBCD8C0C44643C0357EC33F02D59DC0C889F3BDFB2293C049D625C08CC3BE3F4B7EF4BF4E04263F39CEF63F07969D3F931A2AC093FB4BBD81783DBC76D9353F3EC91CBEC7E88D40E6733940220ECBC0107BB740733F564038C6FABFE17914C0A118C53F2AECE0BEC84A85401C1425BFCF1036BF6DC44DC0FAC2733EC3E3603E0062A8C024D09440AC20DCBF5DF926BFADEB2240BBCC28408B4528C0DA124640D2342F3F2CD200C1E42A0B40F87210C0A28431BD0C6BABC0C696E04048210D40EE566DBF71342BBF561ED04027F704BF58D21B3F127A12C09C02433E0AF36DC0E16BED3FA2E917C06581C6BFFC34983FC95C99C0912D0F4073C8733FD667C440FD394CBF1FC53A3F3A59A0BF26ADCF3F7E40733F6A029B3DC9CA5EC08B4D8D3F2FEA863F71F7DABDD3DD58409C01113F9174B5BEBB8824C01A586B40F8829740EB923D3F282C9B4036E48FBF3DDA0BC029B883BED929B0C033EDD0BF9CF63FC0468A2440CA20C7C08322243F1BDDCBC02AA9113FFCF43BC05B1615400C9724BE4C8A47400F806BBE4008D93FF5BD4F3DB195EFBF300513404A92204051D50BC0B54118C0C82AA3C0723901C03D7817C01AAD4240029350C0EB2F63C02FC0A23FC6124D3F062265403F9593C0EF648C408B03863E484D9540347816403E79B53F1109BFC0CD2E4640DA33F2BA15E605BFC4B92F3E97FED23F0E51993EB14AB23E3FE730C0D5B4B13FBF8B933F21F69DBFE8D837C030BEE13F94D17DBFACB083BFBF3412C012B593C02586FA3FB09FD6BF59CE20BFF3C919BFB81B67C0442A4540EADA503EA59C4E4056CE08BDE9E8D6BF57E3ED3EA3F0893F335B7C3F202BABC0"> : tensor<2x6x9x10xf32>
    %1 = stablehlo.constant dense<"0x62A77440BA00E03FFEFA2DBF6B822FBEF12BD63F553E8BC0C821B13FA984D2BE8E962EC05C5E8AC0156E9DBF010B6DBF842599BF10D80BC040EC84BF03B638C01B4699C04B7116BF2A8D55BF9961863FC596FEC0A2EAA1BFEB38333D80C83BC0C4012240B67406C034FABFBED15461BF74D091C0AB6823C01CFB513FFFE89D40048FCFBFF7DDD340657DC33FB49FAE3F72D01EBED52706C0B9D95C40B5171740BD44BEBF231635C0D1F49E40C04EB73E737B5DC0674924C09D82D7BFC8CE533FEB2F723F16C4D7C069DA1F4084EB97BFCDCB67BC8738C9BEDF6ED43FC8D57AC0CF0C97C033DCB8C0DAAA203F2F598A405A391C4098679C3FAC2E81404F43F43FD559DBBE1554953F53273C4052D4A43FDAF0893F2753913F957CF6BF51F095C00636D2BF6584513F816F0EBE623F20C04F2E6D3EFB50A43DC41EF83F9B5559408F4C59C0809728BD8A9E7A3F112E2AC0BCF48FC089B992BFC9A68D40AD9C1B4031EA65BFD24CB9C093689FC0B669E1BEC0AF57BEABD94DBFEF0D8140F109314011D681C064A72FC054FABAC00BCF08C0CB2F7D3F11221840E83432409B41413ED74977404A071F40D0B48D3FA02F16C0F27E9C40D9A7C14098F62240873C00C0AEF48340B3E11B3FAF7410C1176BA53FA78FA13F9E0FB0BF01B42640A57B6CC07A0C6340A374A440BD3F62C0F892BE40C0DAA2409D734ABE35CC8FC04CC4B2BFC1DD6540AE6FA03EBD7D6BC009EC33C06B2BC63F19A70AC0D52A67C0D711B2BE057A29406DF3743ED36FA8C081D33240597E24408C49B5BFBF31704094BFCA3FF8555F40B343923F895B823F2CF6ADBBB3FBF7BD5DB5153F18957E401A6EC9C04A362E40D16A8EC0392564BE30D757C06F678F40604982BECBF92B403444A33FB3384AC0AE9E56BFED3B2D3FE2BFC83F0470ECBE706054C0393F65C0EE534EC07BDFD4BFD1C9C43F8707623F5196833E7E98BB3D44E3DABFE5EEA8BFD73FF4BE664CCEBF08E14A40F48F1DC0CE799740F177C2BFB4BA5140E86EF23FFE75FFBF9E4F2F3F748DFA40599BB640623BCFBF2E470CBF170F2E400158A4BF0C7DF5BF397E3B40A16310400A3011C06A66C33FC00936C09AC586BE3DC83FC0ED3A8240D1BD81BF08AA5FC0745EC6C013175FC0A8383B40321E983F1036623EF5488C3F1F970140C777483F9AC80D3F88F8B2C04CBD55C05B745B4052C013402B8D3FBE01657ABF872A3A40D93D5B4069215B40720A5EC0B3D182BF921CFDBF814B6740FD0C75409A9EA8C0E36133C0608827BD00464D40FFD82B40D1FC203F644F4240DC243F3ED03407405B4E0C4012785FC04888BD3F038B354013A070BFAABEDDC07830ADBFA120BD3F51910E4007346ABE91F58D40D15D92BFA90BE13F24D665BC95743DC05C1EA7BF6C5E3F40BBDB17C01859583F9356C5BFA08870BF1D46D23ECB1ABBBEF82C67BFAF909ABD279CC93F3C872B40DFD6DCBFDC3480C066E87BC054A541BEAD963240879C75C0824D26406ACF203F119C5040D7C484BFFA7E1EC0F22387C054AFEF3F342460BF933313C0C2732FC02F1641407E0F1BC0383D47C0C0DA9EBF2A7C3D3F82442CBF1C9B0E404E6234C024E325C07ECE2A3F59F47EBFBE0958C009A9DB3EFF8983404A12FDBFEBE922C0E4BD393FAE720C40B826BE3F2B9BF73F4E1E5640FD3EA1BF634A09BF7615D0C01651B9C0B202DF3FFFBF11C02CF6C6405474D1BF45F80B402D51F1BF0A3C58C0DA0FC23E8BC4E0403ED5B2402E08E73E4057D5C0608E36C0D2893EC01AF142C079B6BDC0DEF3CC3DEB3CFD3E245010409E8FBB3F384DE9BE278ADA3E6E45EB3FA5D50D405BAEC2BF8117A94030EC0D40E34ECFBFE3D68F3F535A8DC0F875BDBF322301C00C7C503FE3D0BBC0F8C2C6BE3F0527BF75DEF5BFB06072BF5D8C8540E3C6274046961140119FB23F552686C0BAF6CD3E2EFC84BF36B349BF3B470F40C1ACCA3FEF4B9E4016310B41A6F23EBF874E0640CA32AABEFE660940922F83C0C77814C0961083C0FAE978C0"> : tensor<6x3x4x5xf32>
    return %0, %1 : tensor<2x6x9x10xf32>, tensor<6x3x4x5xf32>
  }
  func.func private @expected() -> tensor<2x6x6x6xf32> {
    %0 = stablehlo.constant dense<"0x667DCF4281D10B430630B0C2C23156C220AD7A411E160542F2DA1AC05C25A3C1E1CAFAC1EBA1354100C9AEC05AB54BC0B07CDFC2C05A92C14C61AA42502B4D4249DB8840E0ADE5BFF75B04C3FDCA3A42730E7142C0D52CC2C07593C0931F7342F0081142CA1AA9C21E3ED0411C543EC2CA4E83C2CE9F10C38F1FC4429378ACC149929FC20C172B41153C4AC12C89FBC2DAABAFC2E73DA1C280292ABF12E817C2E6229B416203AB42A28E08C3ACB86B413AD8EDC0313006C341EC07C2CA9B2F42DE97FCC221CFA9C02F1A0442E0AC37428A5A4442004A94C28CB0D4C1C46A25C24949B8C1C935F3C18C76D4C1B6746EC202AD22C17BF33C425CB2BB428F07B5414A5525427F9605422F5D1D42F76D85C2F1C5E6C2705038C131D0A4C16DB52DC208F48741AC22ACC2AAC75E42E32DB74284975041A4DB7941D224B44270A612C222DA1742281FA0C22381D8C25837B4C2D7DB864270D10B4390FA3F40023BA042D6788841B04539C0116A9742E6413B42F834CCC18929BEC1BD217DC2AA5C404245B1D7C15292BEC1E4243BC1E44C2BC0ED7224C280FF1B40CF6361C2F4F185C14D3DE8C1B53EA1427A288741BCBF8E41E143A5C2C6FD54C2802BAA41D40E1BC0B6622342F67B294201CFC4C2A0D2C94021A53BC228D728C1A19D87426010203F217F98C1ACBFD341E842EEC18CF20B41ACC1B242B000D642B816A2C2CF0AB1C275A510C268D0954217DE8C4168FD8AC28834F6C1EC1D0CC2B82EA4C2747BC9C21CDA71C0ACC5C641D22D86C2BAB91AC3985F1E405AE81243C7E17842406E5EBF2DF8FEC2306B5ABF39FFC5416BEC6EC194E955C193D700428079A6C0B01398BF88773F400D1B39C237C7A141763ADEC129C18A424B124942D8670EC0CB6A1B421467CF41BC597AC2B5ABD9C1819550422F467042103AE641BA2343C1FFB4B6414299D7C1AFE2824260780CC3DB747F42B7ADBC424DE010C233EDACC1E1ECC5C214CF504299B0BD4279575AC12D9A6EC208CBB8C26E18F4C2031C91C219A396C254FA3CC2E4EE4142688F84C12A6F9BC20CCF5FC244B6DA420B39EB4110550FC3C1135242291A2B4324116C418CB1BF420399DBC2160B97C224BC6342BBF9B442AA0C66C2BD0CDAC26C620CC2423309C1CB3E91423EAF0343A6B595C1DA4E6C42FE32AA42188EA3C27159D941B1D411C3563B4CC2B8A43A42595DDC417B1970C263FB9AC2127377C28E43C64212F0CA421AB525C284C696C226995DC204C103C30871C94024B067C1D172A541C6A0DB41921803C2E429DE41658325C284ECBBC29EA91E42E2A65E41A0DFAEBF52A7CF418770ECC2B60FD94105A34D42C73A59C2A519934186ABCC42667B94C2BE0BB44116D29E42E9C117C27066DE420895C041BD35CD420C4812439C57D341889235412C4F97C15C14ED413CC5ECC2403BE5C2A74AF1C1F61EB142AD4E6942BB43B5C1C2A2D1C1802864C04B08B34280A83D3FA76CA24233CE9F4216E78E42C8C3AD42D530F441E04F9BC09004F6C13244A541C13378C214473CC293A35EC00011B3BE88E17941B5C95842BF264042952D5BC273232FC12FBDA2C200866B406CB4024332A0ED4298E39AC1F4DA8B42E4410041B6A845C2770045C2B7006BC104D403C21E6672415760CB42B2CA02C2C07B3DC02486F6C173FA4C420B3C7E4198B74A422B7243C14831F3412A45064392EA3842A0FBDBBEC6519FC1DA7B18C240FFE6BE72B165C1860B2DC1CC8C05C1546C0E4207886042582DA9C274BBC2C162E172C266865FC2F03FD4C0DED024C14AD0C5C110163FC2DBE20342D83E0E42C18EB7C1EC5C1DC31B17A8C222BED7C1A608664221609EC1E711B8C2E1C1F84159BFD4C12F3EE14185599B42928C7F420ECB4B4208E013C256BF81C0EE14804281B195C293BDF4C288963CC17ACE9341791CA342568D1C423593674270BABB40F6E17C42E6FD8D4262D6FE427471BD42F1DE95C27E19C3415CB623421F12B3402CB79941F0D08541A9CE43C22E189DC1CF338E41100BBBC024740041561D3141084A92C1F5B87FC296EF7FC2CD4A93410455214018F5D4C0D4C66D40C0EE05428EE47DC27D69FF406B1417C2007F8D3FBEA2E2C0494B42C26ADBF74204B74541D99B3A429CB727C1A876714157203DC2D23E8742B5A28EC16FD793C2CEE05941229F74C23B9F39C25986ABC0BDF780C2D5829041C4257942F0F1C9C1107C72406F5ED4C1869F1CC2B0B38741EAFDE2C2D8E15EC2108341C2D89ACCC04A3CD642356F2943BEEE15C2E5F0B8C2F1691CC255A0BFC1E648F5C274E627C31C7F4BC35FB560C2985035C15CD7CB426667EE424F7E22C300A393BF49198342B53A6A42CD0A74C257DF56C2457D0141275AAC4257A590427AE4B1C28448B4C286519CC2A2A3D7C1318D284201FB1CC299B9A4C2C4985141"> : tensor<2x6x6x6xf32>
    return %0 : tensor<2x6x6x6xf32>
  }
}

