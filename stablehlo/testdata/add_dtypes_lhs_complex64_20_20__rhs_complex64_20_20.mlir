// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<20x20xcomplex<f32>>
    %2 = stablehlo.add %0#0, %0#1 : tensor<20x20xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) {
    %0 = stablehlo.constant dense<"0x70D554402296B840FC3AC83E334995BD5796B33F2AA1ED3E74BAB8BFF099E43F57AE8DBF10069940E7F3F2BF4E64A2C08B1C953E9DD06340B0CB853F6239B73F5B937940B85ECD3EB5CF353F6D6B3F3F6C963A4066B7D8BFAE9BDAC0C9046940C006D440656052BFE81F7140B3DF38C04343F6C0F18D14BFB38A8B3F2B78DA3D0D4CCB3EC5E79CC07C1D0441C07D6D4060AEBEBFECC0B4400F2C53BFB476AABF30279FC043AFDE3F30FB5340A4432A4092BD673FB08F02C04D5B44409AC209404494333E4F735140271236C083E8AB3FEB7DCE3FE2AABD3F4CBC68C04BE98B3F08841EC03DBD8E3F7AF9B83FCA4D2E3EFACC1AC0544E75BFD57333BF3A45AEBF1C854F3F8545CBBF165AD93F98F37AC03866D7BF70E57BBEF53614C0E6021EC07A2BDDBFA642A3C0203C87BF8BADC9BE7902993FB51EF6BF54A90E41821D3ABDFA6A1640A97A9E3E1207E1BE96DA72408EB67F40867894C0313040C00887D5BF049A5940DDC965C0833E0E4091E69FBF0E572DBFC603F33EDCD7AF40F2209B401D611740157610C080D2E5BF828A9E3E42320AC08ADFEA3F2768C0C0F01823BFC2664ABFF8D6434067E51340E230B0408DA80F4086078FBF356D80408A92DD3F483992BF6D0527C0812DFDBF93596140A13D02403452ACC0FCFB943ED1F11D40B6AF8E3F6476104101BA8BBF03754040CB72B0C0291191C0FFEB76C01048ADBA4333D0BE44CFD2C0D3FA2740F505413FB4C872C0D06477C0895E96C0C2CB813FFFC568C0FD68DD3F7A08DEBF3D2A26408B7F683E4E1872BF6338594015B05C40FCC300BF2C5801C1B26B3440E2B828C048D08C3EC4468740B928A0BEFB5218401CD27E4045B5B8C079515FBF93F24C4094B43B3E6ED20FC048A9C6BF575B6340B3CB0540A1AC34BF739AEB3F2AFE823DB8DA97C0F1305540EF5C5BC0EEC35CC074EF30C0D4C48CBDEC19E6BEE7BC2040E9E26D3FB51768C05AF258BF5F7A27402CD704402CF672C0D73345BEDE47F1BEFA64D5BF00F81AC0900130405CD95EC06C95834070044FC0827A35409A5FFB3FD1FD773FF4374BC05C2D2B408E91A2407569B5C031BF7DBF8B2DD8BFB3FC64C0E69BA53F90E93040BD4626C05520664095E9043DE93AA13F0A5FCD4002D992BEFAA52EBEA53CC5C0280B53C0B6E62D404FB760BFA83514BB1E81933F085B2DBFA5951E400D2E7DBF810E40BFFADB343F2E4A9040CB5A24C03242A8BEB3378C409E77BEC0EDB2F03EC3D191C0E25A263F323A6B3E5E84C03F3D9D39BFA90FC43EC21BCAC0D1DCC7BDB7F27C40C7B5124070643A403B4DAABFF3A31F405ACA18C094E46F40A6798C3F941C07C0E3AD36C053B00FC0BB9A55403CD48F3E2AAD1140CEA634C043722ABEE08FAA40F15686403B1F4BBFD5D1A7BF1507A7BD98542C40AF90B4BF1A7EA8BF14A3B8C08C36F43EA50C683F7318F9BF0ED94FC0F0578DBECECBEEBF90959EC0D6661540FD8C0AC0018B2B3E4E934BC0298338BF555BE43F3496C5BEFFF42E40F25F2DC0983016C06213003FFE0F4E3BF58AB3BF7EA580408CD72CBFA66F843FED6C2A3FBCC2B2BEEC13DB4036100140FF50F83F5E3574C000FBBB3F1B6FD340DA3ACD3F5D25713EB8013D408F30404080D475C021E2CB40087B3F40F2EA1A3EB08414C0737A33C069EFB83FA1F1204073F8E1BF3DAB1F40B1600BBEBA2E15BF15AD2EC0E92F02C03B7EAE3F553FAE40B653D73EAD1825405711B9401506C0BF9F1E3240A874863FF6F6A3C00AB4AFBE2C59A13F798B8FC01BB2653F1C2583C0CC08863F5CDD98C05024B93FAD76943E06D5D9BFFEBD1040E833253E81B73F40D54A853ED9979040783022C05403373FD9EC9C405DEB3DC011600DBF66EE0B40E51582C0E4492D3FAFCF3640347D87C0B23A733F530BBB3F9840063FDE2CDB40CF9207BF365E1740B473EE3F157A56BEFC38D4BE77415D3E5C0957C0AD5A4AC018D51440B8B48A3DF9FAE43FB70263409B4E1A40FCC52DBF5C9BD1C08113F43D067B15C0E37F3140EE60D53FF14E04C19A3A3140CEDEAC3F999931BEB8CA72BF21418840BCE80AC07206C43FF38747401C5AE8BFEDDF6240565C2CBDA0E79B3FF9922CC001F53E3F7FECFCBFFCDC4F400BE397BEAC6B8D40C4150A40D853F8BFAC8EF23E55F3FE3F4EE56E40E0E27E402BE3FF3FEA4ED8BE5A81473F4E934BC01B19133E2001A0C0A9BC1CC07A61A3C0273098C02302034145C36740AA788C3F19F9BA3F01809A3EF0897C40F930083CA6D466BF8CFFC43F344394402BE181409BEBCF3F6104D33F1A6ADE40F54C0D3FB2DE00C0F584433F924811BF819C25401C345640513AF9BE29F287BDF9C93DC000157B408974E63EC15E4E404693EEBE2BCF3DC0A3FD0040DB2ED0BEDA790DC0250C6C3FBA4DA33F6C640340B8CC5A40A6B02AC0FD405B40A33F0740B74B324088034240A748FDBEA81225BFC3BF41C0127511407CA77A4042440D4046D58EBEF0409340A039323F5A696EBF3FD34E4004510C3E5B8DF4C099773740CC094E40D40365BF30878E3E5EB1CCBFF58429C05CF500403BABC7BF8820433FC8C43DC0496B5140C453B0C0EC96FFBF8F72D13D34F39440E7E2FE3F39D70A40787CB840FA0F513FB76DA94045B56E3FB11E1BC02A373B40613B17C0910512BA965765BE333EC03F9BE7124047D591BFE60EC9BF5454EF3EB978074038C778C0592BC23F1727EABE20B03F40321D34BFAFDA8C401DC6ABBFB82249402BEF94BF0A1D41C0F4CFD2BFAD87613FB4E0A94012BF0E40184911BF684CDF3F8A3703BFAA3108BEED0BD33F6467D43F62CB19C036B4BA401299224077A80C41A4031940311300C0AA586EBF7BAB203FFA9F2DC0B4D578C0CD2768C00E0D29BFB882493ECA924EC049AC6D3E55274C3F7C3588C061CF38BF7E4D3340548B4FC09F1B77C0D71EF63F6626B3BFFA6EB2BF84C706C01104594026768FC0986660C00090B23ED7BB6340A65B1240FCBF5CC00D5831C0C691BCC0B316B13F88BE01C0E0E2F6BF4CE7E1BFA50D3E4033AA1FC0291B3EC0A28D0441C64AF63F8D3E6D3D8B72C1BF533E7840B4A44740B0183F40D34D9DBE52C3AE400774D6401436B7C070EDAAC006D083C0158D34BF8D5C4D404BCF4240349686BFB5B7553F822973C0856856BF70A546C0392007408B3A97BFE4C3863FF58DBABF718149C0102FBE3FFB0D823F059B073FB16DFD3F9464EEBF2F688EBFAE49DE407CCFB54049ED06C00758E640DD8C04BE420B7F3E6460D3BF3C8E2BC0908AA43E767BA53F26D60EC0EFBE92C066FC5EBFBA4310BF47000EC02C867040530212C0E2F7CE3FB61E963D63C22F40549496BD9817104039CA87C01F7CB23FDEAC9A3F04B1AA3F248375BFE5D31B408E1B4CC01271B73EDE4DA73FF9365A40E2B4B63FEA480540B0C4A0C077A5F5BF00C56EBE99DC3C3FFA8F5C3E917E95C0B881ADBE50E7113F87E41840A190EDBFB96E93C0582B4540B48B82C0C3942C40356B16C0A1A2BBBF9D82BA40019BFA3C42E9863FC4B4054006E61F3F256CA6BD717B6DC0310206400E39463F50E9E240C50C463FDBFD16C00B8C5DBFBFD6473DC291CE3E30D91FBFEBE676405AC718C0505B95BFB7AF9EBE292D51402D750F40D99BB4C089A1043F750196C04A87DE3ED25F1D4094B080400D9EAFC0660E6440CBC5CEBF54303A40359D11BFC7C291C06116413F9B920D3FDFB164BF99CEC5BE314469C0835AECBF489D5BBFC0B03DC013F49B3F4BE1123E891FA33FB9F19C4006F30640A4CB22C09A8420C0127957C0CC13E73F8E954CBFBEC9F03F672FBDBF27E613C0BE2D803F9A4B89C06F4BF9BF8C38DBBE03DC65C0C22731C0462E2BBF852EF6BFF9CC91C08962804055099840579B70401863B240DFF60A402F00B8BFF92D14C035933E40A788CA3E233A843F3812AEBFB694F23F9D28913F13195AC0DFFB0440C9FC69C05156A1BFFB516ABF543EAFBEC6F312C09CFB8F3F59725A40F563F13F28290EC0757144C0F29744BF09AD2F4015C82440CB33654052D8DBBF93FB0740D35E99BC47E5A0C0F2D028BF5A2A8540CD0D7C3F0B40C5C03C8475BF7828D3C05EB8944004DACA3F3F950CBF70C7E2C0966F2A3E899976C0CBFAB03D45AF874017E24EBE6BFC0E3F2B8F8D40A90BA93F562359C0930C943F303B82BE348D983F40879B40365C5CC004E241BFB90EE63D504C53C03F1B63401CD2D8BF467300402A62084064550041A3ED46C087DC71BD0402E7BF9EC24FBE3584D340BE9A82BF3508AD405E4F114050B41F4040FD72BE9A1017401D129AC07723AF3FCC8D1DC0E1840B410B811B409EFDD1BFA4AF80C0A7D5CAC02E898040419A0CC071D398C0C446993F0D8B6B3FB638853E30DE1840C1308ABF73A80DBF2D32F1BF1F1A11C0D08E953FA717D0BDC803A4BF95F23ABFD0352740B910A7C0A22591C020B881C0317BEEC0AE0DBCBF44B602411E1AE8BD2A0337406F9111C0737EAF3FCD109F40E238F63E1B6665BF"> : tensor<20x20xcomplex<f32>>
    %1 = stablehlo.constant dense<"0xBA364AC059CD53C06A117DC00BC45D3F5BF131C0F6E236C0124D8D40C20CB33F7E1D9B3F648386C086303B402EB73BC0DF8216403CD25BC09A83523E493F9F40F4978FBFE5A82EC04F306EC082A757BF793FBCBFA88C0740BFE1A4C01A70FC3F8F1B44BF4B17FB3E216C4BC072B022403F5A3940DEFC20C036244940F98E45C0E7B0A840D5BCB93FE1D23DC0CD6286401EAACDBE518206BFE927CF3FDFAAB13FCE6D48C00B826140C8E772407D10443F5E3AE4BE861250C027F6D13F5C788EBF6F028A400D4FE93E19634BBF01FDB1C09EAFF6BDAF592B3F992F80BEFB75BEBEDCE18DBF001709408E888CBF6CC1A9C0AE7766BE930F50C0D630E640810D11BF268494C0ACB24FC0F491AE3EE5052BC02C171AC01A2094C049D0A9BF575097BE65FE7240660158C0E8A65BBE4DFBDFBF3731A64084E69B40768C38C03B882A3F05B31D40C9460F40DA72303FB9F5AFBF16C575C0478B653FE349D3BF38370D4089C52040EC342A40B778373DE37E7BBB7592B3C00F6D87C095A00CC001DD33C086F44640BAEE3040CDEA6E3FC5B700BF9A3901C07248EC3F02AB28C078B795BC4ED41DC00510703F723B6F3FA3C96E3FABB6FCBF600AA93F6D9DA23E641DA040A7013A3D10018D3FB14004C094098EC0784CA7C061DA93C0CA1EA1C0462E64C029E82FBFD8CD72403F4603C009AB15C0ED2CF73FF1BD9D406844163F8990F8BEB895AC3C071E08C00AFC4DC0560336C017F612C0C66EECBE848991C0963E913F8F0D11C0FDF45640F76F173FFAB67BBF1A6097BF847F46BF4C30F8BF1EA20FC0F83AECBF118FF13F5A171CBFB4F253BFADB17FC00CC027BE455CBE4014FB5BBF2325AABF0E1B9D3D3642A33E4B735840C570834043BB74BFDB6A8D40BA095040235FE23ECC3F8B40543716BF354DE13F643335BF3B945BC00CBCDB3FCE6820C0E2397FBF63D2BEC0B38D54404C1CAABFDA1E7540D7413240D294B8BF432D923F209F45BF9CEECABF38ECC64079FA4140BC50FDBE8D2B75C0307C2DBF2175D03D05BB09407B3BD43EB192DABE9AA97540EFE9294075211F408FDFA7BFF09A4DC0889CA33FAD673DC01F4085C0826C004081B09D3F820D0FC0AB00A2C012A68BC06BD7DA3E33E3AA3F13BCF8404E35833FD8E60F402856BD403558C7BF961B0DC0471D64401BC76D3D42DB35C063D9AEC0124235BF740A07C18EF36FC047740FC0D679AF401169593F65C56ABF40992B4050005F4097F98BBF5B802DC037DC47BF005CB7C013A91240F5896E3F475C4640161E65C065BA7C407F48C44055E778BF5596893F39874CC0E906953F1E7698BFD5E9A53E1AF8D73F0A63E0BF4FFB1E400A09DDC097DDB74013735AC0F794D13F44CB27C021768B3FB846A540C0710C402B70A3BFA549B13FC045943E572A6BC069E73740B551B7409966AFBF88073CBF5026ABC02D7D9FBEC85E35C03AE1C3BF9BB452C09F7933C0688710BFF7D2823F33DACABF8AC09D4061D513C0B523AF40028CB3C00C390D405AB1DFBE16A5E03F73D4B7C0FFF65EC0C2AC4EC0275186BF66AE97BF8D1977BFB2288A3E980628C012EFCB4083C289BF50F4284094492BBF8ADD77BEFEB8054069B5CFBFE3D209C0650878BF32DF0AC0BA2004BC04935BC09E18CABF85A6D6BFC4228AC0CE91A740C17B033FC459FF40006F99C0DFD43DC076113E3F7928C5BF955C9EC01C22F83FFCD7A1400908AD40945588BF197AFE3D7C8A85C00653053E1018FC3EADB2B8C0FC0E3E3F444F90C09E08A23FFAC191BFB8BE5E3F3105184095F4B240399586402ABF4ABF0DA222C0A74237400EC597BEAD4DBD406D2F2EC0075A2F3DCC9AE73E18C12AC0EC016ABFA6FC443FB3528F3FDD8A90BF98A97E40FE017BC0C02222406862AD3FD3A9FF3FA08A41C0FC74C3C0A1E43940F3A3FDBFADB10CC0FAAE9CBFE03226C0AAD3783FDB4568BF261B44BF3AF1F03FA01244BF568C2740F75F80C0A03756C0ED9C1440C70374C0C4D3D7BF308B1C400455023F8656193F964F303F6B634140AF4EF93F12CD25404DC1DBBDE2DCE6C06E31304084F656BF7E0CB5C090AF593FE310723F7E5FA0C0C9F3EFBE6A9C02407BC24A40308D28406BB2F73F7EC85DC0657581407B1CAF3F456B8D403F86324088F619C0488844BF0740DFBF709D37BDDB8ABE3F320E6CBF0BB1D4BF72CC07C059D8083F5E8FF2BD92B97140CB4CDFBE2ECED53FACDC54BE56DDD43FB36A25C028EE57C04D55593F4F5B053F321123C026D427408E7589C0F4F02A3D68570AC1F43D3240E5AC19BF127B8A40F49B8040965C3840267B91BF15EF6DC01CF0CF40428D1E404298FEBF60A10FBF19259ABFF0612AC0235296BE5C53A03FCEC0F93FA6EE81C09DE621408ADA74BE7B599AC09EC2074081DB58BFA011E3BF5D5B0FC0541270BFD2491E405111FA3FFDB985BFD215BABF310C5240974290BF1D42E9BFE726F93D991771BF5F4C8FC015843FC05D623F4046645CBF85124CC05F2EB5C0B7C9F83C41D80F3F33F1AABF1A6A9BBF609B934041488CC0EF6D563F9BCF58BE3E6AFEBF6BC6BB40DF0FF93F4AC685C07C74CBBFA1BD99BF6D9592BCAC62CFBE6CA498C0D253083FFBBF1B3FB28DF13F3F9F16C0E267FE3F3199D0406992C2C0D2B74FC07297CA3F386F15BE08EF18C00775BC3EAA0DD440CB82C83F0E1A4C40018AB4BFD7972B3F360D96C0F31F33BEFE77CDBE61B215C048CC0AC0AA25C33FBBD34B4042D73DBF33AB13409FE2AFBFF5D251C0081EC4BEABC437C0191A67BFE85EC5BFA2B283BF94D4094047AE65C0EC1DB140FADD5A3FCD7A71BE71165040CE99084021C033C0280AFFBFEA147F40DD2EDBBF8925303EE89645C06FA78D3F08768ABF9292A43FE02F5C3FD9B681C00BD4A640FA3DBE40DF85AFB9446C524089638DC0BDCD5E40AB99A1BF23F32CC0DE33C3C0BB27B4C05507D2BFF16B19C05E64A8BFB52649C09F60E6401D91FCBFB31C323F9524CDBF606450C0B1400EBF9A09E53E9D1E0ABF02F65FBF55BF32BF4BCA314054B72640C4A399409BF0CBC04766023F4AC4E13FE90FBCBF05DAC5BF497F92407B849D3FB6C101C121B785BFD8ECE5C080D2DD3FEE22B6BFDF800140C41000C0C6FDA13F7F338D3F82B08340A709E63FF7B754C09479AD401D072C40FD479DBF13B121BBC1379CC00C9BECBE460D7F3E7C247F3FA41B753F7C43884091D055C07D88834079A188C08769D3BF743A7140BE199A3F6995A4C0174B83BE18911CBDD3E6593FC16D5C3F5D553940DDE25CC00C10CEBE6DE93140E75B4440123CA83F1800AEC07843D93F97240A407AFCA53E4724AF40A4914740B4D65540A0BDB13FF94FDFBF69AA90C0B28C74401C5639BF21A899403238C3C0A85D963FFB718A3FB393A5C07F7CA1BFD6052C40A0BAE4BE0944C9BF8004A4C050D3D7BF823E0CC088CF054047491CC09B3AA440873966C0FFBFADBF4422BDC02C3A60C0BA01DFBF55FD5DC0B2FC9D3FCA98673FED981740BCE24D3F381D8E404DC4293F2B79A83F6EC34440CDF087BF318CB53E7B80993F80DA2EBFF420F13F4DA188BF94C83B3F7574744054AE963DD81AC03F504C6940BD5FFDBFE93C63C077254ABF1E213AC0732D19C0491B5D3F700411400E118FBF1AA8E1C0034A50C0C90A463F457407BE4E62713F2A497D40E127A2C0292B29BEEA3E2740C7AF4BC02309323E312C6F40D4B691BF67501E3FED30CABED0F206C0B1C479BFA8887E40131637BED718BB3FE429BE3F718145BFB5C00EC0752DE23F58AFBDBF9700B43EDE235BBF8EFE8FBF7323364011887E408D29553F5DE98BBE418B6C402E767C406EFD9CBEB80DE03F09DA0D407256A7BF954AAFC0023316C01D2494C024F1BF3F6431123DEF336D3E291386BE58C2D0BF060C053FEB02C73F637CD6BFF834A9C00D000B408B0B4F4077B02140E5E36F4064CECD40A074F9BF8BCC83C02AE58BC075CF3FC052E006C0C1762BC090653CC0F1778C404372384033639A3F1CB0614045D622BFC12FB2C05ABC3EBFE3C520C0EB61933FA53D4BC0767FCBC0CC5D62409744F73F199B88BDC728F23E4FA00E41CA61893F5554A23F4DEAA7BF56F760BF8D05DC402E109D4022A81540FC64CC3EB8FBB5C0EE3477407A8C1FC0D35F79C0A7D2FABF1C258340D9CF9D40841402408A0258C0489844C0E15D5240A29CA0C00DC4A8C05E4E11BFB1C6DF3F3BDCDD3F69D9B53F13E5373FDE20A24053FD8DC0BA71BD3F19B806C023C52EC01E1C2F3FEF0FAAC07EE32BC0B2257F407907E13EBB3534402E9D3E4007C611C06FAD843F0AA4BAC0E017E03FFC16D93F2EE94F40840760C03F832DBF424127400ECA303FF7EB85407AF38E40903C5BC0CF95E2BCA2B989BF0F5D7AC0497EA840332902C0317905BFD6516FBF482FA9C05528394014A3F1C0077208C0FC720040A86292BF1BD8C7BECD81DEBF4D198FC0A893D0403BADF7BF9C8F88C084323F3F6BAF273FE077D63E518D0EC0FBA7FD3F74988CC0"> : tensor<20x20xcomplex<f32>>
    return %0, %1 : tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>
  }
  func.func private @expected() -> tensor<20x20xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0x60EB293EEB5E1D400A0A64C0E51A4B3F5F4CB0BFD12E19C0EA3C3E4059D34B4070F2D63D6015143F256D833FF21F00C17026294020CCFF3D231CA03FA20DCD4061C731400EFD14C062BC40C0A8E0C1BD5FEDB83FA887D93EB6BE3FC16B9EB3404E83BB407FA9A9BE1CCF163F087AB1BE249699C05A2046C0C874874038BB3EC0A865B54020F15CC08851A940AD21FD40E818F2BFA2F0A340C3234B3F6085663D0CAF01C1D66CA8407C71E340C3475B40C640EB3E1B51A9C030AB9640D80C853F119F8F40319D6E40EDEA68C0E00287C0F112BF3FDDAB09403FC278C09897383FF67465C09E755040B0C3B13EFE4EA4C0753429C094B186C05BC2CF40FACBF6BF052775C0B7AA9AC04A7F0240BEFCD2C024E582C046FF9BC01A1F69C0F1EC30C0A8680440ACA107C1FDB0A2BF583309C0D571CC40AEBD3C406D0CC14063E61E3F000F9A401E16234044BD7F3EBADF1A4080171F3E3A8E6FC091EA94C0D0CE093FC62FBD40C4536EBF661C11405064A0BF573DC9C0A57970C0230F5340E3640240D22AAF4094E2013F33BA5CBF10CA45BEEEB585C0FE936B40D45E0AC1ACC627BFFE6D50C0F9DA7F4044B44F40160ACE40BC698A3ED016503E0C978A400682D7403B698CBFCA09C1BFB96B81C054E66ABF4F5B4CC04A1620C10ACF97C0EA788CBF86EEDA3EDA294D41402349C0E8272B3F204F65C080CCCA3EE55A51C0D13DF9BEE869C5BE246F0BC1DC0418BFD9C105C066DFC2C054798AC006F413C12C850940C7E9BCC0BED4A2407E5092BFFD78CE3F51A074BFE94BDCBF7A40BA3FEE1B9A3F7B4E16C0944CC6C0DC650D408FB55DC0A4176EC0C4088240B959B4406CA8C23F8ABF2940D940B6C05EB00DBFEFB2D2406A4E89403F014DC01281374088B2D9409717224070546940C97EA03F187DE93F2481AEC04069CCBDD2FDDABF5E96BEC0ECBD70C07605C1C076CA3740825D973FCA4B9840785757BF008712C000917040C8DEA63FBD36ACC099C2C0407DD12340945C0AC0C611C8C084A20440B35558C0EE72C840017D34C02C281A40B4ACB94063E96740FC5930BF297BAE3F5810EF3F53828CC079D77CC0824BBBC06220C9BF34A621403870073F0A24F5C03CAF44BF9E74EB3E0E0F26408E0D63411BFE3C3F78FC0440A0CF7CBEA15B9BC0802C033F73EF2B40C083643D6635D8BFC484C4C0418AE23F55DD16C197FB8FC0917AC4BF02E21F410E01DCBF3F739FBF5304E240ECEE1DC0B8991FBFF091E8C0540506BE2E02B0C042EB7240E0B2533E3CDE5E4066551EC17E7B7640ED602141E4F7A83F9A2F7F40EBD690C068276A40690565C0E7508240E0383240194E77C0A094BDBE9A7012C17A5511418C7848C0A6777A400939AEC0B14F6C3F4CEB2741D18FCC40E47F04C0007D973DF607553EFC567BBF233EBB3F2E328D40BA7CE4C084D883BEBB248EC0DF7B10C0EB9BC2C03637E7BF410DA5C06052F8C0F889E23F034792BFD368B5BF8CDBDF3F2BF641C08A3AE84065E5BFC006179E401D5649C0347817BF07D2A7C07BC35EC01E3994C068223E402C1AEEBFF82D8E3D46816F3FF05E3EC07F815341D2BB703F688E9240E2838FC04FFF9C3FCD250B41C0A39EBC1A81F5BF3EFFFD3F7445553FA15876C03E313C4072DDB43F2749C3BF1C65D4C029A91B404AADFA3F4AE927411DEDD1C0104DF1BE4A391B3FEBDF07C020B3F5C060DBC3BD8B77CD40AFA32D414D0125BF7E0C2D406C1BCE3FB45BAFBFA1A15140831597C016358CC0854A9BC0E5B02140F8FBB3C06A38E23F0E8ADCBFC876D440184112BF7689273F371310C048B0943FB88AFB3F4C77C240A0408C3E16369B3E86119F40C878A6C060FA4BBE6E8CB5400784ECBFE63AD7BFFF4BC540E496FFC039754D4072C086407E250FC0F4BB04C027B294C0C7745B40E1C39B4061962EC0720D923F18E43BBF2535433F2C31A9BFC8CA0CBF7E21BDBF555F7BC0B7309E40486A7CC04774C7BFD2CFBB40586AB3BF615B17C0C45583C074D7203FC94ADEBFC8935D40F1099640364ACAC0D683AB40B9229F3FAF69ECC080FDE63FA1C45A40DC80FAC01D6F1840160682400576DAC074E14440F1D5FF3F265B8C4020B980BD76962B405F1FAEC0E363E940B823893F786B0D41024E9E403A108BC0E48196BE709A7D3ED8066C402714AF4012DC893F636205C037D8ABBF385D29C0608BCE3C5C919CBF42A638C0DDDB5BC00CD79EC0CE9D1D4124B1843FD3B111C0E0D11340509B523F7CF1B23F575C28402350A6C01457CA3F9C6B80C02500DB402815833F2A3CBF4007832F41D3AF5B40459C49C0D80D3DC00AC7BD40E214A240F6CFAD3F441F86BF3CA4A2BFF415B4C0BC4A68407EF0D93F949FA540DAD790C07044DFBEF55FE33F695CA7C080E7B6BD2085993D980FFFBE106F3FBE23C81E40406D46BED324AC4049C5883F9C81AA3FDC07CA40C194CFBFB8E51DC08CF639C0585EAA3F08C50FBF4CFF48BFB4872D40CE686F401D841FC08AFBD2C0D2C4504082EC323FD4A40FC11885D33F46A0FA40BCE8A8C0C4D88E3F51CBE7BF0A5D94C01941FC409092C53E72C45AC083BF91C0788C044059E6B0C0CCB719C0A25E95C0AEFDA5407261264009CF8140B1595A40F077334074033D41C0BBA4C0426BB5C072419040549220C028F818C07892133E9B0E0241002977406A2F034074CC3EC000A1913FB3A124C09CFC81C05ACD8E3F44F732C0608F533F222E523F8CC4F240DF5805C0F666AE40E56822C00078C9C0BBEB01C080C5FEBF71FD8C40783E303F2E57CCBFC87A7940153E83C05FDCAC40753D20400A38B63F3C2C593F1D01FF40783889BE648ED940470CCC40A0AA6DC0484F42BF096C1DC08598CDBF5C089FC084DE15C0488B4C3E86D576C0982AFE3F5CABC54064114C3FD0FA77BF757DA4C09E0DC940152C90C06107D2C028AC85C054F1E0C0283B42C0BA1990C0E2D104408009F4C0A65A6C401DEDCFBF822188406E252F3F2E92D6C039E854C02C41AEC0C90E583F08BC39C0452128C04AAD813F7C62B240559D1340187F15C106B40C4188076C40F5A5B4BF48A643C0394F074179338B4014F7A3C0960AADBF18A6DCBF54F40641D0BEE4C0015A54C068D8C3C0776E0F3F26FB89402818E540E6E63E3F0A4A1FC04C93CF3FF8D9EC3FB7A48AC0CDF706406406C2C0423A173F4CAC9ABF52B809C0715E1C40FBC6A840D0E933C0E9E3C2409E3AC4C0DBE830C074732B41EC55DC400E0CE8C05623DE4023B12BBED2D48C3F07534ABF10725C3E8B5148C0E6EE633F1C4D0C3FEE43C2BF7CF7E23E8F08C0C02C7A05BF6255BD408885FABF40E2E2409A424C408CCCC2405B54A83F6EBE013F513A0CC161E5A6404007F83E6254C44096E8E1C0B902674090E206C0A21C9AC0E02B3A3D681EC340740C7B3F969B023F986422C164BC66C0D22A1BC0AE06354047800EC0A0C0EB3EBEE97BC0AE9849BF016061C03E81ABC0282FCBC0E88FC6BE0F1936C0F67A664000DC963C866229BFEA4F24412599313F36B11740193CA54028F7DFBE28F18B3E34BB20C02297B43FBE1E2A40FDC0C040ACEAC03F34EDBA3F40B64ABF8E59C63F448F81402AA626C010509D3EB8504BC0636782C06A032DC0FE398440CE3C90401C60D8C0E913D1C07626FEC037A79A3F8EE81440DEDC9E40E0E5C3BFB882C0BF30EBE3BF9FB7B040141770C07E328CC0E5B88F400DDB15BFF0C28CBEC3FF47BF801BB8C06E9E34C056A14740212249C075862B400D86D03FA1BD003FBD222B40C0097840A8D180C087040AC0052187C07C2A2E3F10FE02407876BB40413525BF536325C0105196403008B1BE654510C0953FA93FF403B0BF7E6982C05EB0C4C022A588C08BF812C1D25EB040B82D9940966E7F40E501AA40CC560A3F58F46ABF0EB242BF07AAA63F6E8C9CC01E1D4D40DE04F03F697D8D401A3C9C40B5834140F018043EF0CAF8C0BE3AB4C0F4637AC01CC81CC044359FC084CFE8BF1EB1F9401F9298401DEF81BF38F5E93E1CB7B3BF79B234C0FD31EA3FD0DB883FCEEC10BF248486BFD518CCC084D9BEBF1EDCA23FEE0783401891BA3F26013040C0FAE93D6393AAC0967B5540B2BC343FE572CA40846E0BC01B4F2040EA0C5DC0CD37B3C0DEA401419B7A2CC0B8A055C002B51D400668AD40B8F8C43FCE1A4C40F04968C05CA3F0BF185B02415E6507C14E00C1C00E19E9BEEFD1C6BFAE04A940CCE28BBE8B6C2E40F351E640EA5A65408C69D0BF8B7F0AC0122391C0ED56F63E18D1A53FDD306DC0874D16414D702D4006F5A9405A6D2F406052A93D82CD71C02CDB8EC0700736BFC0A726411CB5B5402A83A4C00C6096C00C6A6EC070A296405A7BFE3F70FF9DBE2E990EC05F76643FE9D650BFBEFDC2BF19F28540509325C063F719C094EE4CC094CB83C098A7324003520DC1AC2E37C066D4934063A9CBC024A39DC09358B9C03FCA3EC13C90A1403981C74004308CC0CBCF6640284BCFBF6B1CE53F49942F401A9B1D403745A9C0"> : tensor<20x20xcomplex<f32>>
    return %0 : tensor<20x20xcomplex<f32>>
  }
}
