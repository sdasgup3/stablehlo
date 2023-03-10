// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xcomplex<f32>>
    %1 = call @expected() : () -> tensor<20x20xcomplex<f32>>
    %2 = stablehlo.log %0 : tensor<20x20xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0xD5A8FBBFAA6407C0F2514240CC660A4055DEA43F54AA04C1BC993840A975503FADF3E3BF81A409C03C47E1406FBDDFBF6736BEBF4D777A40529A8B4034439FBF38839CBDA59DBBBFCFA691406C4F08C089C70AC0C34F22C0159E14C0C1275A40492ACABFF6B247C0C9BE94BF5C496FBEFDAA88BF8B3F63BEBE3C63BF37C76440D17EAE40F0580AC058FF55C07DDEC4402085E83FFF3E1740D5DA28C0CB97A8C05836E4C0B9D2FBC049C93140A05F5540534652C04BA70E40D378ECBFA60784C0C9F14FBF6C7D933D510C69BF97F7C3BF13E30DC057A843C0B4585140D9C21DC09B49E8BFF9F01C40CE4FFC3E9709BA40EFC33E407ECF87BF94641540326F2F4001771DBFF3A3A3C0A1FDC0BF648BFA3DADB7B43F96CAA23F940F3D40904353C022E7B7BF1D6EDCBF61EE9240A0A75E3FB7A29CBE4EAFD1BE25C2883F94AB85BE03A78340E8D9EABFE4E391406CED25C0DE58C640AE088AC00F7C38403CBA1E4051E85240D2C2993FE6BDD23EB5163240BE95D63ED0A5CE3E8547B43F8B520C4011E015406AA4D2BFBD4E5540AAE204BFD803C6BD8B5A8A409D00CB3F24D48FBFBB1F0E3EF2F274C0AB905A3FA32AD9BF5CC7854024014EC08602143F8E188440A525E8BD424AD0BE05038FBFA6878FBF283502C0B971C33FE85B714056F91A3F6FECFDBF570BD9C0A0F31740862A7C3F99DF0F406263774062A34B405C7786BFD5188EBD6FCA0AC088C4793F9CDCC93F4F12D83F910F49400BA732C08AC95B40B98B4740350646C0C0D6334046F3613F7F7C12402172904081186D3FD03A9BBFA6AD153FD90025407366C3C088E8223F39E88CC04836424042D3DFBE074E9E40266FA8400DEBBC3CB7A939C06CA4BA3F21108840F451B240140F9ABEEAD84340719B0A40109F6F3F78AA214031C13EBF28FDD43E1DAE09C030378F3D8ACD4FC07485B3BEFED4A63F2A2723BFC913A8BEDEEBA7BF0D8C1E40A5858F40A5AD52C0E15D77C0B3E8A3C0BD0E823E2FFB464009708FBF96F369BF2B4F5F40B21625C08903B7407BFB6B404B2E9EC0CDE392C05C72C1406DEEF33E0D01D63E33F370C05EF4A440859CAABFD71A373F5F2F474085092C3FF962D1BDCF9256C0C149C23F9467F53FC212CD3F8FBC8F40F82CB3BFF264A8403B31FBBF0C5C4840201D77BF8BAC0E40BE5635C07ED4A8BE3C9F143F3454D93FA978E5400BFB93BFC6A545BFB9EA7EC05373E53FB6821FC0EB764640A0E71C40D21614C039625CC037A431BFF69FF14093F92440968E09C0E4419640CC08D2BF3FD328BFFC954B40A99D0BC0EACC353FB8A98640D3647B40801339403B15DC4025BB88BF83219A3D8CA745BFD56002C03CFFC5BFD6E6153F27DC16C0FFC23EC038B7413E32E3FD405304983FC0322B406CD258406E9BFF3FB2A1743EB53D0FBF2CE50FC078BE103FAD4EE2BFD7455DBF4E9F3A3E3F5F56C0E576D93F716CC83F29230D40204CA73F1BA46C40606112406BDEBDBF75C19FC08FB18FBED85BF4BF08F361BE934C03C03D1254C08434274015079A3FB547D2BF95D74840DEF49CC0DF888C3FFFADAFC0171A4EC04FF79E3F439312C0A82286C0DC5A823F7BC4373F6636FDC0DB954B3F366AB53FE2346ABE667DB6BF2037CB3FE3B2C7C0596112BFE82F32408D48BE3F1CBDA3404E9B98C08858863F2F3392BF0DFDBABE90BE83C0DF29B43FD4419FBFF54E7FC06DC41440119812BE7E013B40436F9A3F0A3555C07CEB2EBF522E4BC0412C6F3FB73D84BFCA462AC0D5150C405DA59ABF20043C40120DFE3F01867840610C95C0B3C4AFBF3FD3B740DA2D244011916A403CA929C02B9F7DBFE5D928C0903E37C03F46A03D470095BED421A2BFE7BEA8C093687F3F4B641E408FAEBC3E246138BF34CFB4BE95470140FA4AB44028140140E96050C011B5B5C08A14D6C03FBB57C069D3653FF17C82C02E380940EF43C4BE89ED2A3F33BB1F4008F63EC0D0EA26BF4C37F1C080E3583E2A28F8BFE2FE8240C1A422C0F96A2CC00E0EC53F0E6014BF8685223F1A34ACC0FFCF7F4068FA054044E08D40EB392440E1BB7DC01BC693407DA5BEBFAD8118BF385DA9BF04E4554067BC1D3F076986C0C48905BE3A0B8F40D5E05640BEE4F73F90CCDD3FD308CDBF082304C0E5A6863E82F12CC06F42BB409B1C1D3F92B38CBCB4E41540F345D7BF1F5CD73F45C5193F7BC412C017AC4340210435C0EB1780C0AD56B3C069797DC0958258C0A3A9DF3E28339E40FA8CD33E3F951BBF135993C057287F4010E81BC0D37CDA405AC0B2C0A6567F3FD86ECFBDD5F4DDBFFCF5C73F76A4C43F129748C0794A1F404F9FA3BF30CAC6BF42779DBFA1C8A1C0917E38BF720FAFC0C8EA75C02562ACC0B3FD96C02C2DA03F3D32B2BDB001B3C09581463DBD77D23F983EEBBFD7D3BFC09D1D5140DEB8BCBF262B403E148D68C0DA1407404D7315C08CCFBD3F9C8D07C0E5821D40038E33BCAB84E740BA305A40EA2915C0818D26C09AA83D40960126C043FA29C0A9671840428BF73F4B0CB9C02F2D48C04DE49C3F52490840C2AC404080BF7ABF57C29540ED7FA33FEF016E409C181840176820407A2727401F012EC07D4295C0BECCF4BF9401D04001E40A405966863F66439D3F8AB08BBC3F4E3DC0090C2CBF31D3694078460A40E976F2BF3FC6874064AD1DBFA2388DBFD7DB10C0039FA7BF98E68A40BCB952BF26CF883FBC4EB5BE88835DC0DD880EC0B14AB13F8AFFADC048CBD2BE0AEDBE3F0EA606C06425D03F64372A3E4CB5AABF8DCA14C053AB2CC07F80E840C75D7F404CAE3E40A5C4E83EE5E39DC0DB7E9DBF1F91173F5A2CA1C0CA9F474017E548408FB9DFBF80A79F3FDCBC3540B209AC3FBC9A34BF5E5B8ABD5FEF30406ADBCF3FBE31B4BF2620D9BF447B9B3E34807DBF5F2BC1C0B98A143F77541A40B36C84C0DA4C333F791BD3BF38E3D1BF8354C2BF7DF31ABFF342B6C064F5FB3F2229A3BE812312BE782D064042C924BF4A4734BF4F75E93EE9E29D3F768E15406A0C4EC0794A323F222D7DBFDF4B6D40A23F034024B4D8BFBB39AE3FEB08FC3F58158340421FA34096CD57C0BD56FB3F71831FBFE21127C0CB5D0EC0E4687E40530335C0F6CF03C146FBF13E042846403442C53F9CE6C8BF55A3753ED62DFD3FEA0DF5C00609AE3F2757F03F06D34540EFC0FABF0B235B40A517893F5AE941405CA787C03216193F44A89C3F51669640354CBD3E918106C101322EC053467A3FC2745DC0C1B409C087E337C060F972C081F131C0B0289FBFCE0BAB3F1DE148405CBC4FBD0702013F01ACAA3F89BEF73FF574E9BE8308C7400E2E5B40FF11A540CDBCBDBF505120BFA741AEBFD6AADEBF90CEC3C02E786F3C246C13C09EDC5CBE15F57CC06739A63F8BA6CE3FE72C52C0A08B1FC0EC1CE93EEA41BB4039614E40EC2E443F034B0E3F53E4273FF12202BFD5B9313F914FF1BF6F45924061CB44C04BF6BE3FF170773FD637D240536D1DC07EF05640F11E79C0BAE37BC04EEE84407FE80FC098F4A3C0F6D0A53F99E82A40210FACC0FF2D4C40052D8940A0AC023F0E97C13F6E2BC6C00E02B6BF79A0A04094E60840219A93BFCB4D5CC098E69FC06CD773C082F54A4094878A408C59D1BDE64C063EB9670E3EC37E8540FE16D13FD9A6B34060D265409A5DABBE9C71CCBF8CE1C2BFC98CF7BF9A3A7CBF312931C0CEAE1A40188F2D4013712C40D51F1C40B88542395A26F73FBBA42B4064D24DBF0A19653C56B2E0404AA28940FD3E093F7E0410BF789D83C04C9436C01772333E04C8823F36D97F40A979CEC0158D8CC0FFCB7B4023910DC0EAAEF83D02212B40B80C58406CAC35C02F9AC33F9D9EC13EFF0505C1EB0610BFCB31844068C338C0F5B2D03E05D9853F64B454406F1A983F433625C0848127C08C27EAC0F9422D3ED5A102417DC3EF3D42F5A73EAEEFA2BEFBDBB940F2BFBDC028EC3F40F2FD0740203ED9BFAB9D7B4041715FBFF2770640488C503FFBBBEBBEBFC237C005EBC23FE0010FC0E7CC843F3063D8BFE337AEBF1C3C413EA5E18B40E353E03FC16D1440C44AF5BFE3978940D543E93DB5624B402B1D6DC03E583D401AF9D93E2D4BD7BF0F03E9BE6D1718BF84C5803EFA2E4940F82905BF118BD93E8E12DD3FBA422B3EB8113340656C6E3E5E95374042B2804022D6643F26F4653E40712E40E1627AC0884F0D406CC734BFF8017EC029F2B840932829400605273FA525D9BCBBE9C73F6FA5E9C0DBE38FBFADB7BA3F5D20AF40EC3E6ABF55732D40194A363F9909AB405F534540ACA3A4BF72D493C059945F40AF38693E6DEDCAC0E2DD9AC03DE899406005CF3F3A1EF84006BC29C0CE84DDBF31AC653FFAE43BC0520D56BF18BD8F4068B8263FF9AFC73EE0752E3F43881DBE17B0A640608EB5BDCE8FB4BF940A3CC0D738BD3F214019BFAA796A3E1F2290C082DC34C065F1D5C0D1E32DC0DAA0E03FBCFAD53F93A15F4008055A409C9789BF94FF3840845BE43F1D6BC6BFFFB951BF904B64C0"> : tensor<20x20xcomplex<f32>>
    return %0 : tensor<20x20xcomplex<f32>>
  }
  func.func private @expected() -> tensor<20x20xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0x53C1873F567414C05D6BA83F966F1E3F72230840F455B5BFE27F8C3F9BE08C3E3270833F5FCB10C032A2FD3F563679BEC844B73F5B83F73FB28EC13FC03F8EBE6377C43EC9BACFBF50A7CE3FC715E0BE773D9A3FFACD11C0655EB53F06CE0A40DB40A03F108502C0FF1A2E3E985C3CC05080B13D4BF23BC07FE2A63F5D37E83F4173E23FE13CC1BE6F1DF93F8C64044017CC8B3FAD5A6A3F6F06E33F823E02C070371740AEA613C040E0BB3FE049603FA57DB03F7CE92240B11BC13FE7F2FEBF24E350BE0C67434009C9133F13DD06C02915AA3FAEAF0CC00F7AB43FBE5225BF3AC48E3F2C4E0D40EFC3E13FE53CBE3F906E933F8917AFBE1FF9A33FAD8A5D3F9CCED13F6E62D8BFC4E9D33E62E143401158243FEFB63B3F6C82BE3FEE3E57BF5DC54E3FC60711C07A5BC53F29B23F3E78C72BBF9A970DC02CEFC23D076975BEFAA7C03F33C8D6BEA71FD43F6D5E04BF75640140C6A41BBF38F8AA3FC2E2353F1BA1A03FA9F8B23E5861843FBC43B63F11A20ABF1D3D443FDD31753FE1F27F3F7C9A863F3FD01CBF7EA09B3FF2351EBE156FBB3F84ECCB3FF2212A3F3ACD1DBFA9E1AB3F166CC4BF1E39243FBF628DBFD4E3D43F0DF927BF82B8B63F0F40B73F88AC5CBF4BD8EBBFFA26EB3E44AE16C0BD026F3F63DB1F402689AB3F01FC223EC04AFA3FBF7BEDBFC6AF713F435AC93E2DB6BF3F11A3853FDCC69A3FC045A3BEEB4B463FE427CDBF271A1E3F7C25823F26C2A23F2CF3893F2563BE3F63361040146EBD3F0B1548BFC4448A3FD1D29B3EDA8FCF3F6AFE8C3F2360D83EA9286BBF56DF783F4E84AC3F9448E83F8B6A4240189DD63FA57122407924CD3FFE57D43FB495D43FD2908F3B29BE963FEB402B40AD3FF93F40446B3F17C78F3F8E9BD53F52BF5B3FB2DED03EDCE6773FAEDD92BECCCE483FAA9CB0BFB2C6963F3C4EC6BF768B993E13B3EA3F5068AABED99E2AC03EF0833F38B3034018ADDB3F701622BF27F1ED3F07E70DC0A39E913F92A0BE3FB7D3BC3E0E471DC014DDBB3FA7F822BF9D74F53F1B99123F7157F43F012A19C0CDB3E63F6112A13D7479AA3FC1E8BABFBC0DD63F518D81BE729E943FB925AC3F2CA5C5BE779A1ABE87C8A63FA5DA2D401E6B6A3F0334323F3338C63FE3AC9ABE0EE3DC3FEDC1B6BED8E3973F232299BE8F25A43F8A7767BF06D5CEBE0497054004A8FF3FA04DAB3FE0ACA83EFF6023C007B5BC3F24FF2D4017C1B03F04DC0F40678D9B3F1CAA41BF5FD1A03FE2553CC0CCE904402B70A83E1E24D23F1601004084F2113FD69A30C0C2CAAC3F10E319BFF8BCB93FB4A9B33FA0D6CA3FD675223F0D58F83FD7C81DBEF40282BEDA9FBCBF216E703F777F1FC0362D633F63E5A9BF2C0D8C3F6401454057430540EF26183E1632BB3F6806673F17DF323F44D9F33D791A573F4148E8BF3C591E3F9871A1BF7F04FEBD25C33B40DF61A93F30052B40D8B57E3F1310743F7BEAAE3F1E929D3F055E803F4C4A13BFB003CE3FC77745C0EB3A273F83B241C0B81DAE3FA60208C08C3C873F8FFEDC3EDCE2A13F65650340C8ADCE3FE2F73A406AE6EC3F321C27C0B921753FD97589BFE61BBB3FF4CE394055A50440527BBDBF47A2F83EA39C873FD817BC3E3A6CDDBF1E64EE3FC42CA9BFB6B5853F45FDE23F9826D63FDBDEA43FC9FACA3F90323B40AEF9393E6E4135C00336BC3F94FA3340660AB73F9FC2EFBFAF6B583F27F17BBD0753933F637EC83E8AAB9C3F601D3CC0DB2E993FA8BE3640E23D863FD178F8BF86956A3F681D01BFA800A23F6C1B183FD0B4E63F223760BF1255E33F8718E73F55C5BF3F76CA753F2028853F2D2C32C0EAFCAD3F822E14C0F08799BFB26FA7BFFC68D83F723CE7BF47427B3F2A03983F17FF58BEA1868CBF83D8373F2B37DF3F4E02E53FE900B03E377CF03FD4D905C000E10040FB302BC06BF0B63FF052ADBF3F47473F252A35BEEAFC723F359AA73FCCE98E3FFD4A3BC0714D01409D434740AF5BC13FA0D600409296A73F14EE14C03ED6FE3EFF5AB8BE114DD83FEF07BABF81DDC03F6504F73E2E1CD13F2550063F702CE73F12EF11406CF7F13E08B630C0FBC2A33FAB50F93F2D10B93F126BB6BF3EB9BF3F89CBCC3F216AAD3FA9F0053FA6CA5B3F24033FBF10A63B3F0DF440400F84EE3FB0370040B1C4F9BE7633E5BC1C8C873F5F6D1FBF068C143F3791AF3E2D9DAB3FC4B70D40FC74CB3F2DE60BC0AE89F63F5EB021C0130F9D3FD7D740405801CD3FFCC4AA3DB191C43FEBDBD9BFE953C53F68690CBF5B570B40F5892FBFBD9B263B8A42CFBDD8F9583F41211A404D01A03F04BA8EBF17B3833F16EDF2BEC1082F3FE72F1EC0ECB6D03FFEFF3FC0F22FF33F9ADF21C0DCFDFB3F2C061BC0D218683ECF2B8EBD7661DC3FEA8148409C11673F344757BFCFE0F53F851F294057E5CA3E9CF5404087C1B73F0E622740D43C823F22DA244005C0963F98021240604DFD3F7D41C93F2F88B53F6E8319BF2AA3AF3F9FA6124000E5A73F3D0A16C0A87A8F3F879D2E3F470CF13F785429C01422663FB735863FD082933FA60CA1BECF23CA3FDE6B883E8E08BE3FD191113FB6A9A43FAD554E3F56D5D73F7E4E06C06DE8F43F54B0ED3FE045613F11B9E63E5BEE523ED36063BCED088E3F6CC33AC08C0AB93F1CB7083F0BA0C43FFECEFE3FD98E6F3E181E05C01E14763F437E27C0102BBE3FC3E83FBED2F5F23D37CEA3BE0619B53F1C7824C06AC4DC3F9D22A9BF9C85DF3E9B88EB3FBA5C7A3F00F31E403C5D973E9C30B9BFE694A23F1F0D12C06D5B07409290003F86368D3F670E1B3E902BD03FC36B39C050D2CF3F9515BABF2B60BE3FC7DF493F33A2433FEC6421404E87923F3157E23EA248B0BEB3F342C01B22953F88F5073F3A5A4A3F58DD10C04BE50F3D4AF9A2BF08B6E63FD3ED42403E81C83F308785BF174F153FA1A895BFD4D84D3F2A4219C0B466DF3F159DD6BFFEAE303F135A24BE911A3E3FFCC2D13F229040BDCCEC13C00A3B8C3E48BD9B3FE6BEB03FBC6471BFA4D8423E150E75BF13D1B83F1856013F449A463F36B91D40C8CBC13F81B58F3FBEB5E73F519B15BFB505393F63539DBEA4C19D3F1FE51BC001DDCA3FB44F1EBF58110740F4644540D9D39E3F7C75EC3E04B6EC3E075B3F40B05A044045B5A8BF533F573F45AE713FEC0EA63F729C10BFB884A33FA2379B3E364AD33FC64373BF163F9E3E27E38E3F877BC63F10C6A03D86720B40BC0535C08CCEA33FFCD0A5BFE396A33F0AAB0DC0403EC63F539B20C0C80C1A3F557E1440046D923F7E5B84BC337DB53E3ECF9A3FB9F72F3FA9EA6CBEC4E5FA3F15D9003F4A14D73F78448FBEC716CF3EC51F00C024D5EC3F2784ECBFCF9D553FF03FC8BF051BB03F3A0AD0BFC87D3A3F7EB5643F8752B53FAA811FC0B487E23F781FBF3F0D62993F82EB6E3E2BD11ABE2B245E3FECE519BE23FB0C403894CC3F1A22FB3F0A4F9D3FD4262C40B451F23FED5CB63F078EB63F34FD0C402008DB3F5A7116C0BDBAC63FE704FEBE8C19D53F663639405F6BE53FD00E8EBFFC83D63F12566E3FCA6FEF3EEE659F3F8DAEEC3FFF9D3AC02C30D93F1241CE3EF207A53F0070F2BF6742EB3FAD5E1FC0EA0ED73F6345703FBFA3E5BF18E70E40B2E5B63FEACBC43F9A09E23FABD1A43F0431A43FDE55BEBDFE9C4A3F2B5418C07A63463F0DE82AC06F97A63F00201B4089A5AB3F3C3C483F054B643F207B9F38F600993FD861723F6B3F5FBEEFF247409BEE064081B10C3F4C1C81BE58394FBF1326CE3F304022C0F56C133DDE51B33F6FB801401F1082BF0E1FE33FFA4F1A40F39E4B3F5A8D45400CE5BA3F8BA1663F59D5953FBB742940349D07406B3EC3BFC4BFB63F3163DA3FCCF7883FBB154040A4C59F3F4A0BA23F42B0853F71D791BF6A3A0340BE07F5BF96660640DF68C63F6AFF86BF472F9D3F5A60E13F7811D03F5A69F23F95182B40860E803F2E892CBF5350B23F18B55FBE2FFE4F3F0D61BD3E0EA2883FFB69DDBFD8A77E3F1CFC78BF5D532F3FEB9A82BF85B6A23EB23E4040E958C63F2F3AC33E03FC8C3F3BC930BFCBBDBA3F53F3D83C95EDCA3F12A05CBFCB268C3F3D59123E25280E3F6D2638C01C65E0BEB76E2F40EA54943FEBEC27BE446A133F982FAA3F15ED833F7D6BC13F104E873FE5B1BE3FE739B53F30F55F3EAFC7803F388ABE3FCF4DC03F9A2D2840B970B23FC699DFBF4AB5EC3FDD9BDB3E503CDABE355326BD15AB0040A816AEBF7F531C3F3C8D0E40DE55DB3F27A429BEB0E3833FAF8C833E41EEE83FEFF1053F04A9C83FFDD1EBBF7562A03FD354853DD8E6044094571FC020E5CF3F4817A63E929F06404FBDA8BEA7E02A3F00742A40DAD88E3FFE4D37C0EB9EC13F1E70133EC49F77BEF686863F2B4ED33F54D7CC3FAA28B13E6D18D1BFDD65983F5A3A2B40C6C0E3BE3FAE314020E2D53F663325C0B3FAFC3F8A5B30C0C8A2623F07D9423FFCE5CA3F37CF453F3A28903F29A2F63F28305C3FE92137BF1613A63F44F5E5BF"> : tensor<20x20xcomplex<f32>>
    return %0 : tensor<20x20xcomplex<f32>>
  }
}
