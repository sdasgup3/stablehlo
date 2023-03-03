// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<-4.000000e+00> : tensor<20x20xf32>
    %3 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.clamp %2, %0, %3 : tensor<20x20xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<20x20xf32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %7 = stablehlo.multiply %6, %5 : tensor<20x20xf32>
    %8 = stablehlo.constant dense<-2.72614237E-10> : tensor<20x20xf32>
    %9 = stablehlo.add %7, %8 : tensor<20x20xf32>
    %10 = stablehlo.multiply %9, %5 : tensor<20x20xf32>
    %11 = stablehlo.constant dense<2.77068146E-8> : tensor<20x20xf32>
    %12 = stablehlo.add %10, %11 : tensor<20x20xf32>
    %13 = stablehlo.multiply %12, %5 : tensor<20x20xf32>
    %14 = stablehlo.constant dense<-2.10102394E-6> : tensor<20x20xf32>
    %15 = stablehlo.add %13, %14 : tensor<20x20xf32>
    %16 = stablehlo.multiply %15, %5 : tensor<20x20xf32>
    %17 = stablehlo.constant dense<-5.69250624E-5> : tensor<20x20xf32>
    %18 = stablehlo.add %16, %17 : tensor<20x20xf32>
    %19 = stablehlo.multiply %18, %5 : tensor<20x20xf32>
    %20 = stablehlo.constant dense<-7.34990637E-4> : tensor<20x20xf32>
    %21 = stablehlo.add %19, %20 : tensor<20x20xf32>
    %22 = stablehlo.multiply %21, %5 : tensor<20x20xf32>
    %23 = stablehlo.constant dense<-2.954600e-03> : tensor<20x20xf32>
    %24 = stablehlo.add %22, %23 : tensor<20x20xf32>
    %25 = stablehlo.multiply %24, %5 : tensor<20x20xf32>
    %26 = stablehlo.constant dense<-0.0160960332> : tensor<20x20xf32>
    %27 = stablehlo.add %25, %26 : tensor<20x20xf32>
    %28 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %29 = stablehlo.multiply %28, %5 : tensor<20x20xf32>
    %30 = stablehlo.constant dense<-1.45660715E-5> : tensor<20x20xf32>
    %31 = stablehlo.add %29, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %31, %5 : tensor<20x20xf32>
    %33 = stablehlo.constant dense<-2.13374049E-4> : tensor<20x20xf32>
    %34 = stablehlo.add %32, %33 : tensor<20x20xf32>
    %35 = stablehlo.multiply %34, %5 : tensor<20x20xf32>
    %36 = stablehlo.constant dense<-0.00168282702> : tensor<20x20xf32>
    %37 = stablehlo.add %35, %36 : tensor<20x20xf32>
    %38 = stablehlo.multiply %37, %5 : tensor<20x20xf32>
    %39 = stablehlo.constant dense<-0.00737332925> : tensor<20x20xf32>
    %40 = stablehlo.add %38, %39 : tensor<20x20xf32>
    %41 = stablehlo.multiply %40, %5 : tensor<20x20xf32>
    %42 = stablehlo.constant dense<-0.0142647391> : tensor<20x20xf32>
    %43 = stablehlo.add %41, %42 : tensor<20x20xf32>
    %44 = stablehlo.multiply %4, %27 : tensor<20x20xf32>
    %45 = stablehlo.divide %44, %43 : tensor<20x20xf32>
    %46 = stablehlo.custom_call @check.eq(%45, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %46 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x74A50F406FB30240DE5A0DC09D5C8EC0F07968BF06565B40463FE7406EDED73F4B3D503B834A33BF757F2B40EE0A7440CBEFB7BF3A62FD3F518E2DBF1205BDC00C6DEE3FA85A95C08FFED5C0D5F458C096EE33C0D54DE4BF04F8F0BE5DCFA43FB7B9BFC0B29A04C0D77AF1BF9470A03F8D7E84BE456DC33D14A03840754041C076B93D4026767E404FF903C0D458D23DC71C08C08D11B540534CDBBF42F5B2C0129C664067EAC8C060CF11C0C61AD3BE518C1540960E20C0457B6D40F3C6E4BFF3BB6040613756C0E77A3E3E9021333F38C0313FCF893D40434A5B4042F5EC3F01E55B40A09065C03BEC833E5D94864022B31AC05C6809BF238084BDB5A886BFFF2162BF4932F9C02937963FA23635C06F1335C050D557C082298540A17F893D388AA63E5E2789BFB8318BC0E841033F211F20C0C9D810C08AA686BF02E484C04FEF0AC0E3CD1BC080BC9BBD4276623F1F579640DB340C400CA6A240C37D5B3FA0A988BF5AFF98BFDD15D1C0E80BB7BFEDD13B3EBB3B83C0CAA195C0D4E808C1372DABBFB7E3014076A579C00F958FC02E724CC0037691C0D250A9BE52ABACBF5D317DBE9284E5BF9E69C1C0331D18C01E702DBF000E67C0B0CD1F402EEE1D407267C040CF27D3C08E7939BFDD297440D71BB83FFBD6A5BEF54861BF78981A4056A505BF779312C076AFE2C04E5F19401CEE20BE3A0B3940E1721FBFFA8C18C00A7C9C403F4531BFD8A50BBF8EBD853F34B46CC0FD89904038880740CE108B3E3F5B07C0780AD2BF5BB2B9C0FF60ECC0E7FE9C3F4F71E13EFD0C5340ADBDDA3F266BEF3F24E50DC03D814B40C29009C164120E40C7B990C09BC42240D714833FC27322BFAA83AB3EAFB7E340B69023C0D9B94C400A717140310EA4C0E6D1973F9F49AD3DAAB23CBECB3F4040C157BBBFFC27B3BE1FB6853E55F2643D747253C0F3D6A1C08095BBC03C4139C0BCAFA9C02C428B40721B2E40A4C2F3BF1E0CF8BFC04C25C0543EC53F1E867D40446D3D402A4E90C0AD216FC052D58F3F0A9C153F396D2540ED332BC00975E8BF8C8519C0515D2AC07312FA40D09A9F3F09E034C0BD922A40CE5681BEA7D085BFAC4839C02745BF3F38BE61408ACE56C09A80C4BF860510403E8E40C038B982C0C5D52DBFB6DEC73F6AC439BEF81EA44049CC93C019132EC01372D8BF58E5773F8722B9C095266ABF99653F3EFB75F53EE41EF43F13A46C40117919409145CE3E2C48C3BF414F0540C45063BFFB2775BF3EA569C06667E53C7189913DF0F4B940FD38613F15A7323EFD977340CA1EF8BF8E0A6DBF42ED944025AA7C402797E9BF03E96BBEE33BC7404D062E4005233AC07A6DF53F7C9EEF3FE4DB91BF092CA2C0BB54123F8590A0C0546B34C02425A83F26C9904020EF0CC026F739C0CBEC40BF173A15C07C0483BF937B9240DFC12440568464BE4FB713BE7E2A98BF31AE7740EE2D874018168C405DD39C3DDE0AC03F95EBADBF3ECF824073DD4440B6988F401D9DB9400D114C40E68BF63F16C72FC0305AF2BEBF0CA8BFE54296C08B58A24090645E3E356DB340B7DD83C026297ABFACF26E40D0C758C020068D3F04B64B3FB8D2CCBDB1D7ACC0729E8E40EF39DC3F9F626E3DEFA7B7C01817E8BF0FDB81C043E7D8403ECFD2BF41B639402A30B7C00B92DA407C6BE43F7490A3C05BAE623F91AC74408628C2BFAFE9E1C0F1FA483F0DE85FC023DE3EC0CB337D406E89844050FCA540A286AFBEBF80854082617D40776858BFB6DC75BF9C867640D191733FA982FFBF5D3E5EC0BF2686BE1620A13F80DF3CBFC546C9C01D2623400A3359C00FC29140C41BB3BE88019D40B421994090DA3B3FCC3E0BC0BABDB040C656B5BF821E9140FF4D8A405C5CEF3F2EF715C0283DA83EEE2A393F429667C01099A1BF4E94DE3F9EAF293FE7E061BFFDE00C40CF584DC0184120C0328CF4BF716C6EBF1FB39CBF0D8CEEBFED58A43E84B223BF00A2C43F5ED28F407B801DC08EAD4CC0E549373F8E9475C0A34BE3BFE2E26F40BE9BCE3FB613E2BFA397703F632659C0B7A1733FF67E0EC0A24E53402A729F3FB7DC53BF6FC0B23F60C2B2BF0D6B2240FACB1140B22BBC3F4B54603FBB46173EE3DA1840119E9F402FFAE93F21CF11C0334695BF91241CC0AA59EEBF73C73CBFD5C72340B66AA5C00E33FA3D0A5A7CC0437EA73F2FF425409F22663E654453C03E9A3DBEF8CC6640DD2E15C0B5295EBEE95417BFE7D399C05C1A193FA8775CC0A7EEFABF"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x859D7F3F02027F3FE48A7FBF000080BF0C0B4DBFEAFF7F3F0000803FCFA07B3FE4F86A3B7E942DBF1EF67F3FFDFF7F3FF43675BF99B07E3F288E29BF000080BF66D77D3F000080BF000080BFE3FF7FBF69FB7FBF32047DBF4B18FDBEE46E6E3F000080BFFC217FBFEF0B7EBF37786C3F8C3B92BE1BD9DB3D0DFD7F3FBAFE7FBF30FE7F3F0000803FD4177FBFE884EC3D7C537FBF0000803FFE0E7CBF000080BFFBFF7F3F000080BF91AC7FBF735EE1BEA9C17F3F73E57FBFFDFF7F3F790F7DBFF5FF7F3FD9FF7FBFF07A543E2F782D3FEA822C3F28FE7F3FEBFF7F3F6BBC7D3FEBFF7F3FFBFF7FBF18A1913E0000803FB5D67FBF905C0DBF704D95BD31FA5CBF91D549BF000080BF2A2C673FEEFB7FBFE0FB7FBFE0FF7FBF0000803FEEEA9A3D747FB53E0ACD5EBF000080BF9F17083F98E57FBF29A67FBF93F85CBF000080BFBE737FBF46DA7FBF3964AFBD16014A3F0000803F62807F3F0000803F1952463FA5725EBF66B768BF000080BF15F574BF5194513E000080BF000080BF000080BF340071BF27F37E3FFEFF7FBF000080BF97FF7FBF000080BF804FB8BE2B8E71BF82FD8BBED3207DBF000080BF28CD7FBFA07829BFFDFF7FBFE3E47F3F52E07F3F0000803F000080BFC1C731BFFEFF7F3F8C43753F6EC9B4BE016549BF5ED67F3F82270ABF5CB17FBF000080BF0AD27F3F2E1B34BE2AFD7F3FD3201FBFE7CE7FBF0000803F1E2D2CBF693F0FBF68495C3FFDFF7FBF0000803F104C7F3FD624993EC3497FBF3ACD7ABF000080BF000080BF8ECC6A3FC3DBEE3ECCFF7F3FC5FD7B3FFFE87D3F8D8F7FBF8BFF7F3F000080BF09917F3F000080BFE1EA7F3F833A5A3F226921BF0088BA3E0000803F42EC7FBF9BFF7F3FFFFF7F3F000080BF9712683F8611C33D7B8952BE97FE7F3F282776BF2C33C2BE2484933EEE08813DCDFF7FBF000080BF000080BF3BFD7FBF000080BF0000803F2CF87F3F34307EBF0B6E7EBFFEEE7FBFF27E783FFEFF7F3F22FE7F3F000080BFFFFF7FBF3652633F986A173F2DEF7F3FD9F57FBF42627DBF95D27FBF11F57FBF0000803F23136C3FCCFB7FBF44F57F3FCBE58EBEDC575CBF3AFD7FBFD525773FF7FF7F3FDCFF7FBFC05678BF4BA07F3FA1FE7FBF000080BF0AC129BFC207793FC0564FBE0000803F000080BF25F87FBF0DB47BBF7C42543F000080BFD3DD4DBFB87A553EE191003FB4357E3FFFFF7F3F6AD27F3F1CC0DC3EF91278BFDD2C7F3F7F714ABF670953BFFCFF7FBFB964013DDBF1A33D0000803FB15C493FA98F473EFFFF7F3F076F7EBFCB434FBF0000803FFEFF7F3F117A7DBFC6C782BE0000803F1FF87F3F71FD7FBF35497E3F7CEC7D3F6D9764BF000080BF84C4143F000080BF9EFB7FBFE8D16F3F0000803F23877FBF68FD7FBF20A636BF0FC07FBF912D5ABF0000803F2FEE7F3FF1A27DBEDF8725BE664368BFFDFF7F3F0000803F0000803F0E9DB03D3F55773FC70172BF0000803F1DFF7F3F0000803F0000803F92FF7F3F4C597E3F45F97FBF1E58FEBE17C86FBF000080BF0000803F9D0D773E0000803F000080BF744055BFFDFF7F3FE3FF7FBFA07B613FE3533D3F9059E6BD000080BF0000803F212B7C3FBF57863D000080BF625A7DBF000080BF0000803FFDEA7ABF55FD7F3F000080BF0000803FF8067D3F000080BF031E4A3FFDFF7F3FCED277BF000080BF8DAD3B3FF3FF7FBF60FE7FBF0000803F0000803F0000803F1191BEBE0000803F0000803FA8A244BFB35A53BFFEFF7F3F2451523F3AC87EBFF4FF7FBFD2FA93BEF9C96C3FE90634BF000080BF8DEB7F3FE5FF7FBF0000803FF826C2BE0000803F0000803F905B333FED767FBF0000803FFF7274BF0000803F0000803FFDE77D3FADC37FBF7A38B73E3993313FFBFF7FBFB0016DBF5E6F7C3FF5C4263FDFB349BFA3867F3FA0FF7FBFE3E57FBF2A3C7EBF60EC4FBF74A66ABF91D97DBFE644B33ED95822BFE05D783F0000803F2DDF7FBF98FF7FBFD64F303FFEFF7FBFA6EB7CBFFFFF7F3FED407A3F11CD7CBF72F1503FE3FF7FBF6358523F8A947FBFCAFF7F3FBBFF6B3F5A1642BF4CA4733FEBA473BF3DEA7F3F7BAC7F3FC75E763F52E5483F9B76293E16D07F3F0000803F04827D3F90AC7FBFCAA166BF4DDB7FBF08D67DBF28F733BF9FEC7F3F000080BFF6750C3EFEFF7FBF668E6F3FECEF7F3F8C5F7F3ECBFF7FBF068653BEFBFF7F3FD7BF7FBF44CE76BEBECA18BF000080BF07321A3FEEFF7FBF5E937EBF"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}

