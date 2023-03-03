// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xbf16>
    %1 = call @expected() : () -> tensor<20x30xbf16>
    %2 = call @integer_pow(%0) : (tensor<20x30xbf16>) -> tensor<20x30xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x30xbf16>, tensor<20x30xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xbf16> {
    %0 = stablehlo.constant dense<"0xDF3EF1BFC740FCBD843F9C402E40794067BFB13FEAC0A940D93BAABFDBBF813FDBBF18C0863F2B3F94405840DE3F34409EBF34C058C0C4BFD0BFA1BE6540B140BE3F76408A3D3CBF98BD5040544045BFAF3F66C09FBF80C01340B6BEA03E943F03C0053F34401D40193F3C3FCF4057C0F6BFDEBF11BFDFBF04BF5640B84067C01BBE3540B4408C3E66408F3F043FCC401CC01540BEBF813F4E3FF1BEC63F7C40C03E9CC00ABFB2C08C403CC0193FD7BD6FC01A40E4BE00C04B3F77BEECBF76BEFEBF0E401DC01DBF1E3F2C40E8BF95C07C40C3BF623F8DC002C0CA3F26C0C9C0A1BF13C0BC3F7840834090BFFBBE0F40C1BE07BF4DC04D403DBFB240E9C051BF14BF934032C08B40004061C0C1405E40BCC041C01BC087403EC086BF23C0EFBE65BF40406C3F4E3F533F89C01340024091BF8A3E16BF673F4C402040283E44BFB73F9D40A1BFB33F3C3F04C08FC05EC058C08B4036C04440964058C0F2BF8F3D45BFB9BF72BFBDC060C052C00740BFBEB040C73EBCBF8C40403F084005C0B23FC53E323EBB40123E70BDB73FB7BF8C40024198C0EABF0540433FC5BEAE4012C053BFCA3FD33F733F74C0A1BFE1BF5D40813E31BF48BFC1C0843F52C0344082BF464027C01B4031C0BBBDED4041406D3F3FC03C40C43F76BFD43FF7BFF24015407F3E41C02540E0C0B940513FC33FD6BF573FB540BD3F5540E4BD64BE344079BEF63F663F4FBEDB40E4BF1D3F794088BF39C0E0BF863EE1BF993E2640473F36C09BBF5E408A3E1340A74073BEB53FC04067C06DBFCBBFB3BEAC405AC0D7BE65C0933FC0BF6AC0AF407E40BABFE93F804042C0A5C0B3C0174069C03BC09140D0BD04BF4B40D1BF2B4014C0883F24BF93BF68C039C055C06F3F803FCFC0963F503F35C04C40BF3F18BF17C051BF183F6EC0643E964046404640FC3F014035407FBF2B4051405D4093BFA640A1C0A1C086408C3F3DBF38400F40DCBCAFBF2DC00D4063406040F73F904041BFA7BD33406A40F9BF5B3F84C088BEB53FEA3F6CBFDBBEA0403DC081BF25404FBF90BFC9BF2840574065C0A1BE0040BF40063F2CC0D03F4D404E4062C0EBBFDCC040C08640923E09C02540004090BF1940373FE1C0F34058C017407ABF073EF1BFBF3F173F323F9B3F50C068C08DBF8AC0923FBDBF3BC0E0C0524091C06D3F7AC03EBFF53E66BFB13F9DC087C08B3F3ABE3F40CBBE783E97BFB1BFA8C0B03EF33F833F8BC0C83F643F9C407B40903F0840C7409840244063C0674095BF68BFDC403EBF0DC0833F7BBF213D65C0A03FAC3FB8BE43C04840B33E54C04DBEC2BF38402140E9BE33BF83C09E40F7BF8B3E8ABFB73F8940D64004C09B3F22C086C007C0F13F8CC0983EC1BF3640C2BF10408C4006C08AC02FC0613F113F903EC53E8B4056C00F40973D593FF53F19C0A7405A40EFBF5A4094C09ABF97BF9BBF25C01A40B540FC40CD40AFBF3B3F0AC0F93F8AC092BF1A40C6BF243F2340A2BF8340CBBF88BF363F09C0ECBF5B400CC06DC0A93F08C006C1BABF00BF11408A3E1A4011407340733F6F4038C0EEC090BF90C0F740E9BFEBBE48C01F3FAAC0134025C0A540C540D53E5A3D16C093BF804020C0C83FC33FED401F3F9840A94098BF8EC09D3F5FC0143F3A3F264030C003409CBF573FA2C0B6BF2F404A40D0BE33C02C4002407C3E76C0B53E"> : tensor<20x30xbf16>
    return %0 : tensor<20x30xbf16>
  }
  func.func private @expected() -> tensor<20x30xbf16> {
    %0 = stablehlo.constant dense<"0x000079F9807F00802842807F807F807FF8B5285D80FF807F00008ED9ABF02340ABF080FF9443711A807F807F2172807FCBD280FF80FF88E6EFEB0080807F807F9E63807F000029A30080807F807F8AA70E5C80FF76D380FF807F00800000B04C80FF6F03807F807F5B102923807F80FF4AFB21F2478B42F2A882807F807F80FF0080807F807F0000807FAA49A802807F80FF807F9EE323409F2B00806D67807F000080FFE38680FF807F80FF5B10008080FF807F008000FF492A0080A1F7008039FE807F80FFE6924B13807FD7F580FF807F20E61C3480FF80FF326980FF80FFA0D480FFA962807F807F3ACA0080807F0080CB8480FF807FCFA3807F80FFF8AC308D807F80FF807F007F80FF807F807F80FF80FF80FF807F80FF94C380FF00803DB5807F21389F2BCB2D80FF807F807FC7CA0000928EF835807F807F000008A73B60807FA0D4245E292380FF80FF80FF80FF807F80FF807F807F80FFDDF900008AA742E15DBA80FF80FF80FF807F0080807F0000A9E2807F1B25807F80FFBB5D00000000807F000000803B603BE0807F807F80FFC0F6807FA0260080807F80FFCBAD32694B6DB23A80FFA0D42EF3807F0000A89DBFA880FF284280FF807FD0C0807F80FF807F80FF0080807F807F503880FF807F8866CABBD06DA1FB807F807F000080FF807F80FF807FF82C206690EE982F807F4F63807F00800080807F00804A7BAF350080807F64F4E612807FDBC480FFBBF200002EF30000807F892880FF13D1807F0000807F807F0080355F807F80FF50B8C9E90080807F80FF008080FF294C9BE480FF807F807FA0E16776807F80FF80FF80FF807F80FF80FF807F0080A882807F78EC807F80FFDB44989629CC80FF80FF80FF1E39803F80FF124E6F2C80FF807F5064938F80FFF8AC930F80FF0000807F807F807F847D807F807F1ABF807F807F807F29CC807F80FF80FF807FA747CFA3807F807F00800EDC80FF807F807F807FA17B807FD1A50080807F807F6CFC2B3180FF0080355FC07621B80080807F80FF23C0807FF1AB3ACAE3E8807F807F80FF0080007F807F140480FFEF6B807F807F80FF1DF780FF80FF807F000080FF807F007F3ACA807FBB2080FF807F80FF807F51BD000079F950641A0F3B1E135180FF80FF42C880FFAA4B4FE380FF80FF807F80FF503880FF1EA40000AFB5285D80FF80FF0D470080807F008000009ACE28DD80FF0000327A894180FF3F68E434807F807F3A4A807F807F807F807F80FF807F48CD57B6807F1EA480FF8941A0BD000080FFDE537D5A008080FF807F000080FF008093E5807F807F0080A49E80FF807FA1FB000063C63B60807F807F80FF135180FF80FF80FF797980FF000051E5807F93E5807F807F80FF80FF80FFAE33470B00000000807F80FF807F00004330DE7A80FF807F807F9EF8807F80FF5ED09ACE13D180FF807F807F807F807F0EDCE62280FF6C7C80FFAACB807F6DE79816807F07D5807FC9E9DBC4122080FFA1F7807F80FF80FFDF5880FF80FFA0E10080807F0000807F807F807FB23A807F80FF80FF3ACA80FF807F67F6008080FFF61380FF807F80FF807F807F0000000080FF29CC807F80FF3F682066807FF613807F807F13CF80FF665280FF300D2022807F80FF807F8FD1982F80FF92DF807F807F008080FF807F807F000080FF0000"> : tensor<20x30xbf16>
    return %0 : tensor<20x30xbf16>
  }
  func.func private @integer_pow(%arg0: tensor<20x30xbf16>) -> tensor<20x30xbf16> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<20x30xbf16>
    %1 = stablehlo.multiply %arg0, %0 : tensor<20x30xbf16>
    %2 = stablehlo.multiply %0, %0 : tensor<20x30xbf16>
    %3 = stablehlo.multiply %1, %2 : tensor<20x30xbf16>
    %4 = stablehlo.multiply %2, %2 : tensor<20x30xbf16>
    %5 = stablehlo.multiply %3, %4 : tensor<20x30xbf16>
    %6 = stablehlo.multiply %4, %4 : tensor<20x30xbf16>
    %7 = stablehlo.multiply %5, %6 : tensor<20x30xbf16>
    %8 = stablehlo.multiply %6, %6 : tensor<20x30xbf16>
    %9 = stablehlo.multiply %7, %8 : tensor<20x30xbf16>
    %10 = stablehlo.multiply %8, %8 : tensor<20x30xbf16>
    %11 = stablehlo.multiply %9, %10 : tensor<20x30xbf16>
    return %11 : tensor<20x30xbf16>
  }
}
