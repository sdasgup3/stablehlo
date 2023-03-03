// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xi1>
    %2 = stablehlo.is_finite %0 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xFC83F23F4690A5BF8D3177C03E034EC0CAA7F0BF9029CFBECBE25ABF1C6BDFBF5D9B57C05DCEE8BEDA15063F008FA0BDC50BB7C0CBD542BF507D50C0951889BD8B70D63FAE4B0BC0C9502740BCE403C0CB8B1D4117A9923C82D640BF0AB0CF3F5B4DA2C02439D6C051B110403DFCBF3F587BA5BF8A2A8D40793C5FC07E8594C01FF79A403E4E14C04B02BB3F09FFC6BED74DB5BF7BA69D40179081C0337E4D3F9FBE2B3FD8ECF33F94ED3C405CBC8240CE18B2C0026617BFAD2D8CC0CB82C7BF8E3593C05293903F0F736EBFF0388240E4E83CC0BF1601C17A57DFBFED69EFBEB381D7BF5B31C53F0CCF3A406CD584C07EB5B2C05AA632BF478FF23F05D0B8BF97BC873FC7F1923FBCE4BE4076A57DC0600C733FD7E308C0B1097D40632C9ABE7BFE253F180F033F7BEA993E4E35503E9E0B453F0CD87240FC8DA1BF6D859E3F45EEAD3FD3C68240FAA3523FE77DA1BFB2BF2AC0BD08B240A54FD940C9320940B96BB640C3FDBF3B69286840986BCFBF0A3096BFCF683BBF4557D23F2F7740402CCB14C0CBC490C0BB3E47C0BA0FD5BF9CF739407EE42D40B8D0BABD9F97C5BF5E2D71C02EAF1DC0FBED773F5D8932C0935CF14041C26040A2EAA3C0BD36953F9F7C9C3F8C6423403A56CB3FEB7DD83DD1C36840EF0F7ABFA006C73F574DCAC0EDC48ABFECC39EBFB5C336409DECBA3FF17F523F9EEA84C09E5F9EC072C778C0B13B6FC03994D7BFC2A3C7BEAC650ABFEF1F65BE287104C02EE174C035F3714002219C3F5F86EABFAD59DFBE8A31C0BE5DEF80C03AF51B40C05C5EBE09EC6F40BACB3E4032AAE24043E289BF27AD5A400DBB423F8C16A6BF56C6954083CECB3E7CEFE53F3CB8B2405F2EF33F8630233F92FF9F40E88F0BC0CB5C1940639DB9C0C8943E409AAFF3BEC511B6BFDF8254BFF18A02BF53A00BC0B2AE35C023256FC0714289BFA267613F22C5BBBFECBC31BF668EB93F05E252C08B7A6C3DD2840641ABBC4C3EC864D2C0A884E83FE5E367BF9BFBB53E0804B7BEC84A75C0D5E24140260E9C40BFDBF43FD9B551C0D6170F3F12F375C0C331A53E7B89F2C0F1F7A93F108E11406D7286C0E44DB4BF966BADBF086416C02F795EBE73F080BE7CB640BFD86309403904BD40D15745C02C055F40F80F48C0ABFAB540B90F333FDAA3D73F9CA95940087F3140A9394DC0C1E99D40C627E03F2936E4BF56223FBFDE692DBFB67DA8C0BBE59ABF1436D83E94F8AFBF7FDDC03F5C5B8C3F1907D940BDB6E33FB97174C0B5C18040669CB23FD0FC7F4051D985BF8D2C6A4023230EC0C3C60240A0A69A3F7621DB40CA083640C7881EC010189DC099312540AEC8F33FEF0264BF62F79A3F360A99BDAA31B5BF52EF444095F59C4002A28A3F2E4573C008E85FC091FB9EBF32707B3FA17C8D3F1A33FFBE9B66A53FC1D2694066E20E40E22C94C0D3098DC0E07702404EE053BF35A974407F438F40C0C91F402445F640C52F81BFEC7261404F1762C0076393C0137B3440C0C1843F296B9C40E1D52B407F9E62C0AFAC41C04F81A83F1A912C3F56CFFABFD23DB03FCBE63A3FACFCDEC031905D401E94943FCD8C5440F84C8840DA8CEB3F78AF6B3E8404FF3FFAA58ABFF0772DC0539E713E766ACD3F536001401BC3F6BF86F273BFBCEE2640F65794BE2A8C39404AD3B74069CA2F40DF544AC08F881A3FB4F3B53F5A61373F307F0740CB3D9DC03161A23F56039F404672684058EE08BFF6DD2EC0AAA8B43FF804CA3E02D83DC00622FEBF0BABE2BFC626A1BED335CFBFDD653DC05E10C73FD2598040B0961B3F96B1C9BFD056AA4088D12A40D8A9343F92E322C016B11640FC7914BFDBD4923D5A681F40F75B9A4023893EC04F9904BF845D65C0F0C4EBBFF2997EC0AF300140E6034940CEB8833F3C798BBFD9AF1C404A33DC3F1647A8BD1FAAD33F852B52BF93335E40387A64C00F54D6BF75E98B3E866E843E228DC2BFA32A3EBE6A4E9040568328BF5FE2AC3F1A4C58C0F025A8BF0C60DC4041940D3FD1C436401CB6A0BFB456BCC08ADEFBC0D48BD7C078E10FC1FCDD6EC0BD744B40B5D0A3C0A022544046FB4C40344C5CC0535E97BCED96D1BFDEF82EC060EDD8BF13747CC0EDC894BE40BF393F2F3196BFCDF48DBF90C22B3DEE36AA3FE544BC40139AA540C153E3403E49D33FC5291D409F4E9C404421FCBE2805C43F68839CC0986FE0C0080607BF68321A3F52AF3CC0CEF5BE3F350FBE3EAEE0A2BD989823C0F8FC28C0509DE63F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xi1> {
    %0 = stablehlo.constant dense<true> : tensor<20x20xi1>
    return %0 : tensor<20x20xi1>
  }
}
