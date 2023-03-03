// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.constant dense<-1.000000e+00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<20x20xf16>
    %4 = stablehlo.compare  NE, %0, %3,  FLOAT : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<20x20xi1>
    %5 = stablehlo.multiply %0, %0 : tensor<20x20xf16>
    %6 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f16>) -> tensor<20x20xf16>
    %8 = stablehlo.subtract %7, %5 : tensor<20x20xf16>
    %9 = stablehlo.sqrt %8 : tensor<20x20xf16>
    %10 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f16>) -> tensor<20x20xf16>
    %12 = stablehlo.add %11, %0 : tensor<20x20xf16>
    %13 = stablehlo.atan2 %9, %12 : tensor<20x20xf16>
    %14 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f16>) -> tensor<20x20xf16>
    %16 = stablehlo.multiply %15, %13 : tensor<20x20xf16>
    %17 = stablehlo.constant dense<3.140630e+00> : tensor<f16>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f16>) -> tensor<20x20xf16>
    %19 = stablehlo.select %4, %16, %18 : tensor<20x20xi1>, tensor<20x20xf16>
    %20 = stablehlo.custom_call @check.eq(%19, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %20 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x87356CC22FBBC6C3F9C39EBFFF3D63C2013E26C315C4E93BFF361CC2DA3E494107432B46E2C47E3CEA3E2231143B14C1E34221C54BBD174439C3D62AD525113D33B94935C6B70BC597BF3736D13B2C3D6ABEF2C343C6FAB73D465FBFA6461EC33446CE40413EF6C36ABD60C1743A67BC8340F8C03EB542C3AAC05EBC603D2844CC42BDBCF643ADC490B95548DE24EAC0A441473DF93DA6C12CC7D5C4BF3C0FBE05C7783A54B9D0BD0AC156B886BD4645113A6746923B55C0FBB97DBAA3B93B460C40D0B704418145E73F67456CBE72BF37BED140D8B444C0D83AF9C2144289BA41B182C2573378C4A0BC0D35104538C4EE453235A8367EC45DBF0443544047C1803CE9237EBCBD4724B4DC3DDDC59AC12640A640F23D6AC35039A23949C082C518453FC164BD90C63F4274347FC2B04131C4D2BD52C4E0BC8E3D2EC4F245844543AC5FC40A3B9CBF91C4D23E41BC9440354180B623BD1E3CD6BCF33D73C190BCC23D55C1523F6E3D2ABED6BF4B40C93A6844B03C7BC53DBCAAC4C9A8453E5B40A4C2013CD8C23A4015415EB57EBEC2C4E44037C847B69FC02442DE408EC28F2E83BE11C155B87BC0A3B5204492BF593A78BFDCC1FD3DBDB7D0B75C44CC465040FB393F3B4E3C28C006C5ADC40EC074441DC55840AC3CD5BE36403E4176B11E4159BF7A3CC3C6E7BFD9BC453F7AC49742A7C0D13B074463C18BBDD54247C43BC5513EC9C6BCB26B3EA9BCB0B683300A41A743B3BECFC08BC73B3D2A465441A3434BC078C6A2C1D9C76AC48E358443173E60B2DB393941E13E79B4E1405940DC46B53C43C64F3FB8C14342C73625C1F1C307BD95420DC1883FB1C44A3D4ABB63C689B0B54006B1993F03C4ACC4BC347932AC431A40D0B282BEED44E042E3BEE5C328448341D4C2B0354FC5793C39445B48764272B82FBCAB36F3C476C0BB3C04BC9739533E413D6D4046BC55C41C4427BCD14568C49FC66FB02AC46A42AAC4363F77BCC9410144BEBBC544D6C20AC16340D7BAEA46EEBF8845D4444334DDBD94BE6E41B24428BB63BABAC22B3DE2C133C00CC57BC14F44B34472C51C34AE4483BF24BF86BE08C0C4BBE2BDDB3F213E3342A8418D3DA13C5640E7A03DBBAFBECA3C"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xDF3C00FE5F4100FE00FE00FE00FE00FE00FE00FE00FED130793C00FE00FE00FE00FE00FE00FE00FE00FEA43DC23700FE00FE00FE00FE00FE00FE113E313E00FE8F40F03C284000FE00FEB03CE03200FE00FE00FE00FE2F4000FE00FE00FE00FE00FE00FE00FE00FE00FE00FE0F3900FE00FE00FE9E3F00FE00FE00FE00FE00FE00FE00FE00FE00FEAE4000FE353E00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE08399A4000FE00FE494000FE00FEAF3900FE443500FED5400941B54000FE00FE294000FE00FE00FE00FE00FE00FE00FE00FE833F00FE5A3800FE00FE0E41F13E00FE5B3D00FE00FE003D00FE00FE00FEF63C913C00FE00FE00FE00FE00FE00FE393E00FE00FE543F00FE00FE00FE00FE00FE00FE00FEC13A513A00FE00FE00FE00FE00FE00FE00FE283D00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE8C3E00FEEB3700FE00FE00FE00FE00FE00FEF53F00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE783800FE00FE00FE00FE00FE6F3E00FE00FE00FE00FE00FE00FE00FEA73F00FE00FE00FE00FEE63F00FE00FE00FE00FEDF3D00FE00FE494000FEBA3F00FE00FE3D3900FE00FE00FE2740294000FE00FE00FECF39003700FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FEF83E00FE00FE00FE00FE00FE00FE00FE00FE00FE00FEE03200FE00FE00FE00FE00FE00FE00FE00FE223F00FE00FE0140B83D00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FEDD3C00FE00FE163FFF3900FE00FE6A3F00FE00FE00FE00FE00FE00FE00FE00FE883C00FE00FE00FE00FE00FE00FE00FE00FE6F4100FEDA3E00FEEA3E00FE00FE00FE153D783D00FE00FE243F00FE00FE00FE00FE00FE00FE00FE00FED43C00FE00FE00FE00FE00FE524000FE903C00FE00FE00FE00FE603A00FE00FE00FE00FE00FE00FE00FE00FE00FE00FED73E00FE00FE00FE00FE00FE00FE00FEC64100FE00FE00FE00FE314100FE00FE00FE00FE343D00FE00FE00FE00FE5B41FE4000FE00FE00FE00FE00FE00FE00FE00FE00FE3E3D00FE00FE00FE00FE00FECC4100FE00FE00FE00FE00FE00FE00FE00FE533E674100FE00FE"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}
