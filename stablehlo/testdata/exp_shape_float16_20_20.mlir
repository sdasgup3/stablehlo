// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.exponential %0 : tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xD83F794384BDD63CD4BA3B42F93CA7C503BC96C54BBE2E3A36B9AAC233C1B5C15CC2B1C034C2EF45113E01413A403CC42B4305C2763525B548BFF241DEC028C4CCBA9F3C5C38B341C4B42FC0D641E5BA23BD98C025C45F44BE431138E4C335C8904029BEF641B63D843AC2C1743DE63E01C0D93A16BCE4C167C3A032F035B33F0544E73DBBB4BC38CCB8744204376D40A9C27837113DD331D4C49C41E6445B438745FDC2FAB827BEF241CE3A7D40DB40014067BFAFC7A647A43FC5B9653CBF387B4412405B43C7451AC152453BBDF842033C05B4DEBC28C50A363C403439C6B89C433FBDB144C4C0B244CF3E22BF3D3C94C415B66FB9403F35C50A40F3382DC0303FE3448EC19B401C3AEDBCFEC324B4E1456F43BA3A27C02FBC56A1763EEC3CE5C0BDC172C0D0C03CBC064022401EC0CD3D6D4444C5D3C59B3713C71F3C1A40CABDF73E354247356F422DC456401B4051C4613F5D46BA3A5F38EB43F4BA25C4A2416EC325BD4A404F4395C2213FE141073902C5EDC434BF1DBC3944DEC03D3F9B432ABFBC36453CB0C4FF4078409CB896C57A3CF7354A3DC52B164119426EB5C0B947466BB239C10BC13BBCF3381F41D5423BBC314018BECCBC3DC3F1C016C642C0B0C4AEAA0DC508BF8F43543EA2C410418A3797BB763F79BF39C422C8A74268B0CBC40435D6BAC13363C2A9BF9242DB3E72C27BBD62BFD8437F378A366AC2864184C020A6DEC05DBE9B4062AECCC06D45EEB5EEC396C54BC551BC51C3783C0ABD884773BA01B5A842A73FBF4124C8C7437BC2D43D14B9A941B73F4BBFF0BEABBDD6C44543493CA94338AEA2C3B53EA6217FC0463C8E3CD23C1442F3BB77C03F3EE8C0FEBE76B881C18A45EA41D83C8BBA18B852410BC25F3CBBAF9CC5A8B995C1BE3C4D3DB838A9C5B7BC363F21A45E46AC4151C121C532C4EDC020C27B3DC23E9D45073AA7C28BBF01C74B40DE4029B72B4187BC333503BDFEC239B7223C29C08EC485BCE6C20A3C19444A3FB441D7C16AC40E3FBC4371C59BC36D4483BCDF3E643FCAC472C080373C3BCDBF3D37B0B889C31335B03AD04300B52BC5E6BA94C45FC2D33E34381E42BD3E2BBFC0B52246473B47BCBBB8B7BF9ABE233DD044"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x1B473E510834B342D136A24DEF42301BDE35AE1BA33255402C389228C12C612B5329212EC229E65D8F441A4A23486B238150502AA13DCD392F31E34C9D2D0324D7365A42E63E524CF039E72FA04CC2366E34702E0F24F2540052A63EF424440BE548DC32ED4C2B448440312BD1439C455330B540C335BB2A5226EC3CCC3DDB46F6526044F4393B3F64384C4E343E92489528613E1943CC3C1920214C3158F250DC5BC7274B38E032E34CAF40B748AB49674707318B0F1968C146C73700423D3F8455A747F2500C5DFE2C645A543413507441393ABD34E71DD63D2748AA3F68389D515034D056E82DD7567C456131C541432178390E3820469C1D89476D3FEF2F08462458F62B00494B40AC34B5242D3A965D2451A34003309F35EB3B0745D8428A2D442BEF2EC52D8D357A47E5471630444439554A1D0D1A6F3EF0129B41C6478733B445924D903D3D4EDE235F48CA47D62254468860A340E83E8D52B6360F242E4C3C266C344548D450C328F145BA4C803FD91E6E1F4931B93543549D2D1C469A515631183ED141B720144AAC487F38AE1B2042CF3D8143403C5C4A464DB339CC3729608C3AB32C242D8E356D3F794A9D4F8E351148FA32D334DD26692DA9189D2FB720983B8F1E84317951DD44FB20494A683E32367546F1308123370CF54EF93A3E20793DCF36193D4129B730AE4E8D451A2911340E315052643E053E2E29EA4BB12ED03B9D2D853200493E3BD12D1A5B8639DB24AE1B261D703599261D428A3449672537DA39F94EC6466C4C260C1B5203294B443E383D4CE1462B31A631C2331120BD50D741C251423BA22559450B3CC22ED2413F42AD42394DEC35DD2EC444812D92319538152CF45BCF4CB7421037CC38264B3D2AF741173B811BE437DB2B8C428743373F221BEC341146DF3B8D60434C7C2C111EB723742DFD29DF436B45485C40409928DB3071134748B3491D39A04A2935893D9234C32718399F41FF2F63212B3511287D4186533046544CE72A3322D545FA51701CB62539552E35924558464320EF2E643EF1408D304A3E7438EA257E3D9D403752DA39D51DC13643214B298245C43E534D644555319639335FF8407E356E38A73025323943B157"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}
