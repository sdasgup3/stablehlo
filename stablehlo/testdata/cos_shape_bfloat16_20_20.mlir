// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.cosine %0 : tensor<20x20xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x834027C05D40BB40FE3F1140413FC03E00BE1A40E04089BF57BFA1403FC023C0903EE13E5C3FB5BFE8BF124009415740CC3F8EBF354083BF983F323F1F3F2FC021C04340FCBFCABF7DBF8C3FDCBDEDBE5D40CC3EDB3F143E6C40BD3F33C080C019C0AF3F633F8140164098BF113FB1C0053F144006400EC01B40EB3FC83F543F9C40813FF0406EC05FC08EC032C09BBFB94040C0943F08BE7D40A940BD3F6940CBBE50BF84C0C34005BD0CC081C0B43D5EBF443EFB3F47C013C070C0D6C031C0DD3F793F423FA0402740864033C0404083403BBEB93F93408340F23E8040CCBB35BF93C073C064C0A4BF04C013C0BA3F03C030C00C401B4095404EC0A840B43F2840BA406CC04DBF2AC04EC007C036C0313E0DBDAC3EC0C07340CEC0AE3F5A3F1EC0013FE1408D3F1FC084C0D43F4240B23F8B4095BE5E407AC080BFDAC03CC09D4014C001BF8940434005C1B04040404140453FEC3D96C0FA3F324053403D40ACBF4FBF97C0CCBF9C3DBDBE224097BF4FC0B240BD3ED7408D40D43F63C017C122C014C03D40CCBFC0C02DC0BE3F8CBE94C0D73FC1C09240443F8AC085C0C3BFA5C09CBFFEBFDCBF04400FC0ECBF6240BB3FAF404FBD13C062BFD84058BF0AC098C08CC083C0E1BF9C3F053D01BFF9BF04403FC080BE78C059BFDDBDC1BF87BF2EC04FC01B3FEDBF3DC01AC1D53DF2C04F3F96C01C406CC003403BC01CC08D3F18402840D4BF7FC085BFCBBFF2BE16C05DBF38BF5D40F9C05CC0863D6EC0693F91C00BC042C084C03DC0C24096C03EC08D406540C13FF13F4340324050C04D4081C0B33D23403B3E02BF01C0EABFDDBF03C0A4BF27C09FBFD04003403D3F4BC058BF6240343F613F3CC01E407CC04FC08EBD333FF53FBE40F23F02C0C240DE3F11408FC073C0B240683FCEC00641923E52403240493F7F3FC0BF2DC08FBE1C3F523FB6C0EABFD83F54BE11BDD5BCADBF2AC01FC029BF7BBFB1C04EBF37BF8A3DF3BF29406D3FA1BF1E40DBC093C0613F1ABCA7BD3E3E22C030C090C017C06AC023C02DC0F1C09F3F0EBFBFC098BD71BF3040D0BFFB3FDB3E81BF064084BEAD400FC0ACBE35C0CD3F9140D240EF3E93C031C186BF6F40BDBFEB3F90C0DD3F0C3E"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x14BF5DBF74BF683FCEBE24BF3B3F6E3F7E3F3EBF413FF63E2B3FA13E7DBF54BF763F683F273F203E75BE27BF27BF7ABFBCBCE43E74BF053FBF3E453F503F6BBF4FBF7FBFC6BEF0BB0D3FEB3E7F3F653F74BF6C3F0FBE7D3F5BBFC13D71BF27BF3BBF4F3E223F21BF33BFBF3E583F3B3F5E3F2DBF00BF1BBF41BF86BE083C2D3F263E093FB13E57BF71BF8BBE70BFB43E603F7DBFCE3E7E3F30BF0A3FC13D61BF6C3F303F0EBF7B3F803F14BF21BF7F3F263F7B3FC3BE80BF2ABF52BF6B3F6EBF1FBE103F3A3F913E5DBF00BF71BF7DBF14BF7C3F003EF2BD14BF643F27BF803F433FF2BD4BBF6ABF923EF2BE2ABFF03DEBBE6DBF14BF41BF66BD7FBF033F283E5FBF643F5BBF323F62BF7FBF03BF75BF7C3F803F723F763F4BBF7D3F573E293F48BF603F3C3FE83E4BBF0EBFAFBD7EBF373EB8BE753F72BF39BF0A3F5D3F7BBF453E2DBF603FD6BE7FBFE3BE353F7DBF7EBF383F7E3FCCBCBFBE70BF7DBF7BBF673E313FD03BBCBC7F3F6F3F52BFC33E7FBF403F6F3F683F9ABEAFBD6BBF80BF52BF2DBF7BBFBCBC763F68BFB13D763FB3BDDFBD783F19BE393FC7BE07BF423DDC3EB13ECEBE17BEF2BE1EBF8ABE6DBFE13D303F803F2ABF233F653F2A3F0DBF1A3DA9BE14BF3EBEB13E803F603FBBBEF2BE7DBF783F3EBF293F7F3F813DFD3E69BF7FBF523F8EBE7BBF7BBF7F3F933E313FCCBC43BF5BBFEBBE7ABF43BFE83E38BF5FBFAFBD2ABF023F78BC643F33BF263F413F74BF953D75BF7F3F57BF1D3F38BE11BF7EBF0EBF7BBF7A3FCCBC7CBF9ABE68BF813D9DBE7FBF70BF7EBF80BF21BF7F3F54BF7C3F603FDCBE82BE1FBEEBBE923E5DBFA53E7A3FEBBE3D3F80BF2A3F6DBF433F233F7BBF48BF33BF7FBF7F3F443FACBE713FA1BEE4BE7A3F27BE24BF77BE4BBF403F1E3F7D3FFFBE763F7EBF70BF353F0B3F913D68BF763F523F2F3F543F82BEEEBD7B3F803F803F5F3E62BF4BBF4A3F0E3F3B3F313F413F7F3FA5BE61BF1A3F9E3E48BF593FF2BD233F803F7F3F7C3F52BF6DBF58BE36BF5FBF54BF68BFA23EA53E5A3F733F7F3F173F6DBF5EBDC3BE693F093F00BF783F243F1EBF723F74BFFCBC38BE763F653FF2BD893D003F54BFC13D86BE58BE1FBE7E3F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}
