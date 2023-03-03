// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xi1>
    %2 = stablehlo.is_finite %0 : (tensor<20x20xbf16>) -> tensor<20x20xi1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x58BE4CBD294045C00FC0343EB53F36C07FBF0AC0B0BF2FC0253D9E3F43400A40C3C01FC0594054C00E40C440C03D02C0A03F99BFB4BE5FC035404FC021C18AC004C0A3BF313F85BF82406D3F5140233FE9BFA740F2BD1EBE61404B40E73F6F40D6C0FABF93BF0141ACC0803F16408540E23F8DC0D5BEEDBF84C0BABFE83F0E40DB3FB6C0B0BF323E7A4069BF2D3F5740B9C0003F78C075BF9AC07EC0C8C03FC071BEBC3F8F3FBF3F493FE3BF4FC05EC00DBF6EC04F408FC089BF5240173F54C0A8BFB0BF70C06F3F89BD05404740963FACBE633E81C015BFBC3E1BBF593F3CC0FABE7DC017406DBE3CC0FF3FF6BF8240074093C09BBF2AC0A24084C072C015409DC07DBD3BBF76BFE53F0DC062BFE4BF8E3F11C0C93E32BE423FBF3FEDBE633F814049BF2FBF3640943F95C0B6C020C01C40654041408C40CA3E843C9B40D3C089C0694032BFABBF32BF11BE26C09F40C83FC6401B40CD3F04BEEBBE96C059C0843E1FC0FCBE3A408140F33C4C40843F56C0CE3FAFC080C0194000C0A63F013F97C0E8BFAD3ED9BF214021C084C0AE3DABBFB54080404EC023402DC0C8BE1BC08340CCC085405AC029C0E5BFB4BFDD3F3CBF8CC0DE3E2240C03FD5BEE0BF2E407EC07CBF43409340ECBF07BF4FBF133EEBBF274025401440D3BF89C00140C8BF523D1840A63F48BF40405BC04B40FDBFF9BF113F0EC095BFAA3FF6BE3740A2BF494085BF82C0524056405FC087C018C04B404E3F7AC0D83DE140B6BFB3C0AD40213FB54063C01AC0B8C04FC00440283F034093BF8ABE3540C8BFF7401041F13F49C0CBBFC0BFE6BF1CC00DC0B740DEBF9FBF693F86C01CBFB0BE8D40A4BF9C3F28C11DBE9BC044BFFF3E8D40E3BED63F6840EF3F18C020C0EFBFCCBF90C0D740EFBFAE3F4CC0A3BF6C4061C01A408840BBC051401AC028BE32C0A840993C6FBFBE3FA2BED24085C04E40E5BF794017408DC040C0394011C00E4043C0B1BF73C0D3BF04C08EBF61C070C058C0D83F9040E2C010C02540A9C0FEBFFBBF2D3F413FEB3E8740C1BE9F3FAE3FA340E9400F4081BFD23F05406B4017C0BDBFAB3F7E3E0F409C40754094BF29BFE9BF1A3E7A3E91407540F7BD643FC9C03D40CB3F6A3F"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xi1> {
    %0 = stablehlo.constant dense<true> : tensor<20x20xi1>
    return %0 : tensor<20x20xi1>
  }
}
