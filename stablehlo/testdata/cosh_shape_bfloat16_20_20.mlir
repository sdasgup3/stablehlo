// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %3 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %4 = stablehlo.constant dense<-0.693147182> : tensor<20x20xf32>
    %5 = stablehlo.add %2, %4 : tensor<20x20xf32>
    %6 = stablehlo.exponential %5 : tensor<20x20xf32>
    %7 = stablehlo.subtract %4, %2 : tensor<20x20xf32>
    %8 = stablehlo.exponential %7 : tensor<20x20xf32>
    %9 = stablehlo.add %6, %8 : tensor<20x20xf32>
    %10 = stablehlo.convert %9 : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0xF63F21C1BDC02E40F7BF48C07C3FA4C0AB3F643EB83E04BF5540DD3FAF3F713F813F80BF41C09E3F703F43C001C05DBE844076C009409CC0A0BF0540CA3F7F3F6DC0573F9D4026401DC05D40F0BF3EC0893FF4401F402F3F87BFD940303F33C081BF43BF09C05EC0483FC33ECDC0FEBFD2BF6E3FFF3E5CBF80BF2B401440A93F5AC087C00C4030C0C0BF9540F7BF8740E13F6CBE82400BC082C03540AF408C403F4033C087C0FCBDD74046C09FBE894073402BBF7BC083C08ABFE73E5940693F5140E7BF9AC06C3E03BF0C3F253F79C08840D43F8ABFEEBD85405640DABFD1C0A83F85402C409EBE8E3E9C4049C027C05E400B40BAC049C0EEBD6341343D9BBF42404DBF67C0723F943FE0C04740F43EBABF42C011C09FBFE3C0C73FA8C0354006BF9BBE7840AABF9340B3BFA4BF51C0904015400D4109C01040684004C03EC04EBF05C0ABBF0540B640A9C007BFADC0E1BF8740323F33C069BF50BFF93D513FE8BF15409BC086C01E4043C00C4007BF37C03AC0CE3F28BE34BF9EC0933FB6C0FC3F573F703F513FA9BEEE3F123FE9C0764020C0913F5FC06A402BBDAEC02B41D93DCC3F28C0AB3CB54098BFFDBF08401E3FE3BE9AC065C018C016C03F40983FC4BFDF3EF83F12C088C0924035C002C0CF3F9B40EABE3A3F13404A40DFBF03BF93BE45C0F13FD73FA53D073FA9BF9CC0774010C02F40EB3E123F4A404EC0E53F50405FC0E840FE3FCFBF7EC0AC40EA3F7FBE6BC0BD3F903F10407BC08E3E13BF5C40763EC23E49C0B9BF2240A8BFECBFDE408F4062C0FB3F043F0C3E013F08C07A4097400A4059408840C53F08408B3F0BC0EB3F40402840C1BFF1BE773FA8BF2840EBBFAAC05A3FB7405F3FABBE993E97BFD33F873F03BF4D40B03F233FECBFF1BF3A4067C0E9BF814001C0A13F88402C40DDBF9ABE31C083BFBE40893E81C056C073C01D40174006C0C93FF13F70C044C081C050BEB340C43EED3FF6BCFFBF88C0CBC0B23ED040884092C01C3EA93FE43FA0BF654015409340A13F14C09CBFD2BF4FC0404062C059BF87C0343F48C0E63FA3C035C0B8BFE43F8ABE84400CC0A940EA408DBFF3BFCC3F643EC03FEA3F4F3F4DBF33BEC4BF56C0BD3FD9BF3BC0"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
  func.func private @expected() -> tensor<20x20xbf16> {
    %0 = stablehlo.constant dense<"0x5F4037463843F44061403641C33FA8420240833F883F913F5F413A400640BD3FC73FC63F2441EF3FBC3F29417440833FF841BB418A408342F23F82402240C53FA241B03F8742D740BB407D4156401C41D13F8044C1409F3FCE3FDC439F3F0441C73FA73F8A408141A93F893F97436D402B40BB3F903FB23FC63FE940A3400040714108429040FB4017405342614008423F40833FE9418E40E9410841ED421F421F4104410842813FCF433141863F1142B2419E3FCA41F041D23F8D3F6E41B93F524148407642833F913F943F9C3FC4410C422E40D23F813FFF4163413640AC43FF3FFF41EC40863F853F83423941DB4081418E4027433941813F3149803FEA3F2641AB3F9441BE3FE03F094434418F3F104026419C40F03F17441E40BF420841923F863FC141014046420940F83F52413442A64052458A409940964180401C41AC3F8240024082401443C542923FDF423F400842A03F0441B93FAD3F813FAD3F4940A6407E420442BE4029419040923F0C4113412640823FA13F8B42DE3F14436A40B03FBC3FAD3F873F5240953F3644BB41C440DB3F83419B41803FE642AB46813F2440DE40803F0F43E53F6B408840993F8D3F76428F41AE40A8401F41E53F1B408C3F63409E400C4240420841784028407E428E3FA33FA1403C413C40913F853F2E4157403240803F923F00408342BE419940F7408E3F953F3C41484145404F41834130446D402840D441D8424C40843F9D411340DA3F9940CA41853F963F7941843F893F39410F40CA40FF3F4F4001442F4289416840913F813F913F8840C74160428C406E410C421C408840D33F8E404E402141DE4018408E3FC03FFF3FDE404E40CB42B13F1843B43F873F863FE43F2D40CE3F913F454107409B3F4F405740134194414B40E1417440F33F0C42EC403A40863FFF40C93F3D43853FE1416341B241BB40AB40844021405740AA412B41E141833F0643893F5140803F6F400C428E43883FA6430C424042813F00404340F23F8F41A6404642F33FA340EB3F2B404B4121418941B13F0842A13F36414640A34208410E404340853FF8419040C5423B44D63F5A402440833F17404C40AC3FAB3F823F1B406341134034401541"> : tensor<20x20xbf16>
    return %0 : tensor<20x20xbf16>
  }
}

