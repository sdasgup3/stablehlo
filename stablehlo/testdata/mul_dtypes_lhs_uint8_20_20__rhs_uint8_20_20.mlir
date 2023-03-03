// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xui8>, tensor<20x20xui8>)
    %1 = call @expected() : () -> tensor<20x20xui8>
    %2 = stablehlo.multiply %0#0, %0#1 : tensor<20x20xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xui8>, tensor<20x20xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xui8>, tensor<20x20xui8>) {
    %0 = stablehlo.constant dense<"0x02000201030302000300010105060000010300010301010103020201000206000101030000000401010102000301020004010300030000030300000304030501010002010000000000000203030400020003050101010300010100010002010003000104040402010303000002010204030200020003030004030200000002030401020004020100020100010001010200010100020308010102000300040504030302020003010206020401020307030303010106010300000501010000000203030301050002000200020209010100030202060705000302020204010303000000020300000004070702000702020203000100040102010001010001040001000203010500040200000201040305040503020005010300020201020004010300010306030004000503030201000105000003020000050200000001010605040502000401020304010105020001050201000205000003000002030001010003020300020302010005010302000200030003010501000106000000040301000402010200020108000201030000010001"> : tensor<20x20xui8>
    %1 = stablehlo.constant dense<"0x00020200030003000200040101030105000000050104010006020403000107030306010002010502000500060504020104010600000303000102000301030705000500000206000501000503000205020003060002030201010100000506000000040400020000020304050500030104020102000003040004030502000100050002050102030500040004020303010201020201050103020003050504010003060200020004010102010000020103000203000101030100010101000302010300030305000200040000010403010501030105000401040100010102010700010301020102040300020503010500000800050102010502010400000001020104020200090005060202000101010601010102000000030103010200040001010203020102010201040303000302050101010201000002050504030501000103040400030103000202010003010101000402000501000200010101000000000104050102010302000101030101050000010000030000000201000702030100010800000207040100000101040200020204"> : tensor<20x20xui8>
    return %0, %1 : tensor<20x20xui8>, tensor<20x20xui8>
  }
  func.func private @expected() -> tensor<20x20xui8> {
    %0 = stablehlo.constant dense<"0x0000040009000600060004010512000000000005030401001204080300022A000306030000001402000500000F04040010011200000000000300000904092305000000000000000000000A090008000400091E000203060001010000000C00000000040008000002090C0000000302100602000000090C0010090A000000000F00020A00080605000800000200030104000202000A0318020006000F0004000C12060004000C01020C02000004031500060900010603030000050100000000060009090500000000000002081B01050009020A001C050003000202080115000000000403000000000E230600230000100000010004050401000000000108000400040009000018040000020104120504050600000003030002040008000401060002030C030004000F09000602000105000003000000190A0000000100060F10140000040300060801000F020001000802000A0500000000000200000000000C0A03000209040000050303020000000300000300000002060000000C03000020000004000801000002010C0000020004"> : tensor<20x20xui8>
    return %0 : tensor<20x20xui8>
  }
}
