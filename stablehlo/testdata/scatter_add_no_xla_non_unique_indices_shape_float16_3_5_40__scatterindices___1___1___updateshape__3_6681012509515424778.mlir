// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>)
    %2 = call @expected() : () -> tensor<3x5x40xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xf16>, tensor<2x1xi32>, tensor<3x5x2xf16>) -> tensor<3x5x40xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xf16>, tensor<3x5x40xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>) {
    %0 = stablehlo.constant dense<"0x4CC1DE417FC0FBC21DC19D4758C2504481C0F9B99B440EC3E6456F3CF83C65C12D449544ACC10CBCF8C4ABC25EC2CFBCB145833CBDB8C0C0D34094411ABC603BF33005C45CB9A2424AB99AC753BD7B4302BE453FD1BAC8BFBB4511C5FBC7C6B293BA7ABD5B3E9E3D55465A38E1394DC452C3FFB5F3416D40AABD98C25F3E8AB9A1C4E3BD0647B13D333F7EB08D38513E10C2A040364603BB32BF0DBF89436EC49F3C66448C363B3F9AB9B6C37DC34CC2E93D69C0DFB78B3C74BF3145C24188BC2EBCC3C17BBC99C54D3ECCC401B80042DAC435C07B3C29BD073F0D42FC41BABEDD41D33C36381F407B45E537C7BCD73F91C0C1BDE6412E4211C4C2C1083E85BDA8449C3AC2C0033DC7C09B32D2BD95BC3234D8C357C17CBC47C09BC0EF3F12BA71AE21C1574688BE473E0BBF3AB6A4B8913C0DC1D1C49B415138543F48C2223C6942D3469941FC3932C49044214130B34B2E0D3EEDC5FCB820BC1940A6252EC33AC7F6C1FBBF00C4063F073FAB440D44CAC2C0C4F94251B2893A0AC1AEBB414057C406C090BCD93D1DAC78BF3FC6704410318F3F93BA0140DE422042713FA03CAE3CC04432409143D53E873CEBB9103D5743B2BC56400645FBBD7D3F56363FB86FC0F63E683AECC1BFBF5DBEF83ADBBC0F3CD63350BA883CA9C0ADBA4F4374BB8AC15FB88CBC13BF573C03C085343D47C038DC42DF44B0C58BBF9F44473AEAB903B8FEC34D39DC3ECB36D4BDC43C8A3945C1933CFBBE8BC06BC5804468C21842C4B777372A3A3D3125480544CEC36BC4473CB7B8E2C17A4245BE514041B6AABCB246D1B676BFA544F9B755C036C21FAA5333FAC43A4325C023B5FBB76AC087BC2F3D913F90B574C05FC6B53D4BC4923B8D4288BF883C1C403842F83B913BF8C0CFBC6EB16BC03D45964510B38244D2B99BC53B40DEBD35414E4070C7A9C27A3865401B3CBDC2FFB80D3E53C3BC41D3C1AEC2BC3B7F4180416141F7BFC43F70C460C34146CFB564C3FCC0CE3D28465F413D3EC7C43B458D3387C49F3E14C3D43CD4BE0440713C57B98338CE40A942113DA6C1D33BF340623CFBB3A7C0DE40024196BE73374EC0DB29D3411DB9814253C0AC441A40FBB4B5C330488043B6C27DBC46B019B251C0CDB8283B81BFBF40803F68BC4A429BC863C30DC4AFB2043D964288441F46D9BBDFC1554583472AB6E2BB03BA454675443D3CC2BDD644A6434A4893407E441FB6C943DFC083BEBE44C7461BC50B3CDAB3CBBF553A5B3FCE46FE41F540A3C18847E6C3A4C01D40C4BA004275BA4FB6004009BC3244F846E14352BA8A4031BE0D3F9741F54303BDCA4571BEADBF1D300AC544386C43273DFDC39FBCC4B9CFB90D37A0C1E6C17A3D28B6E54306B30EBEE4C1252B71BC46C0E5424643D0347DC393BFB7401A40B8402C3E81413FB1F538D33C134069C370B11B4303C0F23DFBBE4BC39D41094451C067C336B9983F05BF3AA9C8BD4B3703C19FC232C4ED3C21BB64359BC45DC51BBCA23FEDC5AB3F84420FBEF5C4BBBCD2C109C413ADE23D5A3CF8B43CBFC44167BCAE386B400A3EC64342BA27C040C070BDC43E53A9BF3FF1C10CBC814258BDDDC5E4A8A83CA1C1C742E0C3F63A4E3A67C04F423638FFC27D3A28B3ABBB5A4180454A411C33DE485EBE473A32460C401037B0C4F8440F3F53B665BF9D3BF8435743084443BF"> : tensor<3x5x40xf16>
    %1 = stablehlo.constant dense<[[[6.987300e-01, -1.494140e+00], [4.782710e-01, 6.917960e+00], [2.138670e+00, 2.282710e-02], [-3.509770e+00, -3.599610e+00], [-2.609380e+00, 1.073240e+00]], [[1.473630e+00, -1.200200e+00], [-5.488280e+00, 3.413090e-01], [1.434330e-01, -4.339840e+00], [-3.335940e+00, -4.113280e+00], [5.683590e-01, -5.307620e-01]], [[-9.902340e-01, -3.308590e+00], [-1.591800e+00, 3.650390e+00], [-2.603520e+00, 2.199220e+00], [1.012500e+01, 1.871090e+00], [-1.346680e+00, -2.914060e+00]]]> : tensor<3x5x2xf16>
    return %0, %1 : tensor<3x5x40xf16>, tensor<3x5x2xf16>
  }
  func.func private @expected() -> tensor<3x5x40xf16> {
    %0 = stablehlo.constant dense<"0x4CC147407FC0FBC21DC19D4758C2504481C0F9B99B440EC3E6456F3CF83C65C12D449544ACC10CBCF8C4ABC25EC2CFBCB145833CBDB8C0C0D34094411ABC603BF33005C45CB9A2424AB99AC753BD7B4302BE9B48D1BAC8BFBB4511C5FBC7C6B293BA7ABD5B3E9E3D55465A38E1394DC452C3FFB5F3416D40AABD98C25F3E8AB9A1C4E3BD0647B13D333F7EB08D38513E10C2A040364603BB32BF0DBF89436EC49F3C90468C363B3F9AB9B6C37DC34CC2E93D69C0DFB78B3C74BF3145C24188BC2EBCC3C17BBC99C54D3ECCC401B80042DAC435C07B3C29BD073F0D42FC41BABEDD41D33C36381F407B45E537C7BCD73F91C046C8E6412E4211C4C2C1083E85BDA8449C3AC2C0033DC7C09B32D2BD95BC3234D8C357C17CBC47C09BC0EF3F12BA71AE21C1574688BE473E0BBF3AB6A4B8913C0DC1D1C49B415138543F48C2223C69424A459941FC3932C49044214130B34B2E0D3EEDC5FCB820BC1940A6252EC33AC7F6C1FBBF00C4063F073FAB440D44CAC2C0C4F94251B2893A0AC1AEBB414057C406C090BCD93D1DAC78BF3FC670441031544093BA0140DE422042713FA03CAE3CC04432409143D53E873CEBB9103D5743B2BC56400645FBBD7D3F56363FB86FC0F63E683AECC1BFBF5DBEF83ADBBC0F3CD63350BA883CA9C0ADBA4F4374BB8AC1B2C58CBC13BF573C03C085343D47C038DC42DF44B0C58BBF9F44473AEAB903B8FEC34D39DC3ECB36D4BDC43C8A3945C1933CFBBE8BC06BC5804468C21842C4B777372A3A3D3125480544CEC36BC4473CC9C4E2C17A4245BE514041B6AABCB246D1B676BFA544F9B755C036C21FAA5333FAC43A4325C023B5FBB76AC087BC2F3D913F90B574C05FC6B53D4BC4923B8D4288BF883C1C403842F83B913BF8C0CFBC9EC76BC03D45964510B38244D2B99BC53B40DEBD35414E4070C7A9C27A3865401B3CBDC2FFB80D3E53C3BC41D3C1AEC2BC3B7F4180416141F7BFC43F70C460C34146CFB564C3FCC0CE3D28465F413D3EBEC43B458D3387C49F3E14C3D43CD4BE0440713C57B98338CE40A942113DA6C1D33BF340623CFBB3A7C0DE40024196BE73374EC0DB29D3411DB9814253C0AC441A40FBB4B5C330488043B6C27DBC46B07DC451C0CDB8283B81BFBF40803F68BC4A429BC863C30DC4AFB2043D964288441F46D9BBDFC1554583472AB6E2BB03BA454675443D3CC2BDD644A6434A4893407E441FB6C943DFC083BEBE44C7461BC52442DAB3CBBF553A5B3FCE46FE41F540A3C18847E6C3A4C01D40C4BA004275BA4FB6004009BC3244F846E14352BA8A4031BE0D3F9741F54303BDCA4571BEADBF1D300AC544386C43273DFDC39FBCC4B986BC0D37A0C1E6C17A3D28B6E54306B30EBEE4C1252B71BC46C0E5424643D0347DC393BFB7401A40B8402C3E81413FB1F538D33C134069C370B11B4303C0F23DFBBE4BC39D41094451C067C336B9983F1E493AA9C8BD4B3703C19FC232C4ED3C21BB64359BC45DC51BBCA23FEDC5AB3F84420FBEF5C4BBBCD2C109C413ADE23D5A3CF8B43CBFC44167BCAE386B400A3EC64342BA27C040C070BDC43E53A9BF3F3BC70CBC814258BDDDC5E4A8A83CA1C1C742E0C3F63A4E3A67C04F423638FFC27D3A28B3ABBB5A4180454A411C33DE485EBE473A32460C401037B0C4F8440F3F53B665BF9D3BF8435743084443BF"> : tensor<3x5x40xf16>
    return %0 : tensor<3x5x40xf16>
  }
}

