// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.abs %0 : tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x16A345BFE77C0F4043ED29BF5B2784C0D27785C04FF694C0C0DE2B408F45F33FFB4DB9C02B5B3BBFF41328C09786CBBEEA2E47C0B3E6963F974DF4C0F4237440E3A640C0AFFEA1BFFD63413F7B750AC0E3B644401ECB0E40773C2DC0723306419F7A97BFDCEDD8BE8C9B043F8DCFC23FA02A8ABEDB19EFBF13066B40B81E873F599C4E3F7AA629C015201B3F935546C03DB9DE3F288A2DC05C5342406AEF2DBE52EF80C02E5499BF2A9B644030538E4034C3F6BEB5270F407598823FC9C5B93F0A1B7240A07DC7406950C23FECB3833F81B083C0264F4C3F317476BF3CD575C0B54109BF790795C0CD6FBBC08E5FBB3F751A0A405DA0C33EFEE946BFCB67833E34577B40A488A44060247CBF703D1EC04E5EDCBFE2A67DC0890390C0A749BC3E478D143D9330DABFED10B4C0157CBE40984547BF8A902E3F18618A3F0B304CBFE9964E3FE75535C0918D2A40B3A27740412583402439BA3E050352C02ECAF34018128CC0231774407164183E5A6E1F4088DD48C0FCA43BBF99754A401D50313FB3E7A3402B2F73BF9B7385408C35883F98E1273FBEABDBC08227AE40AC3E4FBE1E9353C08367A7C0C8FA7A3E74F2ED3F96FC9EC0B6EDA8BE2F37573F74EF5E408C073CBFC031C6C0533A8E3E100AB8BFF6FD5CC0F2DC55C0B55F80C03A2BA3C0B3A812C0CEA6A3C0DC0B83BE0E2AD33E97C600C05629A7408F0C39C088BAE03F813134C07937763FE73798C062A896C0957C88BE6B644540F2A0EDC05520D63F7384D73F69B1873FF1D327402A41403E4438BCBFCBF307409AC70ABF9413FB3FDF4AA2C0EBCFE53F17073B401F48CB40A4397740388703414361F3BD05A636C00DA34FC01AF40A409643C4C06EFD94405307CE3E5EFA614012321C4045F3EA3F55A223C05DD42E3F33FA5F4055B5CCC0F6CA0C40520D3AC06ACF1AC0509895BF33232B40D9BB1CC077F0DCBFCF91F4C01AD0EDBF499E82C07297A640921E61C0715A3C407018A740561B1EBE0C74BF3F6EC52040F144EDBF17C4C6C092918EBFAFAEFBBFE35F98C0C91008C0BBACF8BE8FD2D53E1280A2BFF55488402965D23E0EC7094027E2CB3ECCB60440B264B3C025B644BE0439ABBFD340D7C0D4BC7940F7F62AC0EF89C23F747928C07996A0C0E6B0A1C00D33D03F1FB4D5BFB3BC95BF23E840C0DB0471C0C30A2FC0AED10740C8265D3F339A843F54AEA93FCCACE43E8E2A15C1BC5CA740CB0715C0CC0028BE71D552BF0C9CBD3DCF4CBB3FE2B576BFCE69743FCC508E3FA51C78401F8B194029B55940EA058DC03FF35E3FF40CF43F671E744000E5C63F04BEBFBF83E21640AD9EFABF40CEEDC001F30340685EF340B953A93FB65B58405D1A054030A7E63F7CBA8540F1152940DD6A46C06D21813E5C7752BF467014C0282D093EA8A7263EB0027AC0B38B58C0492395C0E84506C0D90D1B400B925640DA71B6C0417C56C0A46511BE3A7C7DBEE85B5E40AB5B5FC032879CBEF15AB83F552DAEBED089B7C0B9AC98BE28ADDF3F501B5E40A0D8DD402B6C2840A423993F9E2C24C04CFCEBC0A12E7FC0A7C220BF7A8E15C06F209D3E8E2FB1C0FC7A474027E2E8BF04486440A88332402ECD74C03EC7B43EE50BD740F70AB4C0A94C51BFCD8681BFEBBDA3401F2E903E4A7662C0ADB5A540E24221400CDDE1BFA8E13440A648C13FA8A24CBE7E6384BF4CB6AFBE47CE8D3E2CE558C0D594C8BE29B8C23F20CCC2C01AF6294036241E4082F033C0151775C0AF1D9DC0BA6587BFA6A7804072C20FBFB87A68C09DAB86C0DF3A09C178E24240739E8A3F0FBF8440360ACCBF77E94DC0E67468C003E728C0AA28253F561181BF2AC9ACBFECDBBAC06255493EA28C8A3FE4566F406E4598C028FD27BF7A1D8CC0C317D7C0B82F2A3E043ACD3E1AEDDF3FEE271140B302FCBF03DE46C0C651ACC0E0A6473F0ABBA33E911723C0281D634081F5003FF52097C0242C74C013727F40ABFA784059FA2C3F9DFFB5BD3A099D40D47A3040DEC18B40B11117BFED2C0C400602A53F848A643F90566E40906DF13E34F69AC00C4427BE29A949400F47C0BFC0D382405311993FEFCE953F29EE913F4C63803FCDB6BA3D1ECEF8BEFA7E123E0C1E484046660AC0B95F74C060D215BFF3D85B3E6ED87540E97E6BBE92ED4A40F65AC640AB7E76BFE560A64076743C40F41693C01C6F833F6F765DC003AD52404BA2253F7BDFB13FEE6D83C03B8805BDC9582C40B0B726C0ADC061C0878017BE112452BE"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x16A3453FE77C0F4043ED293F5B278440D27785404FF69440C0DE2B408F45F33FFB4DB9402B5B3B3FF41328409786CB3EEA2E4740B3E6963F974DF440F4237440E3A64040AFFEA13FFD63413F7B750A40E3B644401ECB0E40773C2D40723306419F7A973FDCEDD83E8C9B043F8DCFC23FA02A8A3EDB19EF3F13066B40B81E873F599C4E3F7AA6294015201B3F935546403DB9DE3F288A2D405C5342406AEF2D3E52EF80402E54993F2A9B644030538E4034C3F63EB5270F407598823FC9C5B93F0A1B7240A07DC7406950C23FECB3833F81B08340264F4C3F3174763F3CD57540B541093F79079540CD6FBB408E5FBB3F751A0A405DA0C33EFEE9463FCB67833E34577B40A488A44060247C3F703D1E404E5EDC3FE2A67D4089039040A749BC3E478D143D9330DA3FED10B440157CBE409845473F8A902E3F18618A3F0B304C3FE9964E3FE7553540918D2A40B3A27740412583402439BA3E050352402ECAF34018128C40231774407164183E5A6E1F4088DD4840FCA43B3F99754A401D50313FB3E7A3402B2F733F9B7385408C35883F98E1273FBEABDB408227AE40AC3E4F3E1E9353408367A740C8FA7A3E74F2ED3F96FC9E40B6EDA83E2F37573F74EF5E408C073C3FC031C640533A8E3E100AB83FF6FD5C40F2DC5540B55F80403A2BA340B3A81240CEA6A340DC0B833E0E2AD33E97C600405629A7408F0C394088BAE03F813134407937763FE737984062A89640957C883E6B644540F2A0ED405520D63F7384D73F69B1873FF1D327402A41403E4438BC3FCBF307409AC70A3F9413FB3FDF4AA240EBCFE53F17073B401F48CB40A4397740388703414361F33D05A636400DA34F401AF40A409643C4406EFD94405307CE3E5EFA614012321C4045F3EA3F55A223405DD42E3F33FA5F4055B5CC40F6CA0C40520D3A406ACF1A405098953F33232B40D9BB1C4077F0DC3FCF91F4401AD0ED3F499E82407297A640921E6140715A3C407018A740561B1E3E0C74BF3F6EC52040F144ED3F17C4C64092918E3FAFAEFB3FE35F9840C9100840BBACF83E8FD2D53E1280A23FF55488402965D23E0EC7094027E2CB3ECCB60440B264B34025B6443E0439AB3FD340D740D4BC7940F7F62A40EF89C23F747928407996A040E6B0A1400D33D03F1FB4D53FB3BC953F23E84040DB047140C30A2F40AED10740C8265D3F339A843F54AEA93FCCACE43E8E2A1541BC5CA740CB071540CC00283E71D5523F0C9CBD3DCF4CBB3FE2B5763FCE69743FCC508E3FA51C78401F8B194029B55940EA058D403FF35E3FF40CF43F671E744000E5C63F04BEBF3F83E21640AD9EFA3F40CEED4001F30340685EF340B953A93FB65B58405D1A054030A7E63F7CBA8540F1152940DD6A46406D21813E5C77523F46701440282D093EA8A7263EB0027A40B38B584049239540E8450640D90D1B400B925640DA71B640417C5640A465113E3A7C7D3EE85B5E40AB5B5F4032879C3EF15AB83F552DAE3ED089B740B9AC983E28ADDF3F501B5E40A0D8DD402B6C2840A423993F9E2C24404CFCEB40A12E7F40A7C2203F7A8E15406F209D3E8E2FB140FC7A474027E2E83F04486440A88332402ECD74403EC7B43EE50BD740F70AB440A94C513FCD86813FEBBDA3401F2E903E4A766240ADB5A540E24221400CDDE13FA8E13440A648C13FA8A24C3E7E63843F4CB6AF3E47CE8D3E2CE55840D594C83E29B8C23F20CCC2401AF6294036241E4082F0334015177540AF1D9D40BA65873FA6A7804072C20F3FB87A68409DAB8640DF3A094178E24240739E8A3F0FBF8440360ACC3F77E94D40E674684003E72840AA28253F5611813F2AC9AC3FECDBBA406255493EA28C8A3FE4566F406E45984028FD273F7A1D8C40C317D740B82F2A3E043ACD3E1AEDDF3FEE271140B302FC3F03DE4640C651AC40E0A6473F0ABBA33E91172340281D634081F5003FF5209740242C744013727F40ABFA784059FA2C3F9DFFB53D3A099D40D47A3040DEC18B40B111173FED2C0C400602A53F848A643F90566E40906DF13E34F69A400C44273E29A949400F47C03FC0D382405311993FEFCE953F29EE913F4C63803FCDB6BA3D1ECEF83EFA7E123E0C1E484046660A40B95F744060D2153FF3D85B3E6ED87540E97E6B3E92ED4A40F65AC640AB7E763FE560A64076743C40F41693401C6F833F6F765D4003AD52404BA2253F7BDFB13FEE6D83403B88053DC9582C40B0B72640ADC061408780173E1124523E"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
