// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.subtract %0#0, %0#1 : tensor<20x20xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf16>, tensor<20x20xf16>) {
    %0 = stablehlo.constant dense<"0x384025C254BEDEC45EC3DEBED73CB846C1C029C81F45A74275C5B34264374DC214440125A7C358C094BF5FC459B6814322C003C415C12E44DE3D0EC3C3C023C1BBBDA4C6AE3DE7B376C177B72A44703F36BCC3387DC0DEBEA93C9B4052444D43453F05C753C4C9C6C8AC93C0F23E5D3D61AF22BF653100C80144204544C29935573C6EBF41BC26BC4EC34BBDED443043EE4300C67B3E3AC093BC3DA77CBF49B5193A413C32421943B4BFDE42C6BCFA3754445D41ABBCCA41904811C444BC2D3D22C049BE903D59B8CABE55C3A8BF3039D5C149C1E4C2F942CB4674B929C5E5AA153EE4C35BC5C6B81FBEC7ABA2BFCB4755C1E441343F533B2DBCBDC026BEA6C29BB8B6C191403737BCBDD73B293D653AB03DBFC0E1C1BCB076C530439B3FA83D363CC03F06BE474411C5E23A1FC4DDC3133BFB42CBBC52BCF343B43EB5B9C8C25E457C4427BE8EBF70C09E38F9B412C408C46E387C4110BCBE420B41243E70BC9E439038F83F4DB9C9C504BC4EC657BF7D40CF40A242DFC1AA3D8DC3CD4198C02A40C131BE4527BCBAC409C0F63CC0C123424CB9AF45763231441CBB6ABA59C2E13E873F5F37F7C455359A447B421D3E003BA040BDC0FFC17EC2133582B81D3B74BFEC32743C29C15BACAEC2EDBFF44288C218C115C74CB373B7793012448BC5F6B033403C41E6C3BFBA0DB6E33C4BC0B9C32EB75CB9DCBC6BBB5FBE43457CC070C54EB314420440383F373B44BE1E4410BFECB1FC3C7EC0DF428ABEF0BED1B88F40A8C125C18E3B6F433F40E2B4A938C24341B1B5B92F42014455C0933E053FA23DCDC15DAE79424C3BDD4194BC6931501A2740A2C5F1C446BA72401A39B4343C3B04BDF3C38DC05F3CB04035C335B5273CDC46F83C9930B54676BE14408B40B1454F42734685432440DF380E4037BE3DBF3FC5EE3D13BE7ABBF0C5CE3A2444A641B7C0D63218A6E635F2C353BE723CCC4273C41544414049C107BB523D03C077454D446CBFF5C222427F3CB9BC49BC3935EAC11FBE654301C068C045C6E4BC8CC24E441B403E445C449836BCC14743B8BF42B918459BB615C664419F3B45BB0E41E5ACACA18A3657BCF6445ABDDEBEE0BE893F9D3A4CC59AB69AC357B1993B"> : tensor<20x20xf16>
    %1 = stablehlo.constant dense<"0xEE333B45833ADAB8083A59C37F3E13C3B9B8EDC1DDC1213C503C544176BDADC0E7C54C3EF9C31640783914BB81BB89C296401A43C1457A408B3F08BD353D96C528C00FC3F24320B50D3C16B482C29B428144624233C5B344083C1D2C8F445B3562BB5E40CFBFCA3E7BC4E53DD6B73BC0FCC5D44318B0063E3EBDF2C75B3AB728513845BE574484441D3AE740CC33D545C9B38442F83D854883BC7142673EE1C43CC5F546D9B52E4401C4F04495C38F3ACF3FC2C168A70B2F9DC173C25DBE4A461ABCC73E36C2C2C386BE9538D1C28935A141143AB1B70FBF184858C40042813B22426B435CBAF63544C158C1513EFAC42ABCAA426F461CB95DBB5541003B39440B47EFC208C1CABF86C218C121C396C0784136C4BA3F3FBFCBC192C4A1381D2D8AC32841483EF83ABF42383FC63EC2C4A13CE5C245B7C742FB3B6AC0513F65B5C9C1C13EE73C5FBA6942E1B8D83414BB75BAB4A407BB7EB5DD3F16C079C1F43F2BC667B09DBE6238D4C11F46F5BFDE41C23D0DB7D1C389C262BE23C47DB67D3F0A35DBBE063BCAC33D420ABB2939FB42F13FD6C08A3F524665BDBCC040C098C091BD243AFA3BB540A7437741464561C8C2A52EBE25C012367A4258C13CB49F45E51B40408D45C2BEE4396DC1B540D9B6B0BE4BC86544193CC9BD4FC4C4390FBE6B408AC43C470D4086C134401CC590366EC27CC3DB47D3BEA8B5B8BC13BE07C43D4465B82642423C6BB3EEC442C025446B425F3FFEBF24C6CC389D432C40293DBCBE54B84BC04EC3CAC35D4599BDB43FA3A8F5C4193D54440DB44FBCA2BDA0C39D38FDAE5EB624BE003A393DF73BDA447538D93DDB4536BAB3323DBD52BC3DBDD72EA53BA1B90C36D5B57FC056C0C94159C4D5BC233FE34425449540A33B04C559B4A2C1163E46C1E72F7BB35F446FC58F3F25BED0BE6CC3CD409AC0AAC3E73C6BC1A93BA8409EC383BC2F36303FBBBED0C0D34457C3F9451DB92BB932C191C1063FB726EBB852B998B217C003C0FD3DB643D83A7ABEE241FB44143F61C2ED3940411EBEB7BF4244E9B486BC5D45733BBA3C913D0DC157336CC30630DB3822B7B2C56D48E33E4EAD4443B0C45D44523D2ABB6B41B2C0CC4236405434AEC87C3F"> : tensor<20x20xf16>
    return %0, %1 : tensor<20x20xf16>, tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x723F27C8CBC043C470C4D43FA0B6214926BF5CC50748964089C67C394F3F80BAFE4838BE203137C428C1F9C2543805475CC490C726C8C43FB4B68AC05EC309422A3939C21BC1B22C7CC3C2B26B47C6BD8EC531C1E9416AC608317A40A0B3A2427B419AC8BEC03EC8684486C37440EA42DE45B2C5BE34C1C85045894ADBC302355D38A4B467C58EC56BC48CC3AF447AC03544A1C8183094C900A47FC2F2C28C44FF45E5C5ED420CB9284004BE324124B5C04090458DBC9241F749BCBA3238FFC42ABC88C27F44AC4240AC3DC4FA3DD734BBC5CEC2EEC1404594BD534314C8EFBB2FBEA8C790C4C1BB693C3941FAC2624A80BE30B6A2C4383EE8AF09C5D3C08CC79EC7E438CC44CC40503F0E43DB442F4240BD5A3FDFC4A83E21C115484A3D563DD24420B927C2D04238C88EBBD0C59C3A5EB4F046F4B978C4F441C44316C11BC22148984187C15EBC6CC5C03CE8B85FC273C294383E4361B99F3D9044464432C2FD48AA394A43D8BCBEC120C751C4C5C4703AB1413A4750350642C8359D422BC4123F933FDD44B641D8C78DBCC3385EC6553C063F99431EC68A45EA3D4B3D04BB3942753C4AB852C7FCC27A3F11C025492E3BB743C0B4C1C27CC6FA41C8B4BBC478BFA2BF70C420BB6FBA04B956C4CF4360BE0A46BDC902BDD83B7344B34207C4BAC0A4469EC4FAC5AC3FF6C055461DC12CB9964243C8DC3797B89CB6C846243FD6C8233580A88C3B1340D545803800A7FAC40EC07D42CA43AC4171C5A4C392BFED4393C0D0B699449C477BC2603C60BDD543CB44F4BFF2BC42445BBC1A429145A73A95C1C734C6443031813E48C0AFC46FB8EA38BEC92AC4F3BB1043DF3E6A3E613A6BC08BC24EC1D43D9844BEBD70C26345094856B8BEC42041D0C3563C4A47F745F845EE446646CA3FBE3AB0C0C24366C36CC35F426240ACC246C3AF44D4418845A1C23BC09243FC3D5CC4C2C29641CE45A3C8C047B1C302C070B3DB43383A6B434644F6BCA0C18C4256429A3A23C10FC3A0C3B02D0C3AFCC6F2C329C2DABFE6C5D645F64300A4AB442C3E1EC86A4139C219C09E4723B9BEC02441883568B71C4877C8EEBEDE37B8C4D348B4C518C296BA9ABA594259C809C112C499485FBB"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}
