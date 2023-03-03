// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>)
    %2 = call @expected() : () -> tensor<5x6x7xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xbf16>, tensor<2x2x1xi32>, tensor<5x2x2x7xbf16>) -> tensor<5x6x7xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>) {
    %0 = stablehlo.constant dense<"0x133F8A3C62BFD0408CBD2B414BBEFE40B1C092BF1A401AC0D8BF0D404B3FFFBECE3CAF3EA43F604019C0C0BFCCBFD0BFA6BFA13E804039C0333E4A4092C0BA400A4072400AC003408940C73F8CBFA540853FC93FB4C0E73E64BF1AC0A140BAC09240A4BF87C09CC065C000BF7D3F3D3F26C088C01BC042BFAB3FF43C0D40CD3F653F6A3F11C06FBE064057BFB8BF823E08BE5740B73ED94095409740B240C5C0133D013F5DC0EF3ECE40C1BF5BC029BFC3C0DA3EE5C0B240D2BE4EBED53FD23C1EC07A404740ED3F773FFE3F5240993F1B4007C02CBE823F9A408A3FA2BF9E3FDEBED2BF9EC0393FDD3F034017C0E5BF103EF240D7BF9E408ABF8740034077BF904054408840303E0EC070BF323F24408D3F1140A73F03C072BF01C021C00DC0123F91C06FC07540454081BF303F51BF74C070BDC6BFD8401940A3BFF5BEB440D5BFB53F98BF483F10402DBD48BF03C01F3EF6C013402A40054078C0013FEDC06CBFE2404BBFC2C0A5BFA240B73E4ABFEBBF70C09ABFAA3E09404AC0AD3F2F40F9C0D9BEC3BF02418B3E743FFD3F0FBF943FC23EBEBFDC3F4640ED402940993E95BFAD3F"> : tensor<5x6x7xbf16>
    %1 = stablehlo.constant dense<"0x01C0B1BF3440ABC02FBF14C0B8BF244072C0AB3F423F7FBF7D4035404CBF5C40354077406040C53FCEBF52C0B13F8ABF5CBFB43D6C40C9BF85BFF7C0FA40453E6D3F6FC0E0BF75BE6B4019BFBFBFE7BEE3BFF93FB2BF5EBE38BF1CBF253F5BBEA23EA6403C40354095400640A940274097406F40CEBDC3BF233FFF3F3440B4401440703FBA3F1B4043C0933F12402FBE4040BEBFE1BFABBE394092BF4B4080C0DAC03F3F14C0C5BF3DC0C0BF0B4084C08B3EE9BEF9BF0140C44004BF03405CBFB1BCF0C03A3FA9C08940C7C0284002C0ADC069C02EC0034077BF3F404EBE66409440393FBDBF27C0FD4094C06CBF0940713F87402E3F2DC065BF69C091BFA53DCAC0193FC4BF93BEA54088C0AC3FA3C083BF7B40933FA6C0"> : tensor<5x2x2x7xbf16>
    return %0, %1 : tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>
  }
  func.func private @expected() -> tensor<5x6x7xbf16> {
    %0 = stablehlo.constant dense<"0x94BFBFBC1FC00BC23F3DC6C1923EA341A741C3BFE93F1940D5C0C74022BFDBBF923DA93F9040AC4076409E400DC0E03F8F3FE23C6C419140333E4A4092C0BA400A4072400AC003408940C73F8CBFA540853FC93FBB405FC0DFC0EDBE9540AE4100C19D3E78C13A40AB40673EE0BFB83F67406C3FDF3FEC3E5C3FD1BB323F05412840254029C1FABE31410CC0B8BF823E08BE5740B73ED94095409740B240C5C0133D013F5DC0EF3EF341B4C0B03E813F78C0593FA1C1FA4173BF41BE1B407E3DF1409040E340A2BE39403DC0B9C0CCBEE0401A4008BF82C003C24E3F3B40F3BFDEBED2BF9EC0393FDD3F034017C0E5BF103EF240D7BF9E408ABF8740C1C0B93F1C415BC1943FA0BD8A40F2BF8840A9BF1040F9BFE7BC764130BF2A412CC15B41C03F1341A2415FC106C104C02ABF1CC0443F58BEC6BFD8401940A3BFF5BEB440D5BFB53F98BF483F10402DBD48BF03C0383FB2C059C0DEC083418F41EEBE7EC15EBFEE410ABF8341943F93C1CFBE82BD39410FC0EC3FC3BD31415741E83F5FC1FF40D5BFE0BF29C28B3E743FFD3F0FBF943FC23EBEBFDC3F4640ED402940993E95BFAD3F"> : tensor<5x6x7xbf16>
    return %0 : tensor<5x6x7xbf16>
  }
}

