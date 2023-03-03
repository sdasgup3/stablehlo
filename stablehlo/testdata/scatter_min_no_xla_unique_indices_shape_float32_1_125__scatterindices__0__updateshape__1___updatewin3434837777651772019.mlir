// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf32>, tensor<1xf32>)
    %2 = call @expected() : () -> tensor<1x125xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x125xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf32>, tensor<1x125xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf32>, tensor<1xf32>) {
    %0 = stablehlo.constant dense<"0x3E7E6AC08CB330C05D7067C031258EC024EE2EC063CA7EC0190383C0E58AC03F253E723E3D64FDC056EC4AC00FE52FBF67F328BF1DB718C00DC4723F44C18CC0E99049BFC8DAA74054677C3F30BD1A3EAE651340A7E5E9BF1E8E38C00DFED53C3EDE1D40DCADE73F07EAC1BF3951994010D69BBF17AE8B400B62183DB905C5BDF34B4DC0601B6F40EB413B3FE85D6D409FC0ECBFCA0ABF3FF16ECCBFB0C200C0F29605402DB9913E5D4D904052E3F33E094B3140A3485440CC8EC6BFAC1A37C08409483F8A7A3D4048C82CC0FDAE7DC05F3E8EC03B9799BD28A987C0E3CB07BF01044140BEBDE7BF1E486ABF464269BF7C4F22C0974F02C0BADEC3BF4AC7933EDE8E97BE916175BF33886F3EB5A2B13FF1F8D0BF4CF5A9BE8668D53F3658563FD7160040E4BB3F409DF41AC038F43B405D1696405F2F8BC09F7C83C0320DA840FA1C4F3F0B60783F1B47AF3FA68D01C05CB0D43FBD0093C04F18C83F5980CC3FAC1307BF4E428FBF6888EBC07D6E014025F49ABE81F92BC057600940DEECC8BF60768040DE4E96C0391396C08CCA89C080EDA63FC9E41CC03BEFC940ED1C0AC0B9CF663E1A93274068F12E40FC0E12C042524D40B241A73F23B417C00A8328BF19AC063D32AB97BE9F424740268D53402CD9BA3F43F8834006EE81C0061A3A40DB036440C1D8DFBF99EC04C0FFDCA5404AE8703D"> : tensor<1x125xf32>
    %1 = stablehlo.constant dense<0.660861551> : tensor<1xf32>
    return %0, %1 : tensor<1x125xf32>, tensor<1xf32>
  }
  func.func private @expected() -> tensor<1x125xf32> {
    %0 = stablehlo.constant dense<"0x3E7E6AC08CB330C05D7067C031258EC024EE2EC063CA7EC0190383C0E58AC03F253E723E3D64FDC056EC4AC00FE52FBF67F328BF1DB718C00DC4723F44C18CC0E99049BFC8DAA74054677C3F30BD1A3EAE651340A7E5E9BF1E8E38C00DFED53C3EDE1D40DCADE73F07EAC1BF3951994010D69BBF17AE8B400B62183DB905C5BDF34B4DC0601B6F40EB413B3FE85D6D409FC0ECBFCA0ABF3FF16ECCBFB0C200C0F29605402DB9913E5D4D904052E3F33E094B3140A3485440CC8EC6BFAC1A37C08409483F8A7A3D4048C82CC0FDAE7DC05F3E8EC03B9799BD28A987C0E3CB07BF01044140BEBDE7BF1E486ABF464269BF7C4F22C0974F02C0BADEC3BF4AC7933EDE8E97BE916175BF33886F3EB5A2B13FF1F8D0BF4CF5A9BE8668D53F3658563FD7160040E4BB3F409DF41AC038F43B405D1696405F2F8BC09F7C83C0320DA840FA1C4F3F0B60783F1B47AF3FA68D01C05CB0D43FBD0093C04F18C83F5980CC3FAC1307BF4E428FBF6888EBC07D6E014025F49ABE81F92BC057600940DEECC8BF60768040DE4E96C0391396C08CCA89C080EDA63FC9E41CC03BEFC940ED1C0AC0B9CF663E1A93274068F12E40FC0E12C042524D40B241A73F23B417C00A8328BF19AC063D32AB97BE9F424740268D53402CD9BA3F43F8834006EE81C0061A3A40DB036440C1D8DFBF99EC04C0FFDCA5404AE8703D"> : tensor<1x125xf32>
    return %0 : tensor<1x125xf32>
  }
}

