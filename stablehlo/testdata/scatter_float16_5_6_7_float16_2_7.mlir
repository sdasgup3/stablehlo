// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %1 = call @expected() : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      stablehlo.return %arg1 : tensor<f16>
    }) : (tensor<5x6x7xf16>, tensor<2x2xi64>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    return %2 : tensor<5x6x7xf16>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}, tensor<2x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x80C4C93CE141A7C2B4A9F33E523FCBBB6EBD80C136445145BEC02231FAC62244B0B22AC0C23D8AC053BCDC3DA84248407346F1B5CEC01234993D11C0033AF4C0B1C3A335512F373C32B608B13348B03B2C41B6400FC818C035B1BFC08F3BF941A2BC67411FB3B040AE452F3AFA3EBAC16F41CE3C9EC1343877C132BEB53B1148E04013C02441A545913852B48F3A69B9F2404045FB3CE64447B32F439840FCBEB4C237C28042BBC0BDB7EA45C53C6EBC0B4135BE80362846D4BA11B44E412DA4F5BCB0C5F63E9BBC71418DBB7C26714261C24BB3242B14C111C024BFA1456A4566BDF99FA4B843C33BC5322C09BF293E3F400FBB81C5C2BC0F376CBAA1C01DB5A64111BEC7B32E3A5DB5B33A73BB6738174212BD84C5E24632C4F83F133C084349C2D5438FAEA3BEEEBC9FC24940EA428BC033C4B341D64084C1CBC06E43214167476937AC4276C68B3ED4409E3DEBC12C334F2EE7B4DDC040434043E1BCD21EC2C593C2913C2EBEAE3D5FBC97BD0AC088C3EAB86DC1A0C0E4B077C4E03D153FDB4729B8DE3E0CB8373B2EBF37C0EA40074490448645CD355CC49D3C6C40B0385BC2BDBC"> : tensor<5x6x7xf16>
    %cst_0 = stablehlo.constant dense<[[-4.511720e+00, 1.255860e+00, -1.725590e+00, 4.497070e-01, -9.832760e-02, 5.367190e+00, 3.607420e+00], [1.319340e+00, -4.996090e+00, -7.148430e-01, 3.132810e+00, -1.239260e+00, -4.736330e-02, -5.988280e+00]]> : tensor<2x7xf16>
    return %cst, %cst_0 : tensor<5x6x7xf16>, tensor<2x7xf16>
  }
  func.func private @expected() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x80C4C93CE141A7C2B4A9F33E523F83C4063DE7BE32374BAE5E453743FAC62244B0B22AC0C23D8AC053BCDC3DA84248407346F1B5CEC01234993D11C0033AF4C0B1C3A335512F373C32B608B13348B03B2C41B6400FC818C035B1BFC08F3BF941A2BC67411FB3B040AE452F3AFA3EBAC16F41CE3C9EC1343877C132BEB53B1148E04013C02441A545913852B48F3A69B9F2404045FB3CE64447B32F439840FCBEB4C237C28042BBC0BDB7EA45C53C6EBC0B4135BE80362846D4BA11B44E412DA4F5BCB0C5F63E9BBC71418DBB7C26714261C2473DFFC4B8B94442F5BC10AAFDC566BDF99FA4B843C33BC5322C09BF293E3F400FBB81C5C2BC0F376CBAA1C01DB5A64111BEC7B32E3A5DB5B33A73BB6738174212BD84C5E24632C4F83F133C084349C2D5438FAEA3BEEEBC9FC24940EA428BC033C4B341D64084C1CBC06E43214167476937AC4276C68B3ED4409E3DEBC12C334F2EE7B4DDC040434043E1BCD21EC2C593C2913C2EBEAE3D5FBC97BD0AC088C3EAB86DC1A0C0E4B077C4E03D153FDB4729B8DE3E0CB8373B2EBF37C0EA40074490448645CD355CC49D3C6C40B0385BC2BDBC"> : tensor<5x6x7xf16>
    return %cst : tensor<5x6x7xf16>
  }
}
