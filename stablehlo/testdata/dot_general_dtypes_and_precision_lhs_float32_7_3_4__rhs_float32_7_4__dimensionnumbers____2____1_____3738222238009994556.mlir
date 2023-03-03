// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xf32>, tensor<7x4xf32>)
    %1 = call @expected() : () -> tensor<7x3xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<7x3x4xf32>, tensor<7x4xf32>) -> tensor<7x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xf32>, tensor<7x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xf32>, tensor<7x4xf32>) {
    %0 = stablehlo.constant dense<[[[-5.06569719, -6.63685608, 1.26659703, -2.23289418], [0.328246891, -3.34304667, 1.73156273, 1.53923786], [3.4846642, 0.705635368, -2.22549462, 0.242756724]], [[-1.70394194, -2.51021576, -2.09778261, 4.19714069], [3.01307082, -2.76771045, -1.0915395, -1.73421705], [0.299419522, 0.845451295, -0.819909095, 3.19720507]], [[1.99056602, -1.93757975, 2.53377318, -3.09514332], [1.75113702, -2.4899106, -0.79994142, -2.40296936], [-0.0599169321, -0.23335208, 2.01964235, -0.321210414]], [[0.842348873, -0.998543977, -0.127073184, -1.44472122], [-1.84733689, -4.396110e+00, -2.45703483, -2.336200e+00], [-0.861453115, -2.29369187, 0.675022483, -3.52336645]], [[-0.564810574, -5.08944082, 1.16774499, -1.53107715], [1.45824754, -5.4955616, 0.416088909, -3.83327866], [2.1967895, 1.75280392, -0.537377656, -0.707910239]], [[-8.01627826, 4.58239603, 5.02564526, 0.80679822], [-0.836623311, 1.24237132, 2.776280e+00, 0.584487915], [-0.0894802064, 4.31166935, -0.373315901, -0.898094534]], [[-2.55243015, 7.37620639, 1.54967582, -0.28037253], [-0.838343977, 3.83477807, 1.47972381, -0.201482028], [0.0161808729, 4.45749187, -1.6296674, -0.955628037]]]> : tensor<7x3x4xf32>
    %1 = stablehlo.constant dense<[[-0.55160737, -0.90700668, 1.06096828, 5.65155697], [-1.60879683, -1.79970551, -3.59757376, -1.44012725], [0.368852973, -3.22422409, -3.70365787, -3.66013098], [2.05148673, 0.567722142, -8.35568714, -1.18737185], [-5.40125656, 2.91858554, 0.214062393, -5.193070e-01], [-0.997602939, -6.650330e-02, 1.78420949, -1.0177536], [-0.352346808, -2.6139946, -5.54016876, -2.50792646]]> : tensor<7x4xf32>
    return %0, %1 : tensor<7x3x4xf32>, tensor<7x4xf32>
  }
  func.func private @expected() -> tensor<7x3xf32> {
    %0 = stablehlo.constant dense<[[-2.46156049, 13.3873253, -3.55140829], [8.76145648, 6.55803203, -3.65796709], [8.92581844, 20.4318333, -5.57411337], [3.9383769, 17.0185966, -4.526170e+00], [-10.7582121, -21.8359184, -6.49712515], [1.583800e+01, 5.11059618, 0.0504906699], [-26.2643356, -17.4213181, -0.232284054]]> : tensor<7x3xf32>
    return %0 : tensor<7x3xf32>
  }
}
