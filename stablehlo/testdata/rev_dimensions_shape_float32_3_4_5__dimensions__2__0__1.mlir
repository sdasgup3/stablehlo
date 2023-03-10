// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xf32>
    %1 = call @expected() : () -> tensor<3x4x5xf32>
    %2 = stablehlo.reverse %0, dims = [2, 0, 1] : tensor<3x4x5xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xf32> {
    %0 = stablehlo.constant dense<[[[-1.15580153, -1.18647683, 3.42077422, -1.76529694, -1.13649952], [-2.85637474, -5.04712057, 0.199800327, -3.37034297, 0.239038169], [5.22281075, -1.72277057, -2.76423025, -0.468372345, -0.401783675], [1.29584312, -2.68716073, -4.00673866, 3.02402735, -3.562620e+00]], [[-4.62835264, 0.933315098, -1.61677313, -3.28550196, -5.13500834], [5.07587385, 5.60044527, 3.78441501, 0.828205883, -4.44870663], [3.48453283, 7.26203108, 1.95305467, -1.73211968, -3.27272105], [1.41651392, 1.80648303, -3.19986081, 1.95118356, 1.22354436]], [[3.70364332, -1.52499723, -2.01245666, 1.88199496, -5.90188169], [-0.353008151, -4.29440308, 0.663977623, 0.137453571, 1.85718787], [-2.92269301, -5.28896284, 1.0011059, 1.41669655, -7.62059211], [-1.50831878, -3.9411943, 0.403646946, -0.612915277, -0.300886393]]]> : tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
  }
  func.func private @expected() -> tensor<3x4x5xf32> {
    %0 = stablehlo.constant dense<[[[-0.300886393, -0.612915277, 0.403646946, -3.9411943, -1.50831878], [-7.62059211, 1.41669655, 1.0011059, -5.28896284, -2.92269301], [1.85718787, 0.137453571, 0.663977623, -4.29440308, -0.353008151], [-5.90188169, 1.88199496, -2.01245666, -1.52499723, 3.70364332]], [[1.22354436, 1.95118356, -3.19986081, 1.80648303, 1.41651392], [-3.27272105, -1.73211968, 1.95305467, 7.26203108, 3.48453283], [-4.44870663, 0.828205883, 3.78441501, 5.60044527, 5.07587385], [-5.13500834, -3.28550196, -1.61677313, 0.933315098, -4.62835264]], [[-3.562620e+00, 3.02402735, -4.00673866, -2.68716073, 1.29584312], [-0.401783675, -0.468372345, -2.76423025, -1.72277057, 5.22281075], [0.239038169, -3.37034297, 0.199800327, -5.04712057, -2.85637474], [-1.13649952, -1.76529694, 3.42077422, -1.18647683, -1.15580153]]]> : tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
  }
}
