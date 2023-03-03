// RUN: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x1x4xf32>
    %1 = call @expected() : () -> tensor<2x4xf32>
    %2 = stablehlo.reshape %0 : (tensor<2x1x4xf32>) -> tensor<2x4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x1x4xf32> {
    %0 = stablehlo.constant dense<[[[-0.0167026278, 4.71085596, 2.45937371, -0.429338872]], [[-0.412544459, -2.02999949, -1.84527063, 3.70861197]]]> : tensor<2x1x4xf32>
    return %0 : tensor<2x1x4xf32>
  }
  func.func private @expected() -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[-0.0167026278, 4.71085596, 2.45937371, -0.429338872], [-0.412544459, -2.02999949, -1.84527063, 3.70861197]]> : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
