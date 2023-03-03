// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>)
    %2 = call @expected() : () -> tensor<3x5x4xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf32>, tensor<2x1xi32>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>) {
    %0 = stablehlo.constant dense<[[[-0.790470063, -0.376827657, -2.54231954, -1.94054329], [-0.127073929, 3.99025154, -0.220204636, 4.24919128], [3.01429486, 2.4406364, -3.44312334, -1.9173075], [4.9877429, -6.07382202, 6.09596968, -1.17617261], [-3.09937429, -3.2480278, -1.11466146, 3.65894794]], [[1.55535305, 3.06394267, 5.10202551, 1.00784862], [-1.21928763, -2.06880212, 0.181157425, 0.0289308783], [-2.71434927, -2.71204138, -2.69678211, 5.0204649], [-0.791190505, 5.32497835, 3.55645466, 0.712908864], [4.05458784, -2.70867419, -3.32875013, -2.53978419]], [[-0.115624271, 3.07909942, -0.737454593, -5.84991312], [-2.25276589, -3.29018068, -1.0779022, -1.80776691], [1.53018463, 4.2541976, 6.96445084, 2.22642112], [-2.29474902, -0.823115527, 0.316226214, 6.18460131], [2.60330725, 1.22173953, 0.553669035, 0.0984589084]]]> : tensor<3x5x4xf32>
    %1 = stablehlo.constant dense<[[[-2.36471295, -1.3231287, 3.5165875, 5.12214947], [0.677324414, 2.05558181, 6.70615196, 1.73773336]], [[-1.04534864, -7.597540e-01, -0.273601145, -0.289725214], [-0.904667377, -0.709358513, -4.80053806, -1.05981922]], [[-11.6888456, 0.292566538, -0.180320784, -5.09515333], [-0.725690305, 0.137041315, -3.93096113, -1.45087361]]]> : tensor<3x2x4xf32>
    return %0, %1 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> tensor<3x5x4xf32> {
    %0 = stablehlo.constant dense<[[[-0.790470063, -0.376827657, -2.54231954, -1.94054329], [-2.36471295, -1.3231287, -0.220204636, 1.73773336], [3.01429486, 2.4406364, -3.44312334, -1.9173075], [4.9877429, -6.07382202, 6.09596968, -1.17617261], [-3.09937429, -3.2480278, -1.11466146, 3.65894794]], [[1.55535305, 3.06394267, 5.10202551, 1.00784862], [-1.21928763, -2.06880212, -4.80053806, -1.05981922], [-2.71434927, -2.71204138, -2.69678211, 5.0204649], [-0.791190505, 5.32497835, 3.55645466, 0.712908864], [4.05458784, -2.70867419, -3.32875013, -2.53978419]], [[-0.115624271, 3.07909942, -0.737454593, -5.84991312], [-11.6888456, -3.29018068, -3.93096113, -5.09515333], [1.53018463, 4.2541976, 6.96445084, 2.22642112], [-2.29474902, -0.823115527, 0.316226214, 6.18460131], [2.60330725, 1.22173953, 0.553669035, 0.0984589084]]]> : tensor<3x5x4xf32>
    return %0 : tensor<3x5x4xf32>
  }
}

