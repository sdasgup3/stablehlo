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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf32>, tensor<2x1xi32>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>) {
    %0 = stablehlo.constant dense<[[[3.99883699, -2.2889142, -2.0827148, -1.82966304], [2.87515497, -1.33448362, -4.25694275, -0.745610237], [0.706648886, -5.96034241, -2.92864132, -3.2561903], [-1.96890676, -4.07475281, 0.892564774, -0.136754081], [2.16252565, 2.82959247, 0.241373911, 1.5832082]], [[0.938377619, 3.14659786, -2.43214631, -3.38430643], [4.48745966, -0.0925513729, 9.81172275, -2.97779131], [-1.40838492, 0.999705493, -0.52686882, -0.30590868], [-2.67220736, 0.965841412, -2.58999181, -4.1852746], [0.470741183, 2.158870e+00, 1.67324352, -1.67577124]], [[4.046772, 0.414446115, 3.26342845, 3.56899834], [1.92945814, 5.76233768, -0.648235559, 2.09894061], [6.4382162, -7.6092186, 1.19895208, -0.811845123], [-0.446528763, 3.58915257, -2.98265624, 7.19873953], [6.30973768, 0.781765699, -0.130891308, -3.0058794]]]> : tensor<3x5x4xf32>
    %1 = stablehlo.constant dense<[[[3.66922283, -6.8615446, 3.04733062, -0.684687197], [4.26369715, -4.69076586, -4.140900e+00, -3.98404884]], [[-2.26237869, 5.15238094, 9.41437149, 4.35512114], [1.6855334, -1.15859449, -1.4864161, 4.03176641]], [[9.24005508, -0.375687599, -0.09597525, 0.772582888], [5.815930e+00, -3.25011778, -2.16552401, -2.11278081]]]> : tensor<3x2x4xf32>
    return %0, %1 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> tensor<3x5x4xf32> {
    %0 = stablehlo.constant dense<[[[3.99883699, -2.2889142, -2.0827148, -1.82966304], [44.9802322, -42.9515572, 53.7170486, -2.03389597], [0.706648886, -5.96034241, -2.92864132, -3.2561903], [-1.96890676, -4.07475281, 0.892564774, -0.136754081], [2.16252565, 2.82959247, 0.241373911, 1.5832082]], [[0.938377619, 3.14659786, -2.43214631, -3.38430643], [-17.1120968, 0.552487254, -137.302032, -52.2865372], [-1.40838492, 0.999705493, -0.52686882, -0.30590868], [-2.67220736, 0.965841412, -2.58999181, -4.1852746], [0.470741183, 2.158870e+00, 1.67324352, -1.67577124]], [[4.046772, 0.414446115, 3.26342845, 3.56899834], [103.688133, 7.03598117, -0.134727135, -3.42609715], [6.4382162, -7.6092186, 1.19895208, -0.811845123], [-0.446528763, 3.58915257, -2.98265624, 7.19873953], [6.30973768, 0.781765699, -0.130891308, -3.0058794]]]> : tensor<3x5x4xf32>
    return %0 : tensor<3x5x4xf32>
  }
}

