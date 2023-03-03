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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf32>, tensor<2x1xi32>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>) {
    %0 = stablehlo.constant dense<[[[-3.29685426, -0.14864704, -0.0231414065, 2.27885509], [7.49583197, -3.49551821, 1.60782123, -0.610424399], [-3.49572206, -4.361280e+00, 1.77353847, 1.70864093], [3.45424414, 2.1430428, 3.25784898, -2.03932452], [-1.32052255, -1.42487848, 1.34178674, 0.247616798]], [[-4.12772226, -2.59224343, -0.615145802, 0.0619651973], [3.31970811, -5.23061085, 5.13009882, 1.83198237], [4.61235476, -0.617777526, 4.99055815, 1.8221302], [0.845035731, 2.82430029, 3.2138741, 0.725496947], [-0.718414664, -3.253618, 3.155950e+00, 3.97462273]], [[-1.60465574, -0.970614433, 0.683007717, -4.48582602], [-2.55651212, -2.97039652, 1.52281618, 0.178371534], [-5.43927336, 2.46750164, -3.21760821, -1.23467827], [0.880077481, 1.00932026, 2.91633582, -2.90946794], [-1.40817416, 3.99042988, -1.68375289, -6.17447805]]]> : tensor<3x5x4xf32>
    %1 = stablehlo.constant dense<[[[2.19123769, -4.07414532, 0.836083174, 0.070098944], [-4.08188868, 5.02543783, -2.72866797, 4.60839367]], [[-8.3974781, 3.11233425, 7.93647337, -1.55597615], [-4.08823919, -3.5552485, -0.258735508, -2.97263885]], [[-1.50073183, -1.28478801, -6.26443815, -3.87654829], [-1.14941204, -3.29719043, 0.604487181, 0.458343476]]]> : tensor<3x2x4xf32>
    return %0, %1 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> tensor<3x5x4xf32> {
    %0 = stablehlo.constant dense<[[[-3.29685426, -0.14864704, -0.0231414065, 2.27885509], [5.60518122, -2.54422569, -0.284763575, 4.06806803], [-3.49572206, -4.361280e+00, 1.77353847, 1.70864093], [3.45424414, 2.1430428, 3.25784898, -2.03932452], [-1.32052255, -1.42487848, 1.34178674, 0.247616798]], [[-4.12772226, -2.59224343, -0.615145802, 0.0619651973], [-9.1660099, -5.67352486, 12.8078365, -2.69663262], [4.61235476, -0.617777526, 4.99055815, 1.8221302], [0.845035731, 2.82430029, 3.2138741, 0.725496947], [-0.718414664, -3.253618, 3.155950e+00, 3.97462273]], [[-1.60465574, -0.970614433, 0.683007717, -4.48582602], [-5.20665598, -7.55237483, -4.13713455, -3.23983335], [-5.43927336, 2.46750164, -3.21760821, -1.23467827], [0.880077481, 1.00932026, 2.91633582, -2.90946794], [-1.40817416, 3.99042988, -1.68375289, -6.17447805]]]> : tensor<3x5x4xf32>
    return %0 : tensor<3x5x4xf32>
  }
}

