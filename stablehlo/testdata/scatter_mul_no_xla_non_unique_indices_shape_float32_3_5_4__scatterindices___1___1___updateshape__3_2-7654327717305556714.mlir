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
    %0 = stablehlo.constant dense<[[[-0.586145639, -0.794766545, -5.73632622, 5.40198946], [0.575104535, 2.103960e+00, 1.66051042, -0.395921677], [6.27761221, -1.9532938, 3.82681894, -0.0561447442], [0.0270300619, 1.31666398, -0.329216272, 1.4759407], [1.74938929, -0.591707051, -4.0440979, -0.783893167]], [[0.371694803, 5.96711159, -4.28136253, -0.321309835], [1.04414499, 0.49229002, 2.40397429, -0.0247075353], [1.27510989, -0.793288767, 4.5915184, 2.9026413], [-1.69950294, -0.588202894, -1.69419408, 2.20097733], [-3.01020193, -1.64370453, 3.60547423, 1.50933754]], [[-1.29320788, -0.989479362, -3.176296, -4.03768587], [1.54506338, 3.9280231, -3.12444448, 0.212974891], [0.89870727, -2.4756043, -1.10772073, 1.4060148], [-2.26019382, -1.85652709, 2.45127058, 3.24847722], [-1.71855044, -5.85817337, -0.234671399, -3.64879823]]]> : tensor<3x5x4xf32>
    %1 = stablehlo.constant dense<[[[-4.554280e-01, -4.41210222, -1.38938212, 1.80325449], [1.92210734, -0.310389876, -1.49253178, 5.68545055]], [[-1.49476552, 1.70856678, -2.88609719, 0.8887223], [3.71810484, 8.00417327, -4.00683117, 2.79141617]], [[-4.88299942, -3.02588463, -1.44837666, -2.94206476], [-0.71765691, 0.396610737, -0.367812574, -0.353884071]]]> : tensor<3x2x4xf32>
    return %0, %1 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> tensor<3x5x4xf32> {
    %0 = stablehlo.constant dense<[[[-0.586145639, -0.794766545, -5.73632622, 5.40198946], [-0.50343591, 2.88131404, 3.44339561, -4.0591135], [6.27761221, -1.9532938, 3.82681894, -0.0561447442], [0.0270300619, 1.31666398, -0.329216272, 1.4759407], [1.74938929, -0.591707051, -4.0440979, -0.783893167]], [[0.371694803, 5.96711159, -4.28136253, -0.321309835], [-5.80303907, 6.73239279, 27.7998104, -6.129430e-02], [1.27510989, -0.793288767, 4.5915184, 2.9026413], [-1.69950294, -0.588202894, -1.69419408, 2.20097733], [-3.01020193, -1.64370453, 3.60547423, 1.50933754]], [[-1.29320788, -0.989479362, -3.176296, -4.03768587], [5.4143939, -4.71401405, -1.66448891, 0.221738771], [0.89870727, -2.4756043, -1.10772073, 1.4060148], [-2.26019382, -1.85652709, 2.45127058, 3.24847722], [-1.71855044, -5.85817337, -0.234671399, -3.64879823]]]> : tensor<3x5x4xf32>
    return %0 : tensor<3x5x4xf32>
  }
}

