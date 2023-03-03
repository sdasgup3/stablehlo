// RUN-DISABLED: stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo -inline | stablehlo-interpreter --interpret
// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x16x2xf32>, tensor<3x2x2xf32>)
    %1 = call @expected() : () -> tensor<1x32x2xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {pad = [[2, 1]], lhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x2xf32>, tensor<3x2x2xf32>) -> tensor<1x32x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1x32x2xf32>, tensor<1x32x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x16x2xf32>, tensor<3x2x2xf32>) {
    %0 = stablehlo.constant dense<[[[-1.85980642, -4.23970461], [-4.73045349, -0.530883729], [-4.16903257, 5.02416182], [-5.77677679, -0.975355386], [4.46340561, -2.29194045], [-0.758056521, -4.536098], [5.37831163, 0.168776125], [-3.56675458, 2.80054331], [1.13978767, 5.87573242], [-1.11913872, -5.70136929], [4.13804626, 2.45081449], [-2.25734782, -0.941460669], [4.14359236, -5.18526125], [2.85359073, -2.54740357], [-5.29141903, -0.421686381], [1.24744737, 2.0226624]]]> : tensor<1x16x2xf32>
    %1 = stablehlo.constant dense<[[[-5.78321695, -6.70092058], [-4.02411413, -1.5279355]], [[4.82193375, -1.90760541], [4.670856, -1.44685578]], [[-3.33052278, 1.00520051], [1.20746529, -0.450568587]]]> : tensor<3x2x2xf32>
    return %0, %1 : tensor<1x16x2xf32>, tensor<3x2x2xf32>
  }
  func.func private @expected() -> tensor<1x32x2xf32> {
    %0 = stablehlo.constant dense<[[[1.07483149, 0.0407993793], [-28.7709122, 9.68201828], [42.9305801, 14.4245548], [-25.2896137, 9.79195117], [49.445137, 26.0551052], [3.36433792, 0.68363142], [21.9545975, 14.8924065], [-32.4109802, 12.4310093], [19.7003803, 45.7192955], [10.8169212, -5.19830942], [-19.5422688, -25.125164], [-24.8427582, 8.00915241], [4.92898369, 17.3407784], [26.7221909, -10.503891], [-1.652240e+01, -41.1446571], [-4.11771965, 2.75197792], [12.6562634, 18.1197853], [32.9406815, -10.675602], [-33.3931427, -15.1714687], [-32.0266876, 10.383934], [1.859260e+01, 19.2658901], [31.4007874, -11.4397345], [-27.4122086, -3.331830e+01], [-15.2822094, 5.6682868], [-3.21807432, 23.0662575], [-4.23948097, -0.402014256], [-15.6770611, -15.8269262], [1.8612709, -1.75779963], [10.8621292, -20.3583546], [-27.4845085, 10.7040596], [30.5859833, 36.4442711], [15.4626732, -5.30613804]]]> : tensor<1x32x2xf32>
    return %0 : tensor<1x32x2xf32>
  }
}

