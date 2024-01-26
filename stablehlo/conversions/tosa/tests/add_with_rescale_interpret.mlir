func.func @rescale_impl1(%arg0: tensor<2x2xi8>)  -> tensor<2x2xi32> {
  %multiplier = stablehlo.constant dense<1431655765> : tensor<2x2xi32>
  %shift = stablehlo.constant dense<13> : tensor<2x2xi32>
  %input_zp = stablehlo.constant dense<-1> : tensor<2x2xi32>
  %output_zp = stablehlo.constant dense<0> : tensor<2x2xi32>
  %ones = stablehlo.constant dense<1> : tensor<2x2xi64>
  %min = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  %max = stablehlo.constant dense<2147483647> : tensor<2x2xi32>

  // conversions
  %c_multiplier = stablehlo.convert %multiplier : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_shift = stablehlo.convert %shift : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_input_zp = stablehlo.convert %input_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_output_zp = stablehlo.convert %output_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_value = stablehlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xi64>
  %c_max = stablehlo.convert %max : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_min = stablehlo.convert %min : (tensor<2x2xi32>) -> tensor<2x2xi64>


  // value - input_zp
  %value = stablehlo.subtract %c_value, %c_input_zp : tensor<2x2xi64>


  // (shift - 1)
  %adjusted_shift = stablehlo.subtract %c_shift, %ones : tensor<2x2xi64>

  // 1 << (shift -1)
  %round = stablehlo.shift_left %ones, %adjusted_shift : tensor<2x2xi64>

  // value * multiplier
  %result1 = stablehlo.multiply %value, %c_multiplier : tensor<2x2xi64>

  // value * multiplier + round
  %result2 = stablehlo.add %result1, %round : tensor<2x2xi64>

  // (value * multiplier + round) >> c_shift
  %result3 = stablehlo.shift_right_arithmetic %result2, %c_shift : tensor<2x2xi64>

  // (value * multiplier + round) >> c_shift + output_zp
  %result4 = stablehlo.add %result3, %c_output_zp : tensor<2x2xi64>

  // clamp to destination type
  %result5 = stablehlo.clamp %c_min, %result4, %c_max : tensor<2x2xi64>
  %result6 = stablehlo.convert %result5 : (tensor<2x2xi64>) -> tensor<2x2xi32>

  return %result6 : tensor<2x2xi32>
}

func.func @rescale_impl2(%arg0: tensor<2x2xi8>)  -> tensor<2x2xi32> {
  %multiplier = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  %shift = stablehlo.constant dense<11> : tensor<2x2xi32>
  %input_zp = stablehlo.constant dense<-1> : tensor<2x2xi32>
  %output_zp = stablehlo.constant dense<0> : tensor<2x2xi32>
  %ones = stablehlo.constant dense<1> : tensor<2x2xi64>
  %min = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  %max = stablehlo.constant dense<2147483647> : tensor<2x2xi32>

  %c_multiplier = stablehlo.convert %multiplier : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_shift = stablehlo.convert %shift : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_input_zp = stablehlo.convert %input_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_output_zp = stablehlo.convert %output_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_value = stablehlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xi64>
  %c_max = stablehlo.convert %max : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_min = stablehlo.convert %min : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %value = stablehlo.subtract %c_value, %c_input_zp : tensor<2x2xi64>
  %adjusted_shift = stablehlo.subtract %c_shift, %ones : tensor<2x2xi64>
  %round = stablehlo.shift_left %ones, %adjusted_shift : tensor<2x2xi64>
  %result1 = stablehlo.multiply %value, %c_multiplier : tensor<2x2xi64>
  %result2 = stablehlo.add %result1, %round : tensor<2x2xi64>
  %result3 = stablehlo.shift_right_arithmetic %result2, %c_shift : tensor<2x2xi64>
  %result4 = stablehlo.add %result3, %c_output_zp : tensor<2x2xi64>
  %result5 = stablehlo.clamp %c_min, %result4, %c_max : tensor<2x2xi64>
  %result6 = stablehlo.convert %result5 : (tensor<2x2xi64>) -> tensor<2x2xi32>
  return %result6 : tensor<2x2xi32>
}

func.func @rescale_impl3(%arg0: tensor<2x2xi32>) ->  tensor<2x2xi8> {
  %multiplier = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  %shift = stablehlo.constant dense<50> : tensor<2x2xi32>
  %input_zp = stablehlo.constant dense<-1> : tensor<2x2xi32>
  %output_zp = stablehlo.constant dense<0> : tensor<2x2xi32>
  %ones = stablehlo.constant dense<1> : tensor<2x2xi64>
  %min = stablehlo.constant dense<-128> : tensor<2x2xi8>
  %max = stablehlo.constant dense<127> : tensor<2x2xi8>

  %c_multiplier = stablehlo.convert %multiplier : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_shift = stablehlo.convert %shift : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_input_zp = stablehlo.convert %input_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_output_zp = stablehlo.convert %output_zp : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_value = stablehlo.convert %arg0 : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %c_max = stablehlo.convert %max : (tensor<2x2xi8>) -> tensor<2x2xi64>
  %c_min = stablehlo.convert %min : (tensor<2x2xi8>) -> tensor<2x2xi64>
  %value = stablehlo.subtract %c_value, %c_input_zp : tensor<2x2xi64>
  %adjusted_shift = stablehlo.subtract %c_shift, %ones : tensor<2x2xi64>
  %round = stablehlo.shift_left %ones, %adjusted_shift : tensor<2x2xi64>
  %result1 = stablehlo.multiply %value, %c_multiplier : tensor<2x2xi64>
  %result2 = stablehlo.add %result1, %round : tensor<2x2xi64>
  %result3 = stablehlo.shift_right_arithmetic %result2, %c_shift : tensor<2x2xi64>
  %result4 = stablehlo.add %result3, %c_output_zp : tensor<2x2xi64>
  %result5 = stablehlo.clamp %c_min, %result4, %c_max : tensor<2x2xi64>
  %result6 = stablehlo.convert %result5 : (tensor<2x2xi64>) -> tensor<2x2xi8>
  return %result6 : tensor<2x2xi8>
}

func.func @main() -> (tensor<2x2xf32>) {
  // inputs
  %arg0 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
  %arg1 = stablehlo.constant dense<9.0> : tensor<2x2xf32>

  // quantization parameters
  %0 = stablehlo.constant dense<1.500000e-01> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<-1.000000e+00> : tensor<2x2xf32>
  %2 = stablehlo.constant dense<13.33> : tensor<2x2xf32>
  %3 = stablehlo.constant dense<-1> : tensor<2x2xi32>
  %4 = stablehlo.constant dense<4.000000e+01> : tensor<2x2xf32>
  %min = stablehlo.constant dense<-128> : tensor<2x2xi32>
  %max = stablehlo.constant dense<127> : tensor<2x2xi32>

  // quantize arg0
  %5 = stablehlo.multiply %arg0, %4 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %6 = stablehlo.convert %5 : (tensor<2x2xf32>) -> tensor<2x2xi32>
  %7 = stablehlo.add %6, %3 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %clamp1 = stablehlo.clamp %min, %7, %max : tensor<2x2xi32>
  %8 = stablehlo.convert %clamp1 : (tensor<2x2xi32>) -> tensor<2x2xi8>

  // quantize arg1
  %9 = stablehlo.multiply %arg1, %2: (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %10 = stablehlo.convert %9 : (tensor<2x2xf32>) -> tensor<2x2xi32>
  %11 = stablehlo.add %10, %3 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %clamp2 = stablehlo.clamp %min, %11, %max : tensor<2x2xi32>
  %12 = stablehlo.convert %clamp2 : (tensor<2x2xi32>) -> tensor<2x2xi8>


  %r0 = func.call @rescale_impl1(%8) : (tensor<2x2xi8>) -> tensor<2x2xi32>
  %r1 = func.call @rescale_impl2(%12) : (tensor<2x2xi8>) -> tensor<2x2xi32>
  %add = "stablehlo.add"(%r0, %r1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %r3 = func.call @rescale_impl3(%add) : (tensor<2x2xi32>) -> tensor<2x2xi8>

  %13 = stablehlo.convert %r3 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  %14 = stablehlo.subtract %13, %1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %15 = stablehlo.multiply %14, %0  : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %15 : tensor<2x2xf32>
}
