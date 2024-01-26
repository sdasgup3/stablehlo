func.func @rescale_impl1(%arg0: tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>)  -> tensor<2x2xi32> {

  %arg = stablehlo.bitcast_convert %arg0: (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi8>

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
  %c_value = stablehlo.convert %arg : (tensor<2x2xi8>) -> tensor<2x2xi64>
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

func.func @rescale_impl2(%arg0: tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)  -> tensor<2x2xi32> {
  %arg = stablehlo.bitcast_convert %arg0: (tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2xi8>
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
  %c_value = stablehlo.convert %arg : (tensor<2x2xi8>) -> tensor<2x2xi64>
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

func.func @rescale_impl3(%arg0: tensor<2x2xi32>) ->  tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>> {
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
  %result = stablehlo.bitcast_convert %result6: (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  return %result : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
}

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>
  %1 = stablehlo.uniform_quantize %arg1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>


  %r0 = func.call @rescale_impl1(%0) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32>
  %r1 = func.call @rescale_impl2(%1) : (tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2xi32>
  %add = "stablehlo.add"(%r0, %r1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %3 = func.call @rescale_impl3(%add) : (tensor<2x2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  %result = stablehlo.uniform_dequantize %3: (tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

func.func @stablehlo_add(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>
  %1 = stablehlo.uniform_quantize %arg1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>

  %2 = "stablehlo.add"(%0, %1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  %3 = stablehlo.uniform_dequantize %2 : (tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}
