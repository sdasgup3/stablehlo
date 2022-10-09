// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: xor_op_test_si4
func.func @xor_op_test_si4() -> tensor<3xi4> {
  %0 = stablehlo.constant dense<[7, -8, -8]> : tensor<3xi4>
  %1 = stablehlo.constant dense<[0, 7, -8]> : tensor<3xi4>
  %2 = stablehlo.xor %0, %1 : tensor<3xi4>
  func.return %2 : tensor<3xi4>
  // CHECK-NEXT: tensor<3xi4>
  // CHECK-NEXT: 7 : i4
  // CHECK-NEXT: -1 : i4
  // CHECK-NEXT: 0 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_ui4
func.func @xor_op_test_ui4() -> tensor<3xui4> {
  %0 = stablehlo.constant dense<[0, 7, 15]> : tensor<3xui4>
  %1 = stablehlo.constant dense<15> : tensor<3xui4>
  %2 = stablehlo.xor %0, %1 : tensor<3xui4>
  func.return %2 : tensor<3xui4>
  // CHECK-NEXT: tensor<3xui4>
  // CHECK-NEXT: 15 : ui4
  // CHECK-NEXT: 8 : ui4
  // CHECK-NEXT: 0 : ui4
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_si8
func.func @xor_op_test_si8() -> tensor<3xi8> {
  %0 = stablehlo.constant dense<[127, -128, -128]> : tensor<3xi8>
  %1 = stablehlo.constant dense<[0, 127, -128]> : tensor<3xi8>
  %2 = stablehlo.xor %0, %1 : tensor<3xi8>
  func.return %2 : tensor<3xi8>
  // CHECK-NEXT: tensor<3xi8>
  // CHECK-NEXT: 127 : i8
  // CHECK-NEXT: -1 : i8
  // CHECK-NEXT: 0 : i8
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_ui8
func.func @xor_op_test_ui8() -> tensor<3xui8> {
  %0 = stablehlo.constant dense<[0, 127, 255]> : tensor<3xui8>
  %1 = stablehlo.constant dense<255> : tensor<3xui8>
  %2 = stablehlo.xor %0, %1 : tensor<3xui8>
  func.return %2 : tensor<3xui8>
  // CHECK-NEXT: tensor<3xui8>
  // CHECK-NEXT: 255 : ui8
  // CHECK-NEXT: 128 : ui8
  // CHECK-NEXT: 0 : ui8
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_si16
func.func @xor_op_test_si16() -> tensor<3xi16> {
  %0 = stablehlo.constant dense<[32767, -32768, -32768]> : tensor<3xi16>
  %1 = stablehlo.constant dense<[0, 32767, -32768]> : tensor<3xi16>
  %2 = stablehlo.xor %0, %1 : tensor<3xi16>
  func.return %2 : tensor<3xi16>
  // CHECK-NEXT: tensor<3xi16>
  // CHECK-NEXT: 32767 : i16
  // CHECK-NEXT: -1 : i16
  // CHECK-NEXT: 0 : i16
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_ui16
func.func @xor_op_test_ui16() -> tensor<3xui16> {
  %0 = stablehlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
  %1 = stablehlo.constant dense<65535> : tensor<3xui16>
  %2 = stablehlo.xor %0, %1 : tensor<3xui16>
  func.return %2 : tensor<3xui16>
  // CHECK-NEXT: tensor<3xui16>
  // CHECK-NEXT: 65535 : ui16
  // CHECK-NEXT: 32768 : ui16
  // CHECK-NEXT: 0 : ui16
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_si32
func.func @xor_op_test_si32() -> tensor<3xi32> {
  %0 = stablehlo.constant dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>
  %1 = stablehlo.constant dense<[0, 2147483647, -2147483648]> : tensor<3xi32>
  %2 = stablehlo.xor %0, %1 : tensor<3xi32>
  func.return %2 : tensor<3xi32>
  // CHECK-NEXT: tensor<3xi32>
  // CHECK-NEXT: 2147483647 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: 0 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_ui32
func.func @xor_op_test_ui32() -> tensor<3xui32> {
  %0 = stablehlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
  %1 = stablehlo.constant dense<4294967295> : tensor<3xui32>
  %2 = stablehlo.xor %0, %1 : tensor<3xui32>
  func.return %2 : tensor<3xui32>
  // CHECK-NEXT: tensor<3xui32>
  // CHECK-NEXT: 4294967295 : ui32
  // CHECK-NEXT: 2147483648 : ui32
  // CHECK-NEXT: 0 : ui32
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_si64
func.func @xor_op_test_si64() -> tensor<3xi64> {
  %0 = stablehlo.constant dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>
  %1 = stablehlo.constant dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>
  %2 = stablehlo.xor %0, %1 : tensor<3xi64>
  func.return %2 : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 9223372036854775807 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_ui64
func.func @xor_op_test_ui64() -> tensor<3xui64> {
  %0 = stablehlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
  %1 = stablehlo.constant dense<18446744073709551615> : tensor<3xui64>
  %2 = stablehlo.xor %0, %1 : tensor<3xui64>
  func.return %2 : tensor<3xui64>
  // CHECK-NEXT: tensor<3xui64>
  // CHECK-NEXT: 18446744073709551615 : ui64
  // CHECK-NEXT: 9223372036854775808 : ui64
  // CHECK-NEXT: 0 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_i1
func.func @xor_op_test_i1() -> tensor<4xi1> {
  %0 = stablehlo.constant dense<[false, false, true, true]> : tensor<4xi1>
  %1 = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %2 = stablehlo.xor %0, %1 : tensor<4xi1>
  func.return %2 : tensor<4xi1>
  // CHECK-NEXT: tensor<4xi1>
  // CHECK-NEXT: false
  // CHECK-NEXT: true
  // CHECK-NEXT: true
  // CHECK-NEXT: false
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_i1_splat_false
func.func @xor_op_test_i1_splat_false() -> tensor<2xi1> {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %2 = stablehlo.xor %0, %1 : tensor<2xi1>
  func.return %2 : tensor<2xi1>
  // CHECK-NEXT: tensor<2xi1>
  // CHECK-NEXT: false
  // CHECK-NEXT: true
}

// -----

// CHECK-LABEL: Evaluated results of function: xor_op_test_i1_splat_true
func.func @xor_op_test_i1_splat_true() -> tensor<2xi1> {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %2 = stablehlo.xor %0, %1 : tensor<2xi1>
  func.return %2 : tensor<2xi1>
  // CHECK-NEXT: tensor<2xi1>
  // CHECK-NEXT: true
  // CHECK-NEXT: false
}
