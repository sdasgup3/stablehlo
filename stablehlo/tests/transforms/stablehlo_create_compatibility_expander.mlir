// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file --stablehlo-create-compatibility-expander='target=1.0.0' --chlo-legalize-to-stablehlo | FileCheck %s --check-prefixes=CHECK
// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file --stablehlo-create-compatibility-expander='target=1.6.0' --chlo-legalize-to-stablehlo | FileCheck %s --check-prefixes=CHECK-NO-DOWNGRADE

// -----

// CHECK-LABEL @tan_op_non_complex
// CHECK: %[[sine0:.*]] = stablehlo.sine %arg0 : tensor<4xf64>
// CHECK-NEXT: %[[cosine1:.*]] = stablehlo.cosine %arg0 : tensor<4xf64>
// CHECK-NEXT: %[[div2:.*]] = stablehlo.divide %[[sine0]], %[[cosine1]] : tensor<4xf64>
// CHECK-NEXT: return %[[div2]] : tensor<4xf64>
func.func @tan_op_non_complex(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  // CHECK-NO-DOWNGRADE: stablehlo.tan %arg0 : tensor<4xf64>
  %1 = stablehlo.tan %arg0 : tensor<4xf64>
  func.return %1 : tensor<4xf64>
}

// -----

// CHECK-LABEL: @tan_op_complex
// CHECK: %[[cst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
// CHECK: %[[complex:.*]] = stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f64>>
// CHECK: %[[real:.*]] = stablehlo.real %[[complex]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[sine:.*]] = stablehlo.sine %[[real]] : tensor<4xf64>
// CHECK: %[[cosine:.*]] = stablehlo.cosine %[[real]] : tensor<4xf64>
// CHECK: %[[divide1:.*]] = stablehlo.divide %[[sine]], %[[cosine]] : tensor<4xf64>
// CHECK: %[[imag:.*]] = stablehlo.imag %[[complex]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[tanh:.*]] = stablehlo.tanh %[[imag]] : tensor<4xf64>
// CHECK: %[[complex2:.*]] = stablehlo.complex %[[divide1]], %[[tanh]] : tensor<4xcomplex<f64>>
// CHECK: %[[multiply:.*]] = stablehlo.multiply %[[divide1]], %[[tanh]] : tensor<4xf64>
// CHECK: %[[negate:.*]] = stablehlo.negate %[[multiply]] : tensor<4xf64>
// CHECK: %[[complex3:.*]] = stablehlo.complex %[[cst]], %[[negate]] : tensor<4xcomplex<f64>>
// CHECK: %[[divide2:.*]] = stablehlo.divide %[[complex2]], %[[complex3]] : tensor<4xcomplex<f64>>
// CHECK: %[[real2:.*]] = stablehlo.real %[[divide2]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: %[[imag2:.*]] = stablehlo.imag %[[divide2]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// CHECK: return %[[real2]], %[[imag2]] : tensor<4xf64>, tensor<4xf64>
func.func @tan_op_complex(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %0 = stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f64>>
  // CHECK-NO-DOWNGRADE: stablehlo.tan %0 : tensor<4xcomplex<f64>>
  %1 = stablehlo.tan %0 : tensor<4xcomplex<f64>>
  %2 = stablehlo.real %1 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
  %3 = stablehlo.imag %1 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
  func.return %2, %3 : tensor<4xf64>, tensor<4xf64>
}

// -----

// CHECK-LABEL: @gather_with_batching_dims
// CHECK-NEXT: %[[iota_dim1:.*]] = stablehlo.iota dim = 1 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[iota_dim0:.*]] = stablehlo.iota dim = 0 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[concat:.*]] = stablehlo.concatenate %[[iota_dim1]], %[[iota_dim0]], %arg1, dim = 3 : (tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>, tensor<4x3x5x2xi32>) -> tensor<4x3x5x4xi32>
// CHECK-NEXT: %[[gather:.*]] = "stablehlo.gather"(%arg0, %[[concat]]) <{
// CHECK-SAME:   dimension_numbers = #stablehlo.gather<
// CHECK-SAME:     offset_dims = [3], collapsed_slice_dims = [0, 1, 2, 3],
// CHECK-SAME:     start_index_map = [0, 2, 1, 3], index_vector_dim = 3>,
// CHECK-SAME:   indices_are_sorted = false,
// CHECK-SAME:   slice_sizes = array<i64: 1, 1, 1, 1, 8>
// CHECK-SAME: }> : (tensor<3x2x4x7x9xi32>, tensor<4x3x5x4xi32>) -> tensor<4x3x5x8xi32>
// CHECK-NEXT: return %[[gather]] : tensor<4x3x5x8xi32>
func.func @gather_with_batching_dims(%arg0: tensor<3x2x4x7x9xi32>, %arg1: tensor<4x3x5x2xi32>) -> tensor<4x3x5x8xi32> {
  // CHECK-NO-DOWNGRADE: operand_batching_dims = [0, 2]
  // CHECK-NO-DOWNGRADE: start_indices_batching_dims = [1, 0]
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1, 3],
      operand_batching_dims = [0, 2],
      start_indices_batching_dims = [1, 0],
      start_index_map = [1, 3],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 1, 1, 1, 1, 8>,
    indices_are_sorted = true
  } : (tensor<3x2x4x7x9xi32>, tensor<4x3x5x2xi32>) -> tensor<4x3x5x8xi32>
  func.return %0 : tensor<4x3x5x8xi32>
}

// -----

// CHECK-LABEL: @gather_with_batching_no_index_vector_dim
// CHECK-NEXT: %[[iota_dim1:.*]] = stablehlo.iota dim = 1 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[iota_dim0:.*]] = stablehlo.iota dim = 0 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[reshape:.*]] = stablehlo.reshape %arg1 : (tensor<4x3x5xi32>) -> tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[concat:.*]] = stablehlo.concatenate %[[iota_dim1]], %[[iota_dim0]], %[[reshape]], dim = 3 : (tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>) -> tensor<4x3x5x3xi32>
// CHECK-NEXT: %[[gather:.*]] = "stablehlo.gather"(%arg0, %[[concat]]) <{
// CHECK-SAME:   dimension_numbers = #stablehlo.gather<
// CHECK-SAME:     offset_dims = [3], collapsed_slice_dims = [0, 1, 2],
// CHECK-SAME:     start_index_map = [0, 2, 1], index_vector_dim = 3>,
// CHECK-SAME:   indices_are_sorted = false,
// CHECK-SAME:   slice_sizes = array<i64: 1, 1, 1, 8>
// CHECK-SAME: }> : (tensor<3x2x4x9xi32>, tensor<4x3x5x3xi32>) -> tensor<4x3x5x8xi32>
// CHECK-NEXT: return %[[gather]] : tensor<4x3x5x8xi32>
func.func @gather_with_batching_no_index_vector_dim(%arg0: tensor<3x2x4x9xi32>, %arg1: tensor<4x3x5xi32>) -> tensor<4x3x5x8xi32> {
  // CHECK-NO-DOWNGRADE: operand_batching_dims = [0, 2]
  // CHECK-NO-DOWNGRADE: start_indices_batching_dims = [1, 0]
  %0 = "stablehlo.gather"(%arg0, %arg1) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0, 2],
      start_indices_batching_dims = [1, 0],
      start_index_map = [1],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 1, 1, 1, 8>,
    indices_are_sorted = true
  }> : (tensor<3x2x4x9xi32>, tensor<4x3x5xi32>) -> tensor<4x3x5x8xi32>
  func.return %0 : tensor<4x3x5x8xi32>
}

// -----

// CHECK-LABEL: @gather_with_batching_dim_size_zero
// CHECK-NEXT: %[[iota:.*]] = stablehlo.iota dim = 0 : tensor<0x3x5x1xi32>
// CHECK-NEXT: %[[concat:.*]] = stablehlo.concatenate %[[iota]], %arg1, dim = 3 : (tensor<0x3x5x1xi32>, tensor<0x3x5x1xi32>) -> tensor<0x3x5x2xi32>
// CHECK-NEXT: %[[gather:.*]] = "stablehlo.gather"(%arg0, %[[concat]]) <{
// CHECK-SAME:   dimension_numbers = #stablehlo.gather<
// CHECK-SAME:     offset_dims = [3], collapsed_slice_dims = [0, 1],
// CHECK-SAME:     start_index_map = [0, 1], index_vector_dim = 3>,
// CHECK-SAME:   indices_are_sorted = false,
// CHECK-SAME:   slice_sizes = array<i64: 0, 1, 8>
// CHECK-SAME: }> : (tensor<0x2x9xi32>, tensor<0x3x5x2xi32>) -> tensor<0x3x5x8xi32>
// CHECK-NEXT: return %[[gather]] : tensor<0x3x5x8xi32>
func.func @gather_with_batching_dim_size_zero(%arg0: tensor<0x2x9xi32>, %arg1: tensor<0x3x5x1xi32>) -> tensor<0x3x5x8xi32> {
  // CHECK-NO-DOWNGRADE: operand_batching_dims = [0]
  // CHECK-NO-DOWNGRADE: start_indices_batching_dims = [0]
  %0 = "stablehlo.gather"(%arg0, %arg1) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 0, 1, 8>,
    indices_are_sorted = true
  }> : (tensor<0x2x9xi32>, tensor<0x3x5x1xi32>) -> tensor<0x3x5x8xi32>
  func.return %0 : tensor<0x3x5x8xi32>
}

// -----

// CHECK-LABEL: @scatter_with_batching_dims
// CHECK-NEXT: %[[iota_dim1:.*]] = stablehlo.iota dim = 1 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[iota_dim0:.*]] = stablehlo.iota dim = 0 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[concat:.*]] = stablehlo.concatenate %[[iota_dim1]], %[[iota_dim0]], %arg1, dim = 3 : (tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>, tensor<4x3x5x2xi32>) -> tensor<4x3x5x4xi32>
// CHECK-NEXT: %[[scatter:.*]] = "stablehlo.scatter"(%arg0, %[[concat]], %arg2) <{
// CHECK-SAME:   indices_are_sorted = false,
// CHECK-SAME:   dimension_numbers = #stablehlo.scatter<
// CHECK-SAME:     update_window_dims = [3], inserted_window_dims = [0, 1, 2, 3],
// CHECK-SAME:     scatter_dims_to_operand_dims = [0, 2, 1, 3], index_vector_dim = 3>,
// CHECK-SAME:   unique_indices = false}>
// CHECK:      (tensor<3x2x4x7x9xi32>, tensor<4x3x5x4xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32>
// CHECK-NEXT: return %[[scatter]] : tensor<3x2x4x7x9xi32>
func.func @scatter_with_batching_dims(%arg0: tensor<3x2x4x7x9xi32>, %arg1: tensor<4x3x5x2xi32>, %arg2: tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32> {
  // CHECK-NO-DOWNGRADE: input_batching_dims = [0, 2]
  // CHECK-NO-DOWNGRADE: scatter_indices_batching_dims = [1, 0]
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [3],
      inserted_window_dims = [1, 3],
      input_batching_dims = [0, 2],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [1, 3],
      index_vector_dim = 3
    >,
    unique_indices = false
  }> ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    stablehlo.return %arg4 : tensor<i32>
  }) : (tensor<3x2x4x7x9xi32>, tensor<4x3x5x2xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32>
  func.return %0 : tensor<3x2x4x7x9xi32>
}

// -----

// CHECK-LABEL: @scatter_with_batching_no_index_vector_dim
// CHECK-NEXT: %[[iota_dim1:.*]] = stablehlo.iota dim = 1 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[iota_dim0:.*]] = stablehlo.iota dim = 0 : tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[reshape:.*]] = stablehlo.reshape %arg1 : (tensor<4x3x5xi32>) -> tensor<4x3x5x1xi32>
// CHECK-NEXT: %[[concat:.*]] = stablehlo.concatenate %[[iota_dim1]], %[[iota_dim0]], %[[reshape]], dim = 3 : (tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>) -> tensor<4x3x5x3xi32>
// CHECK-NEXT: %[[scatter:.*]] = "stablehlo.scatter"(%arg0, %[[concat]], %arg2) <{
// CHECK-SAME:   indices_are_sorted = false,
// CHECK-SAME:   dimension_numbers = #stablehlo.scatter<
// CHECK-SAME:     update_window_dims = [3], inserted_window_dims = [0, 1, 2],
// CHECK-SAME:     scatter_dims_to_operand_dims = [0, 2, 1], index_vector_dim = 3>,
// CHECK-SAME:   unique_indices = true}>
// CHECK:      (tensor<3x2x4x9xi32>, tensor<4x3x5x3xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x9xi32>
// CHECK-NEXT: return %[[scatter]] : tensor<3x2x4x9xi32>
func.func @scatter_with_batching_no_index_vector_dim(%arg0: tensor<3x2x4x9xi32>, %arg1: tensor<4x3x5xi32>, %arg2: tensor<4x3x5x8xi32>) -> tensor<3x2x4x9xi32> {
  // CHECK-NO-DOWNGRADE: input_batching_dims = [0, 2]
  // CHECK-NO-DOWNGRADE: scatter_indices_batching_dims = [1, 0]
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [3],
      inserted_window_dims = [1],
      input_batching_dims = [0, 2],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 3
    >,
    unique_indices = true
  }> ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    stablehlo.return %arg4 : tensor<i32>
  }) : (tensor<3x2x4x9xi32>, tensor<4x3x5xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x9xi32>
  func.return %0 : tensor<3x2x4x9xi32>
}
