/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_DIALECT_STABLEHLO_TYPES
#define STABLEHLO_DIALECT_STABLEHLO_TYPES

include "mlir/IR/AttrTypeBase.td"
include "stablehlo/dialect/StablehloDialect.td"

class StableHLO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<StableHLO_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def StableHLO_UniformQuantizedWithMultiplierAndShiftType : StableHLO_Type<"UniformQuantizedWithMultiplierAndShift", "uniform_multiplier_shift"> {
  let summary = "Uniform quantized type with integer mutipliers and shifts";
  let description = [{
    Uniform quantized type with integer mutipliers and shifts. This type is used
    to enable integer only arithmetic.
  }];

  let parameters = (ins
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    "int64_t":$quantizationDimension,
    ArrayRefParameter<"int64_t">:$multipliers,
    ArrayRefParameter<"int64_t">:$shifts,
    ArrayRefParameter<"int64_t">:$zero_points,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );

  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UniformQuantizedWithMultiplierAndShiftType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::mlir::Type storageType, ::mlir::Type expressedType,
        int64_t quantizationDimension, ::llvm::ArrayRef<int64_t> multipliers,
        ::llvm::ArrayRef<int64_t> shifts,
        ::llvm::ArrayRef<int64_t> zero_points, int64_t storageTypeMin,
        int64_t storageTypeMax) {
      return success();
    }
  }];

  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<` $storageType `` `:` `` $storageTypeMin `` `:` `` $storageTypeMax `` `:` `` $expressedType `` `:` $quantizationDimension `,` `[` $multipliers `]` `,` `[` $shifts `]` `,` `[` $zero_points `]` `>`";

}

#endif // STABLEHLO_DIALECT_STABLEHLO_TYPES
