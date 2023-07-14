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
  let summary = "Uniform per-tensor quantized type with integer mutipliers and shifts";
  let description = [{
    Uniform quantized type with integer mutipliers and shifts. This type is used
    to enable integer only arithmetic.
  }];

  let parameters = (ins
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    "int32_t":$multiplier,
    "int32_t":$shift,
    "int64_t":$zeroPoint,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );

  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UniformQuantizedWithMultiplierAndShiftType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::mlir::Type storageType, ::mlir::Type expressedType,
        int32_t multiplier, int32_t shift, int64_t zeroPoint,
        int64_t storageTypeMin, int64_t storageTypeMax) {
      auto storageIntegerType = storageType.dyn_cast<IntegerType>();
      auto expressedFloatType = expressedType.dyn_cast<FloatType>();
      if (!storageIntegerType || !expressedFloatType) return failure();

      auto storageTypeWidth = storageIntegerType.getWidth();
      auto expressedTypeWidth = expressedFloatType.getWidth();

      // uniform_multiplier_shift_c1
      if (storageTypeWidth >= expressedTypeWidth) return failure();

      // uniform_multiplier_shift_c2, uniform_multiplier_shift_c3,
      // uniform_multiplier_shift_c4
      bool isSigned = !storageIntegerType.isUnsigned();
      int64_t defaultIntegerMin =
          isSigned ? llvm::minIntN(storageTypeWidth) : 0;
      int64_t defaultIntegerMax = isSigned ? llvm::maxIntN(expressedTypeWidth)
                                           : llvm::maxUIntN(expressedTypeWidth);
      if (storageTypeMin >= storageTypeMax ||
          storageTypeMin < defaultIntegerMin ||
          storageTypeMax > defaultIntegerMax)
        return failure();

      // uniform_multiplier_shift_c8, uniform_multiplier_shift_c9
      if (zeroPoint < storageTypeMin || zeroPoint > storageTypeMax)
        return failure();

      // uniform_multiplier_shift_c14, uniform_multiplier_shift_c15
      if (multiplier < 0 || shift <= 0) return failure();

      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def StableHLO_UniformQuantizedWithMultiplierAndShiftPerAxisType : StableHLO_Type<"UniformQuantizedWithMultiplierAndShiftPerAxis", "uniform_multiplier_shift_per_axis"> {
  let summary = "Uniform per-axis quantized type with integer mutipliers and shifts";
  let description = [{
    Uniform quantized type with integer mutipliers and shifts. This type is used
    to enable integer only arithmetic.
  }];

  let parameters = (ins
    "::mlir::Type":$storageType,
    "::mlir::Type":$expressedType,
    "int64_t":$quantizationDimension,
    ArrayRefParameter<"int32_t">:$multipliers,
    ArrayRefParameter<"int32_t">:$shifts,
    ArrayRefParameter<"int64_t">:$zeroPoints,
    "int64_t":$storageTypeMin,
    "int64_t":$storageTypeMax
  );

  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UniformQuantizedWithMultiplierAndShiftPerAxisType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::mlir::Type storageType, ::mlir::Type expressedType,
        int64_t quantizationDimension, ::llvm::ArrayRef<int32_t> multipliers,
        ::llvm::ArrayRef<int32_t> shifts, ::llvm::ArrayRef<int64_t> zeroPoints,
        int64_t storageTypeMin, int64_t storageTypeMax) {
      auto storageIntegerType = storageType.dyn_cast<IntegerType>();
      auto expressedFloatType = expressedType.dyn_cast<FloatType>();
      if (!storageIntegerType || !expressedFloatType) return failure();

      auto storageTypeWidth = storageIntegerType.getWidth();
      auto expressedTypeWidth = expressedFloatType.getWidth();

      // uniform_multiplier_shift_c1
      if (storageTypeWidth >= expressedTypeWidth) return failure();

      // uniform_multiplier_shift_c2, uniform_multiplier_shift_c3,
      // uniform_multiplier_shift_c4
      bool isSigned = !storageIntegerType.isUnsigned();
      int64_t defaultIntegerMin =
          isSigned ? llvm::minIntN(storageTypeWidth) : 0;
      int64_t defaultIntegerMax = isSigned ? llvm::maxIntN(expressedTypeWidth)
                                           : llvm::maxUIntN(expressedTypeWidth);
      if (storageTypeMin >= storageTypeMax ||
          storageTypeMin < defaultIntegerMin ||
          storageTypeMax > defaultIntegerMax)
        return failure();

      // uniform_multiplier_shift_c8, uniform_multiplier_shift_c9
      for (auto zeroPoint : zeroPoints)
        if (zeroPoint < storageTypeMin || zeroPoint > storageTypeMax)
          return failure();

      // uniform_multiplier_shift_c13
      if (shifts.size() != multipliers.size() ||
          shifts.size() != zeroPoints.size())
        return failure();

      // uniform_multiplier_shift_c14, uniform_multiplier_shift_c15
      for (auto multiplierShift : llvm::zip(multipliers, shifts))
        if (std::get<0>(multiplierShift) < 0 ||
            std::get<1>(multiplierShift) <= 0)
          return failure();

      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif // STABLEHLO_DIALECT_STABLEHLO_TYPES
