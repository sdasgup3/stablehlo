/* Copyright 2023 The StableHLO Authors.
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

#include <cfenv>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/StablehloTypes.h"
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "canolicalize-quant-types"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEQUANTTYPESPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// StableHLO uniform quantization types --> StableHLO integer based types
//===----------------------------------------------------------------------===//

void convertScaleToMultiplierShift(double scale, int32_t &multiplier,
                                   int32_t &shift) {
  int32_t exponent;
  const double mantissa = std::frexp(scale, &exponent);
  std::fesetround(FE_TONEAREST);
  auto shiftedMantissa =
      static_cast<int64_t>(std::rint(mantissa * (int64_t(1) << 31)));

  if (shiftedMantissa == (int64_t(1) << 31)) {
    shiftedMantissa /= 2;
    exponent++;
  }

  int64_t adjustedMantissa = shiftedMantissa;

  if (exponent < -31) {
    exponent = 0;
    adjustedMantissa = 0;
  }
  if (exponent > 30) {
    exponent = 30;
    adjustedMantissa = (1LL << 31) - 1;
  }

  shift = (-exponent) + 31;
  multiplier = static_cast<int32_t>(adjustedMantissa);
}

class StablehloQuantTypeConverter : public TypeConverter {
 public:
  StablehloQuantTypeConverter() : TypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([&](FunctionType type) -> Type {
      SmallVector<Type> convertedInputs;
      SmallVector<Type> convertedResults;
      if (failed(convertTypes(type.getInputs(), convertedInputs))) return {};
      if (failed(convertTypes(type.getResults(), convertedResults))) return {};
      return FunctionType::get(type.getContext(), convertedInputs,
                               convertedResults);
    });
    addConversion([&](RankedTensorType type) -> Type {
      auto convertedElementType = convertType(type.getElementType());
      if (!convertedElementType) return {};
      return RankedTensorType::get(type.getShape(), convertedElementType,
                                   type.getEncoding());
    });
    addConversion([&](TupleType type) -> Type {
      SmallVector<Type> convertedTypes;
      if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
      return TupleType::get(type.getContext(), convertedTypes);
    });
    addConversion([&](quant::UniformQuantizedType type) -> Type {
      Type convertedStorageType = convertType(type.getStorageType());
      Type convertedExpressedType = convertType(type.getExpressedType());
      if (!convertedStorageType || !convertedExpressedType) return {};
      int32_t multiplier, shift;
      convertScaleToMultiplierShift(type.getScale(), multiplier, shift);
      return UniformQuantizedWithMultiplierAndShiftType::get(
          type.getContext(), convertedStorageType, convertedExpressedType,
          multiplier, shift, type.getZeroPoint(), type.getStorageTypeMin(),
          type.getStorageTypeMax());
    });
    addConversion([&](quant::UniformQuantizedPerAxisType type) -> Type {
      Type convertedStorageType = convertType(type.getStorageType());
      Type convertedExpressedType = convertType(type.getExpressedType());
      if (!convertedStorageType || !convertedExpressedType) return {};
      SmallVector<int32_t> multipliers, shifts;
      for (auto scale : type.getScales()) {
        convertScaleToMultiplierShift(scale, multipliers.emplace_back(),
                                      shifts.emplace_back());
      }
      return UniformQuantizedWithMultiplierAndShiftPerAxisType::get(
          type.getContext(), convertedStorageType, convertedExpressedType,
          type.getQuantizedDimension(), multipliers, shifts,
          type.getZeroPoints(), type.getStorageTypeMin(),
          type.getStorageTypeMax());
    });
    addConversion([&](UnrankedTensorType type) -> Type {
      auto convertedElementType = convertType(type.getElementType());
      if (!convertedElementType) return {};
      return UnrankedTensorType::get(convertedElementType);
    });
  }
};

class StablehloQuantOpConverter : public ConversionPattern {
 public:
  StablehloQuantOpConverter(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit*/ 0, ctx) {}

  // The dialect conversion framework will call this matchAndRewrite on each
  // Operation in the IR tree. This call matchAndRewrite needs to update the
  // Operation's results and child regions.
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Update the results.
    llvm::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new up.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region &new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(rewriter.convertRegionTypes(&new_region, *getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
    }
    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    return success();
  }
};

}  // namespace

struct StablehloCanonicalizeQuantTypesPass
    : public impl::StablehloCanonicalizeQuantTypesPassBase<
          StablehloCanonicalizeQuantTypesPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    ConversionTarget target(*context);

    // An addDynamicallyLegalDialect callback that declares a given operation as
    // legal only if its all operands and results are non-quantized types.
    auto isLegal = [](Operation *op) {
      auto is_not_quant = [](Type type) {
        return !getElementTypeOrSelf(type).isa<quant::UniformQuantizedType>() &&
               !getElementTypeOrSelf(type)
                    .isa<quant::UniformQuantizedPerAxisType>();
      };

      return llvm::all_of(op->getOperandTypes(), is_not_quant) &&
             llvm::all_of(op->getResultTypes(), is_not_quant);
    };
    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(isLegal);
    target.addLegalDialect<func::FuncDialect>();

    StablehloQuantTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    patterns.add<StablehloQuantOpConverter>(context, converter);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace stablehlo
}  // namespace mlir
