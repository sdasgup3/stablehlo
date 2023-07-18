/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

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

#include <fenv.h>

#include <memory>
#include <utility>

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/StablehloTypes.h"

#define PASS_NAME "stablehlo-legalize-to-tosa"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOTOSAPASS
#include "stablehlo/conversions/tosa/transforms/Passes.h.inc"
#include "stablehlo/conversions/tosa/transforms/StablehloLegalizeToTosa.pdll.h.inc"

namespace {

struct StablehloLegalizeToTosaPass
    : impl::StablehloLegalizeToTosaPassBase<StablehloLegalizeToTosaPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext* ctx) override;

 private:
  FrozenRewritePatternSet patterns;
};

struct ConvertStablehloCompareOp
    : public OpRewritePattern<stablehlo::CompareOp> {
  using OpRewritePattern<stablehlo::CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompareOp op,
                                PatternRewriter& rewriter) const override {
    auto direction = op.getComparisonDirection();
    auto resultType = op->getResultTypes().front();

    switch (direction) {
      case stablehlo::ComparisonDirection::EQ: {
        rewriter.replaceOpWithNewOp<tosa::EqualOp>(op, resultType, op.getLhs(),
                                                   op.getRhs());
        break;
      }
      case stablehlo::ComparisonDirection::NE: {
        auto equalOp = rewriter.create<tosa::EqualOp>(op->getLoc(), resultType,
                                                      op.getLhs(), op.getRhs());
        rewriter.replaceOpWithNewOp<tosa::LogicalNotOp>(op, resultType,
                                                        equalOp);
        break;
      }
      default: {
        return rewriter.notifyMatchFailure(
            op, "comparison direction not yet implemented");
      }
    }
    return success();
  }
};

// TODO(jennik): Move this lowering to PDLL when variadic tensors are supported.
struct ConvertStablehloConcatenateOp
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern<stablehlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::ConcatOp>(
        op, op.getResult().getType(), op.getInputs(), op.getDimension());
    return success();
  }
};

struct ConvertStablehloDotOp : public OpRewritePattern<stablehlo::DotOp> {
  using OpRewritePattern<stablehlo::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotOp op,
                                PatternRewriter& rewriter) const override {
    auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsType = op.getRhs().getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, "input tensors are not ranked");
    }

    auto resultType = op.getResult().getType().dyn_cast<ShapedType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "result tensor does not have shape");
    }

    if (lhsType.getElementType() != rhsType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs element types must match");
    }

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto resultShape = resultType.getShape();
    llvm::SmallVector<int64_t, 3> lhsReshape;
    llvm::SmallVector<int64_t, 3> rhsReshape;
    llvm::SmallVector<int64_t, 3> matMulShape;

    // tosa.matmul requires input tensors to have a rank of 3, so lhs and rhs
    // need to be reshaped first.
    if (lhsType.getRank() == 1) {
      // Reshape lhs to [1, 1, N].
      lhsReshape = {1, 1, lhsShape[0]};
      if (rhsType.getRank() == 1) {
        // Reshape rhs to [1, N, 1].
        rhsReshape = {1, rhsShape[0], 1};
        // MatMul shape is [1, 1, 1].
        matMulShape = {1, 1, 1};
      } else if (rhsType.getRank() == 2) {
        // Reshape rhs to [1, N, K].
        rhsReshape = {1, rhsShape[0], rhsShape[1]};
        // MatMul shape is [1, 1, K].
        matMulShape = {1, 1, rhsShape[1]};
      } else {
        return rewriter.notifyMatchFailure(op, "rhs must have rank of 1 or 2");
      }
    } else if (lhsType.getRank() == 2) {
      // Reshape lhs to [1, M, K].
      lhsReshape = {1, lhsShape[0], lhsShape[1]};
      if (rhsType.getRank() == 1) {
        // Reshape rhs to [1, K, 1].
        rhsReshape = {1, rhsShape[0], 1};
        // MatMul shape is [1, M, 1].
        matMulShape = {1, lhsShape[0], 1};
      } else if (rhsType.getRank() == 2) {
        // Reshape rhs to [1, K, N].
        rhsReshape = {1, rhsShape[0], rhsShape[1]};
        // MatMul shape is [1, M, N].
        matMulShape = {1, lhsShape[0], rhsShape[1]};
      } else {
        return rewriter.notifyMatchFailure(op, "rhs must have rank of 1 or 2");
      }
    } else {
      return rewriter.notifyMatchFailure(op, "lhs must have rank of 1 or 2");
    }

    auto lhsReshapeType =
        RankedTensorType::get(lhsReshape, lhsType.getElementType());
    auto lhsReshapeOp = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), lhsReshapeType, op.getLhs(),
        rewriter.getDenseI64ArrayAttr(lhsReshape));

    auto rhsReshapeType =
        RankedTensorType::get(rhsReshape, rhsType.getElementType());
    auto rhsReshapeOp = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), rhsReshapeType, op.getRhs(),
        rewriter.getDenseI64ArrayAttr(rhsReshape));

    auto matMulType =
        RankedTensorType::get(matMulShape, lhsType.getElementType());
    auto matMulOp = rewriter.create<tosa::MatMulOp>(op->getLoc(), matMulType,
                                                    lhsReshapeOp, rhsReshapeOp);

    // Reshape the matmul result back to the original result shape.
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, matMulOp, rewriter.getDenseI64ArrayAttr(resultShape));
    return success();
  }
};

// TODO(jennik): Consider the case of a non-constant expansion.
struct ConvertStablehloIotaOp : public OpRewritePattern<stablehlo::IotaOp> {
  using OpRewritePattern<stablehlo::IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::IotaOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType();
    auto elementType = resultType.cast<ShapedType>().getElementType();
    auto resultRankedType = resultType.dyn_cast<RankedTensorType>();

    if (!resultRankedType) {
      return rewriter.notifyMatchFailure(op, "result tensor must be ranked");
    }
    if (!resultRankedType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "result tensor must be static");
    }

    auto resultShape = resultRankedType.getShape();
    auto iotaDimension = op.getIotaDimension();
    int64_t iotaArrayLength = resultShape[iotaDimension];

    // Create a const op of [0, 1, 2...iotaArrayLength - 1] to be tiled.
    llvm::SmallVector<Attribute, 4> constValues;
    constValues.resize(iotaArrayLength);
    for (int i = 0; i < iotaArrayLength; i++) {
      if (elementType.isa<FloatType>()) {
        constValues[i] = rewriter.getFloatAttr(elementType, i);
      } else {
        constValues[i] = rewriter.getIntegerAttr(elementType, i);
      }
    }

    RankedTensorType constType =
        RankedTensorType::get(iotaArrayLength, elementType);
    auto constOp = rewriter.create<tosa::ConstOp>(
        op.getLoc(), constType, DenseElementsAttr::get(constType, constValues));

    // Create the multiples attr for the tile op, where all dimensions except
    // the iota dimension are multiplied.
    llvm::SmallVector<int64_t, 4> tileMultiples;
    size_t tileMultiplesSize = resultShape.size();
    tileMultiples.resize(tileMultiplesSize);

    for (size_t i = 0; i < tileMultiplesSize; i++) {
      if (i == iotaDimension) {
        tileMultiples[i] = 1;
      } else {
        tileMultiples[i] = resultShape[i];
      }
    }

    // Tile the const array to the result shape of the iota op.
    rewriter.replaceOpWithNewOp<tosa::TileOp>(
        op, resultType, constOp, rewriter.getDenseI64ArrayAttr(tileMultiples));
    return success();
  }
};

// This legalization supports the case where the Stablehlo start_indices
// directly map to the TOSA indices.
struct ConvertStablehloGatherOp : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern<stablehlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter& rewriter) const override {
    // The input operand must be 3D, with shape [N, K, C].
    auto operand = op.getOperand();
    auto operandType = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandType) {
      return rewriter.notifyMatchFailure(op, "requires ranked operand shape");
    }
    if (operandType.getRank() != 3) {
      return rewriter.notifyMatchFailure(op, "operand must have rank of 3");
    }

    // The indices tensor must be 2D, with shape [N, W].
    auto startIndices = op.getStartIndices();
    auto startIndicesType = startIndices.getType().dyn_cast<RankedTensorType>();
    if (!startIndicesType) {
      return rewriter.notifyMatchFailure(op,
                                         "requires ranked start_indices shape");
    }
    if (startIndicesType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "start_indices must have rank of 2");
    }

    // The result tensor must be 3D, with shape [N, W, C].
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked output shape");
    }
    if (resultType.getRank() != 3) {
      return rewriter.notifyMatchFailure(op, "result must have rank of 3");
    }

    auto operandShape = operand.getType().getShape();
    auto startIndicesShape = startIndices.getType().getShape();
    auto resultShape = resultType.getShape();

    if (startIndicesShape[0] != resultShape[0] ||
        startIndicesShape[1] != resultShape[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "start_indices and result must have "
                                         "same number of batches and indices");
    }

    if (operandShape[0] != resultShape[0] ||
        operandShape[2] != resultShape[2]) {
      return rewriter.notifyMatchFailure(op,
                                         "operand and result must have same "
                                         "number of batches and data channels");
    }

    auto startIndexMap = op.getDimensionNumbers().getStartIndexMap();
    for (const auto& startIndex : llvm::enumerate(startIndexMap)) {
      if (startIndex.value() != static_cast<int64_t>(startIndex.index())) {
        return rewriter.notifyMatchFailure(op,
                                           "start_index_map must be in order");
      }
    }

    rewriter.replaceOpWithNewOp<tosa::GatherOp>(op, resultType, operand,
                                                startIndices);
    return success();
  }
};

struct ConvertStablehloReduceOp : public OpRewritePattern<stablehlo::ReduceOp> {
  using OpRewritePattern<stablehlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();

    // To lower to a tosa.reduce_* op, the body should contain the reduce op
    // and a return op.
    if (bodyBlock.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op, "body required to contain 2 ops");
    }

    auto operand = op.getInputs().front();
    ShapedType inputType = operand.getType().cast<ShapedType>();
    Operation& innerOp = bodyBlock.front();
    uint64_t dimension = op.getDimensions().getValues<uint64_t>().begin()[0];
    SmallVector<int64_t> innerShape(inputType.getShape());
    innerShape[dimension] = 1;
    Type innerTy = inputType.clone(innerShape);

    Value reduceOpResult;
    if (isa<stablehlo::AddOp>(innerOp)) {
      reduceOpResult =
          rewriter
              .create<tosa::ReduceSumOp>(op->getLoc(), innerTy, operand,
                                         rewriter.getI64IntegerAttr(dimension))
              .getResult();
    } else if (isa<stablehlo::MaxOp>(innerOp)) {
      reduceOpResult =
          rewriter
              .create<tosa::ReduceMaxOp>(op->getLoc(), innerTy, operand,
                                         rewriter.getI64IntegerAttr(dimension))
              .getResult();
    } else {
      return rewriter.notifyMatchFailure(
          op, "reducing along a " + innerOp.getName().getStringRef().str() +
                  " op not supported");
    }

    // TOSA reduce ops do not remove the dimension being reduced, so reshape
    // the reduced output and remove the reduction dimension.
    llvm::SmallVector<int64_t, 2> outputShape;
    int outputShapeLength = innerShape.size() - 1;
    outputShape.resize(outputShapeLength);
    for (int64_t i = 0; i < outputShapeLength; i++) {
      if (i < static_cast<int64_t>(dimension)) {
        outputShape[i] = innerShape[i];
      } else {
        outputShape[i] = innerShape[i + 1];
      }
    }

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, op.getResultTypes().front(), reduceOpResult,
        rewriter.getDenseI64ArrayAttr(outputShape));

    return success();
  }
};

struct ConvertStablehloReturnOp : public OpRewritePattern<stablehlo::ReturnOp> {
  using OpRewritePattern<stablehlo::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::YieldOp>(op, op->getResultTypes(),
                                               op.getResults());
    return success();
  }
};

struct ConvertStablehloSliceOp : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto rank = op.getOperand().getType().getRank();
    if (rank < 1 || rank > 6) {
      return rewriter.notifyMatchFailure(
          op, "tosa.slice only supports 1D to 6D tensors");
    }

    auto strides = op.getStrides().getValues<int64_t>();
    for (auto stride : strides) {
      if (stride != 1) {
        return rewriter.notifyMatchFailure(
            op, "tosa.slice only supports strides of 1");
      }
    }

    auto startIndices = op.getStartIndices().getValues<int64_t>();
    auto endIndices = op.getLimitIndices().getValues<int64_t>();

    llvm::SmallVector<int64_t, 2> size;
    size.resize(startIndices.size());
    llvm::SmallVector<int64_t, 2> startIndicesI64;
    startIndicesI64.resize(startIndices.size());

    for (int64_t i = 0; i < static_cast<int64_t>(startIndices.size()); i++) {
      size[i] = endIndices[i] - startIndices[i];
      startIndicesI64[i] = startIndices[i];
    }

    rewriter.replaceOpWithNewOp<tosa::SliceOp>(
        op, op.getResult().getType(), op.getOperand(),
        rewriter.getDenseI64ArrayAttr(startIndicesI64),
        rewriter.getDenseI64ArrayAttr(size));
    return success();
  }
};

struct ConvertStablehloTransposeOp
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern<stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    auto rank = op.getOperand().getType().getRank();
    if (rank < 1 || rank > 6) {
      return rewriter.notifyMatchFailure(
          op, "tosa.transpose only supports 1D to 6D tensors");
    }

    auto perms = op.getPermutation();
    auto constOp = rewriter.create<tosa::ConstOp>(
        op->getLoc(),
        RankedTensorType::get({perms.size()}, rewriter.getI64Type()), perms);
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(op, op.getResult().getType(),
                                                   op.getOperand(), constOp);
    return success();
  }
};

struct ConvertStablehloWhileOp : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern<stablehlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp op,
                                PatternRewriter& rewriter) const override {
    auto* cond = &op.getCond();
    auto* body = &op.getBody();
    auto newWhileOp = rewriter.create<tosa::WhileOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands());

    auto* newCond = &newWhileOp->getRegion(0);
    auto* newBody = &newWhileOp->getRegion(1);
    rewriter.createBlock(newCond);
    rewriter.createBlock(newBody);

    rewriter.cloneRegionBefore(*cond, &newCond->back());
    rewriter.eraseBlock(&newCond->back());
    rewriter.cloneRegionBefore(*body, &newBody->back());
    rewriter.eraseBlock(&newBody->back());

    rewriter.replaceOp(op, newWhileOp.getResults());
    return success();
  }
};

namespace {

// Utilities to generate tosa.rescale operations.

/* StableHLO Multiplier shift quantization type
**
** double convertMultiplierAndShiftToScale(int32_t m, int32_t s) {
**   double multiplier = static_cast<double>(m);
**   double shift = static_cast<double>(s);
**   double result = multiplier * pow(2, -shift);
**   return result;
** }
*/

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);
  auto rescale_op = rewriter.create<tosa::RescaleOp>(
      op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(input_zp),
      rewriter.getI32IntegerAttr(output_zp),
      rewriter.getDenseI32ArrayAttr({multiplier}),
      rewriter.getDenseI32ArrayAttr({shift}), rewriter.getBoolAttr(scale32),
      rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(/*per_channel*/ false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value inputVal, double inputScale, int64_t inputZp) {
  // Output is always int32 type
  auto inputType = dyn_cast<mlir::ShapedType>(inputVal.getType());
  assert(inputType);
  auto outputType = inputType.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, outputType, inputVal, inputScale, inputZp,
                      0, /*double_round*/ false, /*scale_32*/ true);
}

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter& rewriter, Operation* op,
                            ShapedType output_type, Value input_val,
                            double output_scale, int64_t output_zp) {
  // Input should be int32 type
  auto input_type = dyn_cast<mlir::ShapedType>(input_val.getType());
  (void)input_type;
  assert(input_type && input_type.getElementType().isInteger(32) &&
         "expected rescale input element type to be i32");

  return buildRescale(rewriter, op, output_type, input_val, output_scale, 0,
                      output_zp, /*double_round*/ false, true);
}

}  // namespace

struct ConvertStablehloAddOp : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern<stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter& rewriter) const override {
    ShapedType input_lhs_type = op.getLhs().getType().dyn_cast<ShapedType>();
    ShapedType input_rhs_type = op.getRhs().getType().dyn_cast<ShapedType>();
    ShapedType output_type = op.getResult().getType().dyn_cast<ShapedType>();
    // Not a ranked tensor output
    if (!input_lhs_type || !input_rhs_type || !output_type) {
      return rewriter.notifyMatchFailure(
          op, "input/output tensor should be all of shaped type");
    }

    LLVM_DEBUG(llvm::dbgs() << "In ConvertStablehloAddOp\n");

    /* StableHLO Multiplier shift quantization type
    **
    ** auto input_lhs_qtype =
    **     input_lhs_type.getElementType()
    ** .dyn_cast<stablehlo::UniformQuantizedWithMultiplierAndShiftType>();
    ** auto input_rhs_qtype =
    **     input_rhs_type.getElementType()
    ** .dyn_cast<stablehlo::UniformQuantizedWithMultiplierAndShiftType>();
    ** auto output_qtype =
    **     output_type.getElementType()
    ** .dyn_cast<stablehlo::UniformQuantizedWithMultiplierAndShiftType>();
    */
    auto input_lhs_qtype =
        input_lhs_type.getElementType().dyn_cast<quant::UniformQuantizedType>();
    auto input_rhs_qtype =
        input_rhs_type.getElementType().dyn_cast<quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().dyn_cast<quant::UniformQuantizedType>();

    if (input_lhs_qtype && input_rhs_qtype && output_qtype) {
      LLVM_DEBUG(llvm::dbgs() << "Handling quantized types\n");

      /* StableHLO Multiplier shift quantization type
      **
      ** int32_t in_lhs_multiplier = input_lhs_qtype.getMultiplier();
      ** int32_t in_lhs_shift = input_lhs_qtype.getShift();
      ** int32_t in_rhs_multiplier = input_rhs_qtype.getMultiplier();
      ** int32_t in_rhs_shift = input_rhs_qtype.getShift();
      ** int32_t output_multiplier = output_qtype.getMultiplier();
      ** int32_t output_shift = output_qtype.getShift();

      ** double in_lhs_scale =
      **     convertMultiplierAndShiftToScale(in_lhs_multiplier, in_lhs_shift);
      ** double in_rhs_scale =
      **     convertMultiplierAndShiftToScale(in_rhs_multiplier, in_rhs_shift);
      ** double output_scale =
      **     convertMultiplierAndShiftToScale(output_multiplier, output_shift);
      */

      double in_lhs_scale = input_lhs_qtype.getScale();
      double in_rhs_scale = input_rhs_qtype.getScale();
      double output_scale = output_qtype.getScale();

      double max_scale_2x = 2.0 * std::max(in_lhs_scale, in_rhs_scale);

      const int32_t SHIFT_8_BIT = 20;
      const int32_t SHIFT_16_BIT = 15;

      int32_t input_shift =
          (output_qtype.getStorageType().getIntOrFloatBitWidth() == 16)
              ? SHIFT_16_BIT
              : SHIFT_8_BIT;

      double lhs_rescale_scale = in_lhs_scale / max_scale_2x;
      double rhs_rescale_scale = in_rhs_scale / max_scale_2x;
      double output_rescale_scale =
          max_scale_2x / (output_scale * static_cast<double>(1 << input_shift));

      Value op1_rescale_lhs = buildRescaleToInt32(
          rewriter, op, op.getLhs(),
          lhs_rescale_scale * static_cast<double>(1 << input_shift),
          input_lhs_qtype.getZeroPoint());
      Value op2_rescale_rhs = buildRescaleToInt32(
          rewriter, op, op.getRhs(),
          rhs_rescale_scale * static_cast<double>(1 << input_shift),
          input_rhs_qtype.getZeroPoint());

      ShapedType rescale_type_output = output_type.clone(rewriter.getI32Type());
      auto op3_add_op1_op2 = rewriter.create<tosa::AddOp>(
          op->getLoc(), rescale_type_output, op1_rescale_lhs, op2_rescale_rhs);
      Value op4_rescale_op3 = buildRescaleFromInt32(
          rewriter, op, output_type, op3_add_op1_op2.getResult(),
          output_rescale_scale, output_qtype.getZeroPoint());

      rewriter.replaceOp(op, {op4_rescale_op3});
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Handling non-quantized types\n");
      auto newAddOp = rewriter.create<tosa::AddOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands());

      rewriter.replaceOp(op, newAddOp.getResult());
    }
    return success();
  }
};

LogicalResult StablehloLegalizeToTosaPass::initialize(MLIRContext* ctx) {
  RewritePatternSet patternList(ctx);
  populateGeneratedPDLLPatterns(patternList);
  patternList.addWithLabel<ConvertStablehloCompareOp>({"StablehloCompare"},
                                                      ctx);
  patternList.addWithLabel<ConvertStablehloConcatenateOp>(
      {"StablehloConcatenate"}, ctx);
  patternList.addWithLabel<ConvertStablehloDotOp>({"StablehloDot"}, ctx);
  patternList.addWithLabel<ConvertStablehloGatherOp>({"StablehloGather"}, ctx);
  patternList.addWithLabel<ConvertStablehloIotaOp>({"StablehloIota"}, ctx);
  patternList.addWithLabel<ConvertStablehloReduceOp>({"StablehloReduce"}, ctx);
  patternList.addWithLabel<ConvertStablehloReturnOp>({"StablehloReturn"}, ctx);
  patternList.addWithLabel<ConvertStablehloSliceOp>({"StablehloSlice"}, ctx);
  patternList.addWithLabel<ConvertStablehloTransposeOp>({"StablehloTranspose"},
                                                        ctx);
  patternList.addWithLabel<ConvertStablehloWhileOp>({"StablehloWhile"}, ctx);
  patternList.addWithLabel<ConvertStablehloAddOp>({"StablehloAdd"}, ctx);
  patterns = std::move(patternList);
  return success();
}

void StablehloLegalizeToTosaPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

}  // namespace

}  // namespace tosa
}  // namespace mlir
