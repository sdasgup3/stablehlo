/* Copyright 2022 The StableHLO Authors.

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

#include "Interpreter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace stablehlo {

llvm::Expected<SmallVector<Tensor>> eval(func::FuncOp func,
                                         ArrayRef<Tensor> args) {
  if (func->getNumRegions() != 1) {
    return invalidArgument("Expected one region in func %s",
                           func.getName().str().c_str());
  }
  if (!func.getBody().hasOneBlock()) {
    return invalidArgument("Expected one block in func %s",
                           func.getName().str().c_str());
  }

  Block &block = func.front();
  if (block.getNumArguments() != args.size()) {
    return invalidArgument(
        "Expected same amount of func arguments in %s "
        "and runtime arguments (%d)",
        func.getName().str().c_str(), args.size());
  }
  llvm::DenseMap<Value, Tensor> stackFrame;
  for (auto [ssaArg, runtimeArg] : llvm::zip(block.getArguments(), args)) {
    stackFrame[ssaArg] = runtimeArg;
  }

  for (Operation &op : block) {
    auto fetchOperand = [&](Value value) -> Tensor {
      auto it = stackFrame.find(value);
      if (it != stackFrame.end()) return it->second;
      llvm::report_fatal_error("Expected a terminator when evaluating func");
    };
    auto populateResults = [&](ArrayRef<Tensor> runtimeValues) {
      assert(op.getNumResults() == runtimeValues.size());
      for (auto [ssaResult, runtimeResult] :
           llvm::zip(op.getResults(), runtimeValues)) {
        stackFrame[ssaResult] = runtimeResult;
      }
    };

    if (auto addOp = dyn_cast<AddOp>(op)) {
      Tensor runtimeLhs = fetchOperand(addOp.getLhs());
      Tensor runtimeRhs = fetchOperand(addOp.getRhs());
      Tensor runtimeResult = eval_add(addOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      Tensor runtimeLhs = fetchOperand(andOp.getLhs());
      Tensor runtimeRhs = fetchOperand(andOp.getRhs());
      Tensor runtimeResult = eval_and(andOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      Tensor runtimeOperand = fetchOperand(ceilOp.getOperand());
      Tensor runtimeResult = eval_ceil(ceilOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      Tensor runtimeResult = eval_constant(constantOp.getValue());
      populateResults({runtimeResult});
    } else if (auto cosineOp = dyn_cast<CosineOp>(op)) {
      Tensor runtimeOperand = fetchOperand(cosineOp.getOperand());
      Tensor runtimeResult = eval_cosine(cosineOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      Tensor runtimeOperand = fetchOperand(floorOp.getOperand());
      Tensor runtimeResult = eval_floor(floorOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      Tensor runtimeResult =
          eval_iota(iotaOp.getType(), iotaOp.getIotaDimension());
      populateResults({runtimeResult});
    } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
      Tensor runtimeLhs = fetchOperand(maxOp.getLhs());
      Tensor runtimeRhs = fetchOperand(maxOp.getRhs());
      Tensor runtimeResult = eval_max(maxOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      Tensor runtimeLhs = fetchOperand(minOp.getLhs());
      Tensor runtimeRhs = fetchOperand(minOp.getRhs());
      Tensor runtimeResult = eval_min(minOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto multiplyOp = dyn_cast<MulOp>(op)) {
      Tensor runtimeLhs = fetchOperand(multiplyOp.getLhs());
      Tensor runtimeRhs = fetchOperand(multiplyOp.getRhs());
      Tensor runtimeResult =
          eval_multiply(multiplyOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto negOp = dyn_cast<NegOp>(op)) {
      Tensor runtimeOperand = fetchOperand(negOp.getOperand());
      Tensor runtimeResult = eval_neg(negOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      Tensor runtimeOperand = fetchOperand(notOp.getOperand());
      Tensor runtimeResult = eval_not(notOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto orOp = dyn_cast<OrOp>(op)) {
      Tensor runtimeLhs = fetchOperand(orOp.getLhs());
      Tensor runtimeRhs = fetchOperand(orOp.getRhs());
      Tensor runtimeResult = eval_or(orOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      Tensor runtimeOperand = fetchOperand(reshapeOp.getOperand());
      Tensor runtimeResult = eval_reshape(reshapeOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      SmallVector<Tensor> runtimeOperands;
      for (Value ssaOperand : returnOp.getOperands()) {
        runtimeOperands.push_back(fetchOperand(ssaOperand));
      }
      return runtimeOperands;
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      Tensor runtimeOperand = fetchOperand(sineOp.getOperand());
      Tensor runtimeResult = eval_sine(sineOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto subtractOp = dyn_cast<SubtractOp>(op)) {
      Tensor runtimeLhs = fetchOperand(subtractOp.getLhs());
      Tensor runtimeRhs = fetchOperand(subtractOp.getRhs());
      Tensor runtimeResult =
          eval_subtract(subtractOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      Tensor runtimeOperand = fetchOperand(tanhOp.getOperand());
      Tensor runtimeResult = eval_tanh(tanhOp.getType(), runtimeOperand);
      populateResults({runtimeResult});
    } else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
      Tensor runtimeOperand = fetchOperand(transposeOp.getOperand());
      Tensor runtimeResult = eval_transpose(
          transposeOp.getType(), runtimeOperand, transposeOp.getPermutation());
      populateResults({runtimeResult});
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      Tensor runtimeLhs = fetchOperand(xorOp.getLhs());
      Tensor runtimeRhs = fetchOperand(xorOp.getRhs());
      Tensor runtimeResult = eval_xor(xorOp.getType(), runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else {
      return invalidArgument("Unsupported op: %s", debugString(op).c_str());
    }
  }

  return invalidArgument("Expected a terminator when evaluating func");
}

}  // namespace stablehlo
}  // namespace mlir
