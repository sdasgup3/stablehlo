/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_CHECKOPS_H_
#define STABLEHLO_DIALECT_CHECKOPS_H_

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace stablehlo {
namespace check {

class CheckDialect : public Dialect {
 public:
  explicit CheckDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "check"; }
};

}  // namespace check
}  // namespace stablehlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/dialect/CheckOps.h.inc"

#endif  // STABLEHLO_DIALECT_CHECKOPS_H_
