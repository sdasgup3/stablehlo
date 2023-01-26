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

#include "stablehlo/dialect/CheckOps.h"

#define GET_OP_CLASSES
#include "stablehlo/dialect/CheckOps.cpp.inc"

namespace mlir {
namespace stablehlo {

//===----------------------------------------------------------------------===//
// Check Dialect Constructor
//===----------------------------------------------------------------------===//

CheckDialect::CheckDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CheckDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/dialect/CheckOps.cpp.inc"
      >();
}

}  // namespace stablehlo
}  // namespace mlir
