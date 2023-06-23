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

#include "stablehlo/dialect/StablehloTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "stablehlo/dialect/AssemblyFormat.h"

#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/StablehloTypeDefs.cpp.inc"

namespace mlir {
namespace stablehlo {

LogicalResult printStablehloType(Type type, AsmPrinter &printer) {
  return generatedTypePrinter(type, printer);
}

OptionalParseResult parseStablehloType(mlir::AsmParser &parser,
                                       llvm::StringRef *mnemonic,
                                       mlir::Type &type) {
  return generatedTypeParser(parser, mnemonic, type);
}

}  // namespace stablehlo
}  // namespace mlir
