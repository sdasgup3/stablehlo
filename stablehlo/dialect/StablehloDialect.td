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

#ifndef STABLEHLO_DIALECT_STABLEHLO_DIALECT
#define STABLEHLO_DIALECT_STABLEHLO_DIALECT

def StableHLO_Dialect : Dialect {
  let name = "stablehlo";
  let cppNamespace = "::mlir::stablehlo";

  let description = [{
    StableHLO is an operation set that expresses ML computations. It has been
    originally bootstrapped from the MHLO dialect and enhances it with additional
    functionality, including serialization and versioning, to be used as
    a portability layer between ML frameworks and ML compilers.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

#endif // STABLEHLO_DIALECT_STABLEHLO_DIALECT
