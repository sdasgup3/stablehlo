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

// TODO: Come up with a better filename.
// TODO: Add to the Bazel build.
#ifndef STABLEHLO_REFERENCE_PROTOTYPE_H
#define STABLEHLO_REFERENCE_PROTOTYPE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace stablehlo {

class Axes : public SmallVector<int64_t> {
 public:
  Axes(std::initializer_list<int64_t> list) : SmallVector(list) {}
  explicit Axes(ArrayRef<int64_t> array) : SmallVector(array) {}
  explicit Axes(DenseIntElementsAttr attr)
      : SmallVector(attr.getValues<int64_t>()) {}
};

class Sizes : public SmallVector<int64_t> {
 public:
  Sizes(std::initializer_list<int64_t> list) : SmallVector(list) {}
  explicit Sizes(size_t size, int64_t element = 0)
      : SmallVector(size, element) {}
  explicit Sizes(ArrayRef<int64_t> array) : SmallVector(array) {}
  explicit Sizes(DenseIntElementsAttr attr)
      : SmallVector(attr.getValues<int64_t>()) {}
  Sizes permute(ArrayRef<int64_t> permutation) const;
  bool inBounds(const Sizes &bounds) const;
};

raw_ostream &operator<<(raw_ostream &os, const Axes &x);
raw_ostream &operator<<(raw_ostream &os, const Sizes &x);

Sizes operator+(const Sizes &x, const Sizes &y);
Sizes operator+(const Sizes &x, int64_t y);
Sizes operator+(int64_t x, const Sizes &y);

Sizes operator-(const Sizes &x, const Sizes &y);
Sizes operator-(const Sizes &x, int64_t y);
Sizes operator-(int64_t x, const Sizes &y);

Sizes operator*(const Sizes &x, const Sizes &y);
Sizes operator*(const Sizes &x, int64_t y);
Sizes operator*(int64_t x, const Sizes &y);

Sizes clamp(ArrayRef<int64_t> min, const Sizes &x, ArrayRef<int64_t> max);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_PROTOTYPE_H
