/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permutationissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/reference/Prototype.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace stablehlo {

// TODO: Deduplicate the implementations.
// Currently, there's a massive amount of copypasta going on.

// TODO: Add soundness checks, e.g. for permutation size and elements,
// for x and y having the same size, etc.

raw_ostream &operator<<(raw_ostream &os, const Axes &x) {
  os << "[";
  llvm::interleave(x, os, ", ");
  os << "]";
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const Index &x) {
  os << "[";
  llvm::interleave(x, os, ", ");
  os << "]";
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const Sizes &x) {
  os << "[";
  llvm::interleave(x, os, ", ");
  os << "]";
  return os;
}

Index Index::permute(ArrayRef<int64_t> permutation) const {
  Index result(size());
  for (size_t i = 0; i < permutation.size(); i++)
    result[i] = (*this)[permutation[i]];
  return result;
}

bool Index::inBounds(const Sizes &sizes) const {
  if (sizes.size() != size()) return false;

  for (auto [dimSize, dimIndex] : llvm::zip(sizes, *this))
    if (dimIndex < 0 || dimIndex >= dimSize) return false;

  return true;
}

Sizes Sizes::permute(ArrayRef<int64_t> permutation) const {
  Sizes result(size());
  for (size_t i = 0; i < permutation.size(); i++)
    result[i] = (*this)[permutation[i]];
  return result;
}

Index operator+(const Index &x, const Index &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y[i];
  }
  return result;
}

Index operator+(const Index &x, const Sizes &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y[i];
  }
  return result;
}

Index operator+(const Index &x, int64_t y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y;
  }
  return result;
}

Index operator+(const Sizes &x, const Index &y) { return y + x; }

Sizes operator+(const Sizes &x, const Sizes &y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y[i];
  }
  return result;
}

Sizes operator+(const Sizes &x, int64_t y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y;
  }
  return result;
}

Index operator+(int64_t x, const Index &y) { return y + x; }

Sizes operator+(int64_t x, const Sizes &y) { return y + x; }

Index operator-(const Index &x, const Index &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y[i];
  }
  return result;
}

Index operator-(const Index &x, const Sizes &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y[i];
  }
  return result;
}

Index operator-(const Index &x, int64_t y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y;
  }
  return result;
}

Index operator-(const Sizes &x, const Index &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y[i];
  }
  return result;
}

Sizes operator-(const Sizes &x, const Sizes &y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y[i];
  }
  return result;
}

Sizes operator-(const Sizes &x, int64_t y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y;
  }
  return result;
}

Index operator-(int64_t x, const Index &y) {
  Index result(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    result[i] = x - y[i];
  }
  return result;
}

Sizes operator-(int64_t x, const Sizes &y) {
  Sizes result(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    result[i] = x - y[i];
  }
  return result;
}

Index operator*(const Index &x, const Index &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

Index operator*(const Index &x, const Sizes &y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

Index operator*(const Index &x, int64_t y) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y;
  }
  return result;
}

Index operator*(const Sizes &x, const Index &y) { return y * x; }

Sizes operator*(const Sizes &x, const Sizes &y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

Sizes operator*(const Sizes &x, int64_t y) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y;
  }
  return result;
}

Index operator*(int64_t x, const Index &y) { return y * x; }

Sizes operator*(int64_t &x, const Sizes &y) { return y + x; }

Index clamp(ArrayRef<int64_t> min, const Index &x, ArrayRef<int64_t> max) {
  Index result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    auto minEl = min.size() == 1 ? min[0] : min[i];
    auto maxEl = max.size() == 1 ? max[0] : max[i];
    result[i] = std::min(std::max(x[i], minEl), maxEl);
  }
  return result;
}

Sizes clamp(ArrayRef<int64_t> min, const Sizes &x, ArrayRef<int64_t> max) {
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    auto minEl = min.size() == 1 ? min[0] : min[i];
    auto maxEl = max.size() == 1 ? max[0] : max[i];
    result[i] = std::min(std::max(x[i], minEl), maxEl);
  }
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
