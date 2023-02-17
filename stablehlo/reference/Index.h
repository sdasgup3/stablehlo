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

#ifndef STABLEHLO_REFERENCE_INDEX_H_
#define STABLEHLO_REFERENCE_INDEX_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace stablehlo {

/// Represents an index of a tensor.
class Index {
 public:
  /// Creates an `Index` of size `size`.
  explicit Index(size_t size) : index_(size) {}

  /// Create an `Index` whose value at each dimension d is initialized with
  /// `array`[d].
  explicit Index(llvm::ArrayRef<int64_t> array) : index_(array) {}

  Index(const Index &other) = default;
  Index &operator=(const Index &other) = default;

  /// Overloaded indexing operator returns `(*this)[idx]`.
  int64_t &operator[](int64_t idx) { return index_[idx]; }
  int64_t operator[](int64_t idx) const { return index_[idx]; }

  /// Overloaded equality operator.
  bool operator==(const Index &other) const { return index_ == other.index_; }

  /// Overloaded add operator to perform `(*this)[d] + other[d]` for all
  /// dimension d.
  Index operator+(const Index &other) const;

  /// Overloaded add operator to perform `(*this)[d] + array[d]` for all
  /// dimension d.
  Index operator+(ArrayRef<int64_t> array) const;

  /// Overloaded product operator to perform `(*this)[d] * other[d]` for all
  /// dimension d.
  Index operator*(const Index &other) const;

  /// Overloaded product operator to perform `(*this)[d] * array[d]` for all
  /// dimension d.
  Index operator*(ArrayRef<int64_t> array) const;

  /// Get the index array.
  ArrayRef<int64_t> getIndexArray() const { return index_; }

  // Create a new `Index` i with the effect of applying `permutation`  to
  // `this` object, such that `i[i] = (*this)[perm[i]]`.
  Index permute(ArrayRef<int64_t> permutation);

  /// Returns the number of dimensions.
  size_t size() const { return index_.size(); }

 private:
  /// Underlying storage.
  llvm::SmallVector<int64_t> index_;
};

/// Overloaded add operator to perform `array[d] + index[d]` for all
/// dimension d.
Index operator+(ArrayRef<int64_t> array, const Index &index);

/// Overloaded product operator to perform `array[d] * index[d]` for all
/// dimension d.
Index operator*(ArrayRef<int64_t> array, const Index &index);

/// Check if the 'index' is a valid index in the index space of a tensor with
/// shape 'shape'. Specifically, for a shape '(d0)x(d1)x...x(dR-1)' and an index
/// '{i0, i1, ..., iR-1}', we check if 0 <= i[k] <= d[k] for k in
/// {0, 1, ..., R-1}. Note that the check also implies that 'd[k]' >= 1.
LogicalResult verifyIndex(ArrayRef<int64_t> shape, const Index &index);

/// Iterates over the index space of a tensor with a given shape, producing
/// indices in lexicographical order. As an example, for a tensor with shape
/// [2,3], the iterator enumerates the indices (0,0), (0,1), (0,2), (1,0),
/// (1,1), (1,2) and <END> (special past-the-end element which cannot be
/// dereferenced).
class IndexSpaceIterator {
 public:
  /// \name Constructor
  IndexSpaceIterator(llvm::ArrayRef<int64_t> shape,
                     std::optional<llvm::SmallVector<int64_t>> index)
      : shape_(shape), index_(index) {
    if (index && failed(verifyIndex(shape, (*index_))))
      llvm::report_fatal_error(
          "Incompatible index and shape found while creating "
          "an IndexSpaceIterator");
  }

  /// Get the current index.
  /// At any point in time, the iterator can either reference an actual index
  /// or the past-the-end element in the index space.
  /// Dereferencing a past-the-end iterator will result in a fatal error.
  Index operator*() const;

  /// Compare the iterator to another iterator.
  /// Two iterators are equal if they have the same underlying shape and
  /// reference the same element in the index space.
  bool operator==(const IndexSpaceIterator &it) {
    return shape_ == it.shape_ && index_ == it.index_;
  }
  bool operator!=(const IndexSpaceIterator &it) { return !(*this == it); }

  /// Increment to the next index while iterating over the index space
  /// of a tensor in lexicographical order.
  /// Incrementing past the last index will result in a past-the-end iterator
  /// which cannot be dereferenced. Incrementing even further will result in
  /// a fatal error.
  IndexSpaceIterator &operator++();
  IndexSpaceIterator operator++(int);

 private:
  /// Shape of the tensor whose index space to be iterated on.
  llvm::SmallVector<int64_t> shape_;

  /// Current multi-dimensional index.
  /// If the optional is empty, then we're at the end
  std::optional<Index> index_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INDEX_H_
