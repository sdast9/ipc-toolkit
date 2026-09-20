#pragma once

#include <cstddef>
#include <limits>

namespace ipc {

/// @brief An unsigned count that knows whether it is exact.
///
/// The broad-phase resource accounting (BroadPhaseBudget) decides whether a
/// build may allocate from counts computed before the allocation: products
/// of cell extents, pair counts n(n-1)/2, sums over boxes, over the threads
/// of a parallel reduction and over the detect calls of a build. Any of
/// these can exceed size_t on an extreme input, and a wrapped count admits
/// an arbitrarily large build against any budget. Every operation here
/// saturates at SIZE_MAX and sets `overflowed` (sticky under addition, so a
/// reduction join preserves it), which keeps two properties:
///
/// - a decision `exceeds(limit)` is correct for every representable limit,
///   including SIZE_MAX: a count that overflowed is greater than the limit;
/// - a report can say whether `value` is the exact count or a lower bound.
///
/// Saturation is associative and commutative, so a parallel reduction gives
/// the same value and flag whatever the partition.
struct CheckedCount {
    size_t value = 0;
    bool overflowed = false;

    constexpr CheckedCount() = default;
    /// @brief An exact count.
    explicit constexpr CheckedCount(const size_t exact_value)
        : value(exact_value)
    {
    }

    static constexpr size_t max() { return std::numeric_limits<size_t>::max(); }

    /// @brief Is `value` the exact count (no operation saturated)?
    constexpr bool exact() const { return !overflowed; }

    /// @brief Is the true count greater than `limit`? Correct also when the
    ///        count overflowed: it is then greater than any size_t limit.
    constexpr bool exceeds(const size_t limit) const
    {
        return overflowed || value > limit;
    }

    /// @brief a * b with overflow detection.
    static constexpr CheckedCount product(const size_t a, const size_t b)
    {
        CheckedCount r;
        if (a != 0 && b > max() / a) {
            r.value = max();
            r.overflowed = true;
        } else {
            r.value = a * b;
        }
        return r;
    }

    /// @brief a * b * c with overflow detection.
    static constexpr CheckedCount
    product(const size_t a, const size_t b, const size_t c)
    {
        return product(a, b).times(c);
    }

    /// @brief The unordered pairs of n items, n(n-1)/2, dividing the even
    ///        factor first so that every result that fits size_t is exact
    ///        (n(n-1) itself overflows from n = 2^32 + 1 on).
    static constexpr CheckedCount unordered_pairs(const size_t n)
    {
        if (n < 2) {
            return CheckedCount(0);
        }
        return (n % 2 == 0) ? product(n / 2, n - 1) : product(n, (n - 1) / 2);
    }

    /// @brief this * m with overflow detection (an overflowed count stays
    ///        overflowed unless m is zero: zero of anything is zero).
    constexpr CheckedCount times(const size_t m) const
    {
        if (m == 0) {
            return CheckedCount(0);
        }
        CheckedCount r = product(value, m);
        r.overflowed = r.overflowed || overflowed;
        if (r.overflowed) {
            r.value = max();
        }
        return r;
    }

    constexpr CheckedCount& operator+=(const CheckedCount& other)
    {
        if (value > max() - other.value) {
            value = max();
            overflowed = true;
        } else {
            value += other.value;
        }
        overflowed = overflowed || other.overflowed;
        if (overflowed) {
            value = max();
        }
        return *this;
    }

    constexpr CheckedCount& operator+=(const size_t exact_value)
    {
        return *this += CheckedCount(exact_value);
    }

    friend constexpr CheckedCount
    operator+(CheckedCount a, const CheckedCount& b)
    {
        return a += b;
    }

    constexpr bool operator==(const CheckedCount& other) const
    {
        return value == other.value && overflowed == other.overflowed;
    }
    constexpr bool operator!=(const CheckedCount& other) const
    {
        return !(*this == other);
    }
};

} // namespace ipc
