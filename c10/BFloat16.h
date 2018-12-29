#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>
#include <cmath>
#include <ostream>

/// A minimal BFloat16 type:
/// 1. Store value as a 16-bit unsigned int
/// 2. Support only conversions to and from float
///    2.1 Apply truncation (instead of rounding) for converting from float
/// 3. Define no arithmetic operations

// Refer to TensorFlow's bfloat16 CPU implementation:
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/lib/bfloat16

namespace c10 {

struct BFloat16;

namespace detail {
    static inline uint16_t floatToUint16(float f) {
        if (std::isnan(f)) {
            return 0x7FC0; // This is a NaN value in BFloat16
        }
        auto bits = *reinterpret_cast<uint32_t*>(&f);
        return bits >> 16;
    }
    static inline float uint16ToFloat(uint16_t u) {
        uint32_t bits = u << 16;
        return *reinterpret_cast<float*>(&bits);
    }
}

struct BFloat16 {

    BFloat16() = default;

    inline C10_HOST_DEVICE BFloat16(float x) : value(detail::floatToUint16(x)) { }
    inline C10_HOST_DEVICE operator float() const { return detail::uint16ToFloat(value); }

    friend std::ostream& operator<<(std::ostream& out, const BFloat16& f) {
        return out << (float) f;
    }

    uint16_t value;
};

} // namespace c10
