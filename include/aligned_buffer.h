#pragma once

#include <algorithm>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

template <typename T, std::size_t Alignment = 64>
class AlignedBuffer {
public:
    ~AlignedBuffer() { reset(); }

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;

    AlignedBuffer(AlignedBuffer &&other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(std::size_t n) {
        if (n == size_) {
            return;
        }
        reset();
        if (n == 0) {
            return;
        }
        data_ = static_cast<T *>(_mm_malloc(n * sizeof(T), Alignment));
        if (!data_) {
            std::cerr << "Aligned allocation failed" << std::endl;
            std::exit(3);
        }
        size_ = n;
    }

    void assign(std::size_t n, const T &value) {
        resize(n);
        std::fill(data_, data_ + size_, value);
    }

    void clear() { reset(); }

    T *data() { return data_; }
    const T *data() const { return data_; }
    std::size_t size() const { return size_; }

    T &operator[](std::size_t i) { return data_[i]; }
    const T &operator[](std::size_t i) const { return data_[i]; }

    T *begin() { return data_; }
    T *end() { return data_ + size_; }
    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }

private:
    void reset() {
        if (data_) {
            _mm_free(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    T *data_ = nullptr;
    std::size_t size_ = 0;
};
