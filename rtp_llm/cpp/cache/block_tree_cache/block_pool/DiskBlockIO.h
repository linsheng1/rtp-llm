#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {

enum class DiskBlockIOStatus {
    OK,
    INVALID_SIZE,
    ALIGNMENT_ERROR,
    IO_ERROR,
    PARTIAL_FAILURE,
};

struct DiskRead {
    uint64_t offset;
    void*    buffer;
    size_t   bytes;
};

struct DiskWrite {
    uint64_t    offset;
    const void* buffer;
    size_t      bytes;
};

// Optional per-write-batch observation, owned by the caller through the final callback.
// Write workers merge private samples before the final callback publishes this batch.
// Batch wall time runs from the earliest worker start through the last worker exit;
// syscall sums/counts cover actual calls, not a sum of overlapping batch wall times.
struct DiskWriteTiming {
    int64_t queued_us{0}, worker_start_us{0}, sort_end_us{0}, io_end_us{0};
    int64_t pwrite_count{0}, pwrite_sum_us{0}, pwrite_max_us{0};

    static int64_t nowUs() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Block-pool-agnostic I/O for one backing file. DiskBlockPool owns
// allocation policy and maps block indices to offsets.
class DiskBlockIO {
public:
    virtual ~DiskBlockIO() = default;

    // Opens (creating if necessary) the file at file_path and preallocates it to
    // exactly `bytes`. When buffered_io is false, the file is opened for direct I/O
    // and all subsequent read/write offsets, buffer addresses, and sizes must be
    // 4096-byte aligned.
    virtual DiskBlockIOStatus openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) = 0;

    virtual DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes)        = 0;
    virtual DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) = 0;

    // Batch helpers. The first version simply loops the single-item call and returns
    // immediately on the first non-OK status, leaving any remaining requests unissued.
    virtual DiskBlockIOStatus read(const std::vector<DiskRead>& reads)    = 0;
    virtual DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) = 0;

    // Backends without syscall instrumentation retain their original write implementation.
    virtual DiskBlockIOStatus writeObserved(const std::vector<DiskWrite>& writes, DiskWriteTiming*) {
        return write(writes);
    }
    virtual DiskBlockIOStatus writeBlockObserved(uint64_t offset, const void* src, size_t bytes, DiskWriteTiming*) {
        return write(offset, src, bytes);
    }

    // The owner finishes all synchronous and executor-driven I/O before closing.
    virtual void close() = 0;

    virtual std::string debugString() const = 0;
};

// BlockTreePosixDiskBlockIO implements DiskBlockIO on top of open/posix_fallocate/pread/pwrite.
class BlockTreePosixDiskBlockIO: public DiskBlockIO {
public:
    BlockTreePosixDiskBlockIO() = default;
    ~BlockTreePosixDiskBlockIO() override;

    DiskBlockIOStatus openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) override;

    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override;
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override;

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override;
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override;

    DiskBlockIOStatus writeObserved(const std::vector<DiskWrite>& writes, DiskWriteTiming* timing) override;
    DiskBlockIOStatus writeBlockObserved(uint64_t offset, const void* src, size_t bytes, DiskWriteTiming* timing) override;

    void close() override;

    std::string debugString() const override;

private:
    // Validates offset/buffer/bytes against direct-I/O alignment and against the
    // preallocated file size. This is purely parameter- and state-based (it never
    // touches the fd), so it behaves identically whether or not the underlying
    // open() call actually succeeded in getting O_DIRECT semantics from the
    // filesystem.
    DiskBlockIOStatus validate(uint64_t offset, const void* buffer, size_t bytes) const;
    DiskBlockIOStatus writeImpl(uint64_t offset, const void* src, size_t bytes, DiskWriteTiming* timing);
    int               fd_{-1};
    std::string       file_path_;
    size_t            bytes_{0};
    bool              buffered_io_{true};
};

}  // namespace rtp_llm
