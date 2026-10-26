#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CoalescedPaceWriteClient.h"
#if USING_CUDA
#include <cuda_runtime_api.h>
#endif
#include <cstring>
#include <iostream>
#include <future>
#include <thread>
#include <stdexcept>
using namespace kv_cache_manager;
using rtp_llm::kvcm::CoalescedPaceWriteClient;
static std::atomic<int> checks{0};
void                    check(bool v) {
    ++checks;
    if (!v)
        throw std::runtime_error("coalesced write assertion");
}
struct State {
    int                                saves = 0;
    std::atomic<int>                   loads{0}, read_arrivals{0};
    ClientErrorCode                    error          = ER_OK;
    bool                               parallel_reads = false;
    bool                               throw_save = false, bad_uris = false, check_lifetime = false;
    std::vector<std::string>           expected, observed;
    std::vector<const void*>           borrowed;
    const BlockBuffers*                read_forwarded = nullptr;
    std::shared_ptr<TransferTraceInfo> trace;
};
class Delegate final: public TransferClient {
public:
    explicit Delegate(std::shared_ptr<State> s): s_(std::move(s)) {}
    ~Delegate() override {
        if (s_->check_lifetime)
            for (size_t i = 0; i < s_->borrowed.size(); ++i)
                check(!std::memcmp(s_->borrowed[i], s_->expected[i].data(), s_->expected[i].size()));
    }
    ClientErrorCode
    LoadKvCaches(const UriStrVec&, const BlockBuffers& b, std::shared_ptr<TransferTraceInfo> t) override {
        ++s_->loads;
        check(&b == s_->read_forwarded);
        check(t == s_->trace);
        if (s_->parallel_reads) {
            ++s_->read_arrivals;
            auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (s_->read_arrivals.load() < 2 && std::chrono::steady_clock::now() < until)
                std::this_thread::yield();
            check(s_->read_arrivals.load() == 2);
        }
        return ER_OK;
    }
    std::pair<ClientErrorCode, UriStrVec>
    SaveKvCaches(const UriStrVec& u, const BlockBuffers& b, std::shared_ptr<TransferTraceInfo> t) override {
        ++s_->saves;
        check(u.size() == b.size());
        check(t == s_->trace);
        s_->borrowed.clear();
        s_->observed.clear();
        for (size_t i = 0; i < b.size(); ++i) {
            check(b[i].iovs.size() == 1);
            check(b[i].iovs[0].type == MemoryType::CPU);
            const auto& v = b[i].iovs[0];
            s_->observed.emplace_back(static_cast<const char*>(v.base), v.size);
            s_->borrowed.push_back(v.base);
        }
        check(s_->observed == s_->expected);
        if (s_->throw_save)
            throw std::runtime_error("uncertain submission");
        UriStrVec actual;
        for (const auto& x : u)
            actual.push_back(x + "_saved");
        if (s_->bad_uris)
            actual.pop_back();
        return {s_->error, actual};
    }

protected:
    ClientErrorCode Init(const std::string&, const InitParams&) override {
        return ER_OK;
    }

private:
    std::shared_ptr<State> s_;
};
auto copy = [](void* d, MemoryType dt, const void* s, MemoryType st, size_t n) {
    if (dt != MemoryType::CPU)
        return false;
    if (st == MemoryType::CPU) {
        std::memcpy(d, s, n);
        return true;
    }
#if USING_CUDA
    return st == MemoryType::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    return false;
#endif
};
int main(int argc, char**) {
    const bool gpu = argc > 1;
#if USING_CUDA
    void* device = nullptr;
#endif
    try {
        auto s              = std::make_shared<State>();
        s->expected         = {"abcdef", "12345"};
        s->trace            = std::make_shared<TransferTraceInfo>();
        s->trace->block_ids = {"A", "B"};
        std::string  a = "ab", b = "cdef", c = "1", d = "2345";
        BlockBuffers source = {{{{MemoryType::CPU, a.data(), a.size()}, {MemoryType::CPU, b.data(), b.size()}}},
                               {{{MemoryType::CPU, c.data(), c.size()}, {MemoryType::CPU, d.data(), d.size()}}}};
        CoalescedPaceWriteClient client(std::make_unique<Delegate>(s), copy);
        auto                     result = client.SaveKvCaches({"one", "two"}, source, s->trace);
        check(result.first == ER_OK && result.second == UriStrVec({"one_saved", "two_saved"}));
        check(s->saves == 1);
        char         dst[2];
        BlockBuffers read = {{{{MemoryType::CPU, dst, 1}, {MemoryType::CPU, nullptr, 1, true}}}};
        s->read_forwarded = &read;
        check(client.LoadKvCaches({"r"}, read, s->trace) == ER_OK);
        check(s->loads == 1);
        auto parallel            = std::make_shared<State>();
        parallel->parallel_reads = true;
        parallel->read_forwarded = &read;
        CoalescedPaceWriteClient reader(std::make_unique<Delegate>(parallel), copy);
        auto                     x = std::async(std::launch::async, [&] { return reader.LoadKvCaches({"r"}, read); });
        auto                     y = std::async(std::launch::async, [&] { return reader.LoadKvCaches({"r"}, read); });
        check(x.get() == ER_OK && y.get() == ER_OK && parallel->loads == 2);
        auto calls = s->saves;
        check(client.SaveKvCaches({"bad"}, source, s->trace).first == ER_INVALID_PARAMS);
        check(s->saves == calls);
        auto invalid = s->trace;
        invalid->block_ids.pop_back();
        check(client.SaveKvCaches({"one", "two"}, source, invalid).first == ER_OK);
        invalid->block_ids.push_back("B");
        calls             = s->saves;
        BlockBuffers huge = {{{{MemoryType::CPU, a.data(), CoalescedPaceWriteClient::max_operation_bytes + 1}}}};
        check(client.SaveKvCaches({"huge"}, huge).first == ER_INVALID_LOCAL_BUFFERS);
        check(s->saves == calls);
        for (int failure = 0; failure < 3; ++failure) {
            auto f            = std::make_shared<State>();
            f->expected       = s->expected;
            f->trace          = s->trace;
            f->check_lifetime = true;
            f->error          = failure == 0 ? ER_SDK_TIMEOUT : ER_OK;
            f->throw_save     = failure == 1;
            f->bad_uris       = failure == 2;
            CoalescedPaceWriteClient uncertain(std::make_unique<Delegate>(f), copy);
            try {
                auto x = uncertain.SaveKvCaches({"one", "two"}, source, f->trace);
                check(failure != 1);
                check(x.first != ER_OK);
            } catch (const std::runtime_error& e) {
                check(failure == 1 && std::string(e.what()) == "uncertain submission");
            }
            check(uncertain.SaveKvCaches({"one", "two"}, source, f->trace).first == ER_INVALID_LOCAL_BUFFERS);
            check(uncertain.LoadKvCaches({"one"}, read) != ER_OK);
            check(f->saves == 1 && f->loads == 0);
        }
        // Already-packed CPU buffers bypass an additional copy and preserve URI results.
        auto f      = std::make_shared<State>();
        f->expected = {"abcdef"};
        CoalescedPaceWriteClient packed(std::make_unique<Delegate>(f),
                                        [](void*, MemoryType, const void*, MemoryType, size_t) {
                                            throw std::runtime_error("unexpected copy");
                                            return false;
                                        });
        std::string              full = "abcdef";
        BlockBuffers             one  = {{{{MemoryType::CPU, full.data(), full.size()}}}};
        check(packed.SaveKvCaches({"one"}, one).first == ER_OK);
        if (gpu) {
#if USING_CUDA
            std::string bytes(262144, '\0');
            for (size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = char(i * 31 + (i >> 8));
            check(cudaSetDevice(0) == cudaSuccess);
            check(cudaMalloc(&device, bytes.size()) == cudaSuccess);
            check(cudaMemcpy(device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess);
            auto g      = std::make_shared<State>();
            g->expected = {bytes};
            CoalescedPaceWriteClient from_gpu(std::make_unique<Delegate>(g), copy);
            BlockBuffers             many(1);
            for (size_t off = 0; off < bytes.size(); off += 4096)
                many[0].iovs.push_back({MemoryType::GPU, static_cast<char*>(device) + off, 4096, false});
            check(from_gpu.SaveKvCaches({"gpu"}, many).first == ER_OK);
            check(g->saves == 1);
            for (int layout = 0; layout < 3; ++layout) {
                const std::vector<size_t> order =
                    layout == 1 ? std::vector<size_t>{0, 2, 1, 3} : std::vector<size_t>{0, 1, 2, 3};
                auto         rows_state = std::make_shared<State>();
                BlockBuffers rows_source(4);
                UriStrVec    uris;
                for (size_t block = 0; block < 4; ++block) {
                    std::string expected;
                    for (size_t col = 0; col < 4; ++col) {
                        size_t off = order[block] * 16384 + col * 4096,
                               n   = (layout == 2 && block == 2 && col == 3) ? 1024 : 4096;
                        rows_source[block].iovs.push_back(
                            {MemoryType::GPU, static_cast<char*>(device) + off, n, false});
                        expected += bytes.substr(off, n);
                    }
                    rows_state->expected.push_back(expected);
                    uris.push_back("row" + std::to_string(block));
                }
                size_t                   row_calls = 0, singles = 0;
                CoalescedPaceWriteClient rows_client(
                    std::make_unique<Delegate>(rows_state),
                    [&](void* d, MemoryType dt, const void* s, MemoryType st, size_t n) {
                        ++singles;
                        return copy(d, dt, s, st, n);
                    },
                    [&](void* d, size_t dp, const void* s, size_t sp, size_t width, size_t height) {
                        ++row_calls;
                        return cudaMemcpy2D(d, dp, s, sp, width, height, cudaMemcpyDeviceToHost) == cudaSuccess;
                    });
                check(rows_client.SaveKvCaches(uris, rows_source).first == ER_OK);
                check(rows_state->saves == 1);
                if (layout == 0)
                    check(row_calls == 4 && singles == 0);
                else if (layout == 1)
                    check(row_calls == 8 && singles == 0);
                else
                    check(row_calls > 0 && singles > 0);
            }
            check(cudaFree(device) == cudaSuccess);
            device = nullptr;
#else
            throw std::runtime_error("GPU checks require a CUDA-enabled test build");
#endif
        }
        std::cout << "COALESCED_WRITE_PASS checks=" << checks.load() << " gpu=" << gpu << "\n";
        return 0;
    } catch (const std::exception& e) {
#if USING_CUDA
        if (device)
            cudaFree(device);
#endif
        std::cerr << e.what() << "\n";
        return 1;
    }
}
