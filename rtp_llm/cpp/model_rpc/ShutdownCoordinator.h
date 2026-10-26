#pragma once

#include <chrono>
#include "rtp_llm/cpp/engine_base/LoopStopGate.h"
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

// This barrier uses RPCs, not collectives: the engine must keep participating in
// its normal forward collectives until every rank has finished its requests.
class ShutdownCoordinator {
public:
    using Deadline = std::chrono::system_clock::time_point;
    struct State {
        bool   draining;
        size_t onflight;
    };
    ShutdownCoordinator(
        std::vector<std::string> addresses,
        size_t                   rank,
        std::function<State()>   state,
        std::function<void()>    begin_shutdown = [] {},
        std::function<LoopStopGate::State(bool, uint64_t)> fence_engine = {});

    grpc::Status handle(const ShutdownRequestPB& request, ShutdownResponsePB* response);
    bool         wait(Deadline deadline);

private:
    bool waitForEngineFence(Deadline deadline);
    bool acceptParked(const std::vector<std::string>& instances, uint64_t step);
    Deadline     effectiveDeadline(Deadline deadline);
    bool         acceptRelease(const std::vector<std::string>& instances);
    grpc::Status call(size_t rank, const ShutdownRequestPB& request, ShutdownResponsePB* response, Deadline deadline);
    std::vector<std::string> releasedInstances();

    const std::vector<std::string>                 addresses_;
    const size_t                                   rank_;
    const std::function<State()>                   state_;
    const std::string                              instance_id_;
    const std::function<void()>                    begin_shutdown_;
    const std::function<LoopStopGate::State(bool, uint64_t)> fence_engine_;
    std::vector<std::string> parked_instances_;
    uint64_t parked_step_ = 0;
    std::string                                    group_id_;
    std::vector<std::unique_ptr<RpcService::Stub>> peers_;
    std::mutex                                     mutex_;
    Deadline                                       deadline_ = Deadline::max();
    std::vector<std::string>                       released_instances_;
};

}  // namespace rtp_llm
