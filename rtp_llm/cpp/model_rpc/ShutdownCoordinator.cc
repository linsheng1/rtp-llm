#include "rtp_llm/cpp/model_rpc/ShutdownCoordinator.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <thread>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

namespace rtp_llm {
namespace {
std::string instanceId() {
    std::random_device random;
    return std::to_string(random()) + "-" + std::to_string(random()) + "-" + std::to_string(random()) + "-"
           + std::to_string(random());
}
}  // namespace

ShutdownCoordinator::ShutdownCoordinator(std::vector<std::string> addresses,
                                         size_t                   rank,
                                         std::function<State()>   state,
                                         std::function<void()>    begin_shutdown,
                                         std::function<LoopStopGate::State(bool, uint64_t)> fence_engine):
    addresses_(std::move(addresses)),
    rank_(rank),
    state_(std::move(state)),
    instance_id_(instanceId()),
    begin_shutdown_(std::move(begin_shutdown)),
    fence_engine_(std::move(fence_engine)) {
    if (addresses_.empty() || rank_ >= addresses_.size()) {
        throw std::invalid_argument("invalid shutdown group membership");
    }
    for (const auto& address : addresses_) {
        group_id_ += std::to_string(address.size()) + ":" + address;
        peers_.push_back(RpcService::NewStub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials())));
    }
}

ShutdownCoordinator::Deadline ShutdownCoordinator::effectiveDeadline(Deadline deadline) {
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_ = std::min(deadline_, deadline);
    return deadline_;
}

std::vector<std::string> ShutdownCoordinator::releasedInstances() {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_instances_;
}

bool ShutdownCoordinator::acceptRelease(const std::vector<std::string>& instances) {
    const auto state = state_();
    if (instances.size() != addresses_.size() || instances[rank_] != instance_id_ || !state.draining || state.onflight
        || std::any_of(instances.begin(), instances.end(), [](const auto& id) { return id.empty(); })) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    released_instances_ = instances;
    return true;
}

grpc::Status ShutdownCoordinator::handle(const ShutdownRequestPB& request, ShutdownResponsePB* response) {
    if (request.group_id() != group_id_) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "shutdown group membership mismatch"};
    }
    if (request.begin_shutdown()) {
        if (request.deadline_unix_ms() <= 0) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "shutdown requires an absolute deadline"};
        }
        effectiveDeadline(Deadline(std::chrono::milliseconds(request.deadline_unix_ms())));
        begin_shutdown_();
    }
    response->set_deadline_unix_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(effectiveDeadline(Deadline::max()).time_since_epoch())
            .count());
    if (request.drained_instances_size()
        && !acceptRelease({request.drained_instances().begin(), request.drained_instances().end()})) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "stale or premature shutdown release"};
    }
    if (request.fence_engine() || request.parked_instances_size()) {
        // A fence cannot cancel active work or precede group request drain.
        if (!fence_engine_ || releasedInstances().empty()) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "engine fence requires supported, drained group"};
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (parked_instances_.empty()) {
            fence_engine_(true, request.engine_step());
        }
    }
    if (request.parked_instances_size()
        && !acceptParked({request.parked_instances().begin(), request.parked_instances().end()}, request.engine_step())) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "stale or premature engine fence certificate"};
    }
    if (fence_engine_) {
        const auto loop = fence_engine_(false, 0);
        response->set_engine_fence_supported(loop.supported);
        response->set_engine_step(loop.step);
        response->set_engine_parked(loop.parked);
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& instance : parked_instances_) {
            response->add_parked_instances(instance);
        }
    }
    const auto state = state_();
    response->set_draining(state.draining);
    response->set_onflight(state.onflight);
    response->set_instance_id(instance_id_);
    for (const auto& instance : releasedInstances()) {
        response->add_drained_instances(instance);
    }
    return grpc::Status::OK;
}

grpc::Status ShutdownCoordinator::call(size_t                   rank,
                                       const ShutdownRequestPB& request,
                                       ShutdownResponsePB*      response,
                                       Deadline                 deadline) {
    if (rank == rank_) {
        return handle(request, response);
    }
    grpc::ClientContext context;
    context.set_deadline(std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(200)));
    FunctionRequestPB wire_request;
    *wire_request.mutable_shutdown_request() = request;
    FunctionResponsePB wire_response;
    auto               status = peers_[rank]->ExecuteFunction(&context, wire_request, &wire_response);
    if (status.ok() && !wire_response.has_shutdown_response()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "peer does not support shutdown coordination"};
    }
    if (status.ok()) {
        *response = wire_response.shutdown_response();
    }
    return status;
}

bool ShutdownCoordinator::wait(Deadline deadline) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // A previously accepted same-step certificate remains sufficient when
        // an observer has already closed its RPC server. Do not spend the
        // deadline trying to rediscover a peer that shut down successfully.
        if (!parked_instances_.empty()) {
            return true;
        }
    }
    ShutdownRequestPB request;
    request.set_group_id(group_id_);
    request.set_begin_shutdown(true);
    deadline = effectiveDeadline(deadline);
    while (std::chrono::system_clock::now() < deadline) {
        deadline = effectiveDeadline(deadline);
        request.set_deadline_unix_ms(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count());
        auto instances = releasedInstances();
        if (instances.empty()) {
            std::vector<std::string> observed(addresses_.size());
            bool                     all_drained = true;
            for (size_t i = 0; i < addresses_.size() && std::chrono::system_clock::now() < deadline; ++i) {
                ShutdownResponsePB response;
                if (!call(i, request, &response, deadline).ok()) {
                    // A disappeared peer is not evidence that it drained.
                    all_drained = false;
                    continue;
                }
                if (response.deadline_unix_ms() > 0) {
                    deadline = effectiveDeadline(Deadline(std::chrono::milliseconds(response.deadline_unix_ms())));
                }
                if (response.drained_instances_size()
                    && acceptRelease({response.drained_instances().begin(), response.drained_instances().end()})) {
                    instances = releasedInstances();
                    break;
                }
                observed[i] = response.instance_id();
                all_drained = all_drained && response.draining() && response.onflight() == 0;
            }
            if (instances.empty() && all_drained && acceptRelease(observed)) {
                instances = std::move(observed);
            }
        }
        if (!instances.empty()) {
            // Preserve the all-drained certificate at each peer before exiting.
            // Peers can also relay it to a lagging rank after the observer exits.
            for (const auto& instance : instances) {
                request.add_drained_instances(instance);
            }
            std::vector<bool> delivered(addresses_.size(), false);
            do {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    // Another rank may already have completed phase two while
                    // this waiter was still acknowledging request drain.
                    if (!parked_instances_.empty()) {
                        return true;
                    }
                }
                deadline = effectiveDeadline(deadline);
                request.set_deadline_unix_ms(
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count());
                for (size_t i = 0; i < addresses_.size() && std::chrono::system_clock::now() < deadline; ++i) {
                    if (!delivered[i]) {
                        ShutdownResponsePB response;
                        delivered[i] = call(i, request, &response, deadline).ok();
                        if (delivered[i] && response.deadline_unix_ms() > 0) {
                            deadline =
                                effectiveDeadline(Deadline(std::chrono::milliseconds(response.deadline_unix_ms())));
                        }
                    }
                }
                if (std::all_of(delivered.begin(), delivered.end(), [](bool done) { return done; })) {
                    break;
                }
                std::this_thread::sleep_until(
                    std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
            } while (std::chrono::system_clock::now() < deadline);
            // Even if a peer exits before acknowledging delivery, the
            // certificate proves there are no user requests left to interrupt.
            return fence_engine_ ? waitForEngineFence(deadline) : true;
        }
        std::this_thread::sleep_until(
            std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
    }
    return false;
}

bool ShutdownCoordinator::acceptParked(const std::vector<std::string>& instances, uint64_t step) {
    if (!fence_engine_ || instances != releasedInstances()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto loop = fence_engine_(false, 0);
    if (!loop.supported || !loop.parked || loop.step != step) {
        return false;
    }
    if (!parked_instances_.empty() && (parked_instances_ != instances || parked_step_ != step)) {
        return false;
    }
    parked_instances_ = instances;
    parked_step_ = step;
    return true;
}

bool ShutdownCoordinator::waitForEngineFence(Deadline deadline) {
    if (rank_ != 0) {
        // One publisher avoids a race where simultaneous observers close an
        // RPC server while another observer is still delivering certificates.
        // Followers acknowledge through handle(), then may stop immediately.
        while (std::chrono::system_clock::now() < (deadline = effectiveDeadline(deadline))) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!parked_instances_.empty()) {
                    return true;
                }
            }
            std::this_thread::sleep_until(std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(10)));
        }
        return false;
    }
    ShutdownRequestPB request;
    request.set_group_id(group_id_);
    request.set_fence_engine(true);
    uint64_t target = fence_engine_(false, 0).step;
    while (std::chrono::system_clock::now() < (deadline = effectiveDeadline(deadline))) {
        request.set_engine_step(target);
        const auto instances = releasedInstances();
        for (const auto& id : instances) { request.add_drained_instances(id); }
        bool all_parked = true;
        uint64_t observed = target;
        for (size_t i = 0; i < addresses_.size(); ++i) {
            ShutdownResponsePB response;
            if (!call(i, request, &response, deadline).ok() || !response.engine_fence_supported()) {
                all_parked = false;
                continue;
            }
            if (response.parked_instances_size()
                && acceptParked({response.parked_instances().begin(), response.parked_instances().end()}, response.engine_step())) {
                target = response.engine_step();
                all_parked = true;
                break;
            }
            observed = std::max(observed, response.engine_step());
            all_parked = all_parked && response.engine_parked() && response.engine_step() == target;
        }
        request.clear_drained_instances();
        if (all_parked && acceptParked(instances, target)) {
            // Publish an incarnation-bound same-step certificate before any RPC
            // server closes. Surviving peers relay it if an observer exits.
            request.set_engine_step(target);
            for (const auto& id : instances) { request.add_parked_instances(id); }
            std::vector<bool> delivered(addresses_.size(), false);
            do {
                for (size_t i = 0; i < addresses_.size(); ++i) {
                    if (!delivered[i]) {
                        ShutdownResponsePB response;
                        delivered[i] = call(i, request, &response, deadline).ok();
                    }
                }
                if (std::all_of(delivered.begin(), delivered.end(), [](bool done) { return done; })) {
                    return true;
                }
                std::this_thread::sleep_until(std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(10)));
            } while (std::chrono::system_clock::now() < deadline);
            // A missing ACK is not proof that the lagging rank knows the fence.
            return false;
        }
        target = observed;
        std::this_thread::sleep_until(std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(10)));
    }
    return false;
}

}  // namespace rtp_llm
