// EOS-only native GPU publication. No early payload copy or CPU snapshot.
// Metadata and payload transfer reuse the Prefill StorageBackend pipeline.
struct CompletionAccounting {
    std::mutex mutex;
    size_t     bytes{0}, jobs{0};
};
struct CompletionLease {
    std::shared_ptr<CompletionAccounting> accounting;
    size_t                                bytes;
    CompletionLease(std::shared_ptr<CompletionAccounting> state, size_t n): accounting(std::move(state)), bytes(n) {}
    ~CompletionLease() {
        std::lock_guard<std::mutex> guard(accounting->mutex);
        accounting->bytes -= bytes;
        --accounting->jobs;
    }
};

bool configureDecodePublication(StorageBackend::BufferResolver resolver) {
    file_resolver_             = resolver;
    const char* feature = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH");
    if (feature && std::string_view(feature) != "0" && std::string_view(feature) != "1") {
        RTP_LLM_LOG_ERROR("ENABLE_DECODE_PUBLISH must be 0 or 1");
        return false;
    }
    decode_publication_enabled_ = remoteCacheDecodePublicationEnabled();
    std::string backend_config = kv_cache_config_.kvcm_model_sdk_config;
    backend_config.erase(
        std::remove_if(backend_config.begin(), backend_config.end(), [](unsigned char c) { return std::isspace(c); }),
        backend_config.end());
    // Backend selection is independent of Decode publication. Ordinary Prefill
    // GDR and B/C toggles need no second experimental PACE enable flag.
    pace_backend_ = backend_config == "[{\"type\":\"pace\"}]"
                    || backend_config == "[{\"type\":\"pace_ssd\"}]";
    const bool ssd = backend_config == "[{\"type\":\"pace_ssd\"}]";
    if (pace_backend_ && ((ssd && (kv_cache_config_.kvcm_read_backend_type != 9 || !gdr_enabled_))
                         || (!ssd && kv_cache_config_.kvcm_read_backend_type == 9))) {
        RTP_LLM_LOG_ERROR("PACE SSD requires backend9/pace_ssd and GDR; DRAM cannot use backend9");
        return false;
    }
    if (!decode_publication_enabled_)
        return true;
    const auto limits = remoteCacheDecodePublicationLimits();
    if (!limits) {
        RTP_LLM_LOG_ERROR("invalid Decode publication pending byte/job budget");
        return false;
    }
    completion_limits_ = *limits;
    file_decode_ = parallelism_config_.role_type == RoleType::DECODE;
    const bool prefill = parallelism_config_.role_type == RoleType::PREFILL;
    const auto& cp = parallelism_config_.prefill_cp_config;
    target_prefill_cp_ = cp.prefill_cp_size;
    target_prefill_tp_ = target_prefill_cp_;
    target_physical_shards_ = cp.kv_cache_sharded ? target_prefill_cp_ : 1;
    file_capture_blocks_ = target_physical_shards_;
    // CP is the target layout, DP is execution parallelism. They are independent.
    // Geometry checks remain in CPSlotMapper / group projection, never a CP2/4 whitelist.
    if (!pace_backend_ || (!file_decode_ && !prefill) || target_prefill_cp_ < 1
        || cache_config_.seq_size_per_block < 1
        || static_cast<size_t>(target_prefill_cp_) > static_cast<size_t>(std::numeric_limits<int>::max()) / cache_config_.seq_size_per_block
        || !cache_config_.mtp_sub_configs.empty() || sp_config_.type != SP_TYPE_NONE
        || autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false)
        || !kv_cache_config_.kvcm_client_config.empty() || kv_cache_config_.kvcm_instance_id_salt.empty()
        || (pace_backend_ && backend_config != (ssd ? "[{\"type\":\"pace_ssd\"}]" : "[{\"type\":\"pace\"}]"))
        || (file_decode_ && (parallelism_config_.tp_size != 1 || parallelism_config_.dp_size < 1))
        || (prefill && (parallelism_config_.tp_size != target_prefill_cp_ || parallelism_config_.dp_size != 1))) {
        RTP_LLM_LOG_ERROR("Decode publication requires matching target P CP layout, positive D DP, synchronous single-token decoding, and native PACE");
        return false;
    }
    if (!gdr_enabled_) {
        RTP_LLM_LOG_ERROR("Decode publication requires GDR; CPU payload staging is not an alternative");
        return false;
    }
    RTP_LLM_LOG_INFO(
        "DTOP_FILE layout target_cp=%d target_tp=%d shards=%d capture_blocks=%d visible_tokens=%zu dp=%ld role=%d",
        target_prefill_cp_,
        target_prefill_tp_,
        target_physical_shards_,
        file_capture_blocks_,
        cache_config_.seq_size_per_block * target_physical_shards_,
        parallelism_config_.dp_size,
        static_cast<int>(parallelism_config_.role_type));
    RTP_LLM_LOG_INFO("DTOP_TRANSPORT backend=%s end_only=1 target_cp=%d target_tp=%d",
                     pace_backend_ ? "pace" : "file",
                     target_prefill_cp_,
                     target_prefill_tp_);
    file_resolver_ = std::move(resolver);
    return true;
}

size_t fileWireBytes(const std::string& tag) const {
    if (!decode_publication_enabled_ || !file_decode_ || target_physical_shards_ == 1) {
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    const auto& group = cache_config_.topology().group(tag);
    if (group.policy.group_type == CacheGroupType::FULL) {
        RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                "DTOP_FILE requires FULL page RR, tag=%s",
                                tag.c_str());
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK
                                && group.seqSizePerBlock() == 128 * target_physical_shards_,
                            "DTOP_FILE requires matching canonical state/SWA span, tag=%s",
                            tag.c_str());
    size_t bytes = 0;
    for (const int layer : cache_config_.layerIdsForGroup(tag)) {
        const auto& physical = cache_config_.physicalGroupForLayer(layer, tag);
        RTP_LLM_CHECK_WITH_INFO(
            physical.kvScaleStrideBytes() == 0, "DTOP_FILE requires packed state/SWA, tag=%s", tag.c_str());
        size_t payload = physical.kvBlockStrideBytes();
        if (group.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            payload = group.spec->k_block_payload_bytes();
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.policy.cp_slice == CpBlockSliceMode::EQUAL_BYTES,
                                    "DTOP_FILE unsupported state slice, tag=%s",
                                    tag.c_str());
        }
        RTP_LLM_CHECK_WITH_INFO(
            payload > 0 && payload % target_physical_shards_ == 0, "DTOP_FILE invalid state payload, tag=%s", tag.c_str());
        bytes += payload / target_physical_shards_;
    }
    return bytes;
}

std::string fileTag(const std::string& tag, int peer) const {
    return file_decode_ ? tag + "#cp" + std::to_string(peer) : tag;
}

// Select completed native checkpoints, not the final mutable slot. In
// particular, floor(E/unit) is only a candidate: all FULL pages and every
// required terminal STATE/SWA page must still be present at that exact key.
StorageRequest planCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end) {
    StorageRequest request;
    if (!decode_publication_enabled_ || !file_decode_ || parallelism_config_.tp_rank != 0
        || resource.decodeCompletionSubmitted())
        return request;
    if (resource.batchSize() != 1 || computed_end == 0)
        return request;
    const size_t base           = cache_config_.seq_size_per_block;
    const size_t canonical_unit = base * target_physical_shards_;
    const size_t completed_full = computed_end / base;
    const auto&  source_keys    = resource.cacheKeys(0);
    if (source_keys.size() < completed_full) {
        RTP_LLM_LOG_WARNING("DTOP_GPU incomplete key vector computed=%zu keys=%zu", computed_end, source_keys.size());
        return request;
    }
    // State/SWA slots must denote the same token boundary as the P key.
    size_t unit = canonical_unit;
    for (const auto& entry : group_policy_->groups()) {
        const auto& group = topology_->group(entry.second.tag);
        if (!entry.second.is_full)
            unit = std::lcm(unit, group.seqSizePerBlock());
    }
    CPSlotMapper mapper(0, target_physical_shards_, static_cast<int>(base));
    size_t       end = computed_end / unit * unit;
    for (; end > 0; end -= unit) {
        const size_t count = end / canonical_unit;
        bool         ready = true;
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec   = entry.second;
            const auto& group  = topology_->group(spec.tag);
            const auto& blocks = resource.blocks(0, spec.tag);
            if (spec.is_full) {
                for (size_t i = 0; i < end / base; ++i)
                    if (i >= blocks.size() || isNullBlockIdx(blocks[i])) {
                        ready = false;
                        break;
                    }
            } else {
                const size_t required = group.reuseBlockCount(count);
                for (size_t key = count - std::min(count, required); key < count; ++key) {
                    const size_t tokens = (key + 1) * canonical_unit;
                    if (tokens % group.seqSizePerBlock()) {
                        ready = false;
                        break;
                    }
                    const size_t index = tokens / group.seqSizePerBlock() - 1;
                    if (index >= blocks.size() || isNullBlockIdx(blocks[index])) {
                        ready = false;
                        break;
                    }
                }
            }
            if (!ready)
                break;
        }
        if (ready)
            break;
    }
    if (end == 0) {
        RTP_LLM_LOG_INFO("DTOP_GPU no_complete_checkpoint computed=%zu", computed_end);
        return request;
    }
    CacheKeysType complete_keys(source_keys.begin(), source_keys.begin() + end / base);
    request.keys = std::make_shared<const CacheKeysType>(mapper.canonicalCacheKeys(complete_keys));
    request.handles.resize(request.keys->size());
    request.decode_completion                            = true;
    request.computed_tokens                              = computed_end;
    request.published_tokens                             = end;
    size_t                                         bytes = 0;
    std::set<std::pair<std::string, BlockIdxType>> unique;
    for (size_t key = 0; key < request.keys->size(); ++key) {
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec  = entry.second;
            const auto& group = topology_->group(spec.tag);
            if (!spec.is_full && request.keys->size() - key > group.reuseBlockCount(request.keys->size()))
                continue;
            const auto&        blocks = resource.blocks(0, spec.tag);
            StorageBlockHandle handle;
            handle.tag = spec.tag;
            if (spec.is_full) {
                for (int peer = 0; peer < target_prefill_tp_; ++peer)
                    handle.peer_blocks.push_back(blocks.at(target_physical_shards_ * key + (target_physical_shards_ > 1 ? peer : 0)));
                handle.block = handle.peer_blocks.front();
            } else {
                handle.block = blocks.at((key + 1) * canonical_unit / group.seqSizePerBlock() - 1);
            }
            auto charge = [&](BlockIdxType block) {
                if (unique.emplace(spec.tag, block).second)
                    bytes += cache_config_.blockSizeBytesForGroup(spec.tag);
            };
            charge(handle.block);
            for (auto peer : handle.peer_blocks)
                charge(peer);
            request.handles[key].push_back(std::move(handle));
        }
    }
    const size_t limit = completion_limits_.bytes;
    const size_t jobs  = completion_limits_.jobs;
    {
        std::lock_guard<std::mutex> guard(completion_accounting_->mutex);
        if (bytes > limit || completion_accounting_->bytes > limit - bytes || completion_accounting_->jobs >= jobs) {
            RTP_LLM_LOG_WARNING("DTOP_GPU admission=rejected retained_bytes=%zu pending_bytes=%zu pending_jobs=%zu",
                                bytes,
                                completion_accounting_->bytes,
                                completion_accounting_->jobs);
            return {};
        }
        auto lease = std::make_shared<CompletionLease>(completion_accounting_, bytes);
        completion_accounting_->bytes += bytes;
        ++completion_accounting_->jobs;
        request.completion_admission = std::move(lease);
    }
    // One fence at successful EOS. No decode-step fence or CPU capture. The
    // task cannot outlive source ownership until prepareWrite has pinned it.
    setCudaDevice();
    check_cuda_value(cudaDeviceSynchronize());
    RTP_LLM_LOG_INFO(
        "DTOP_GPU retained_plan computed=%zu published=%zu tail=%zu pages=%zu bytes=%zu CPU_snapshot_bytes=0",
        computed_end,
        end,
        computed_end - end,
        unique.size(),
        bytes);
    return request;
}

// The P transaction builds identical spec/tag batches and commits actual URIs
// with the same masks and FinishWrite rules. Only the source projection differs:
// D owns all P peers' FULL pages and the unsliced native STATE/SWA page locally.
std::vector<FunctionResponsePB> transferCompletedDecode(const std::vector<FunctionRequestPB>& requests) {
    setCudaDevice();
    struct Batch {
        std::vector<std::string>       tags;
        std::vector<int32_t>           blocks;
        kv_cache_manager::UriStrVec    uris;
        kv_cache_manager::BlockBuffers buffers;
    };
    std::vector<Batch> batches(requests.size());
    CPSlotMapper       mapper(0, target_physical_shards_, cache_config_.seq_size_per_block);
    // Validate and resolve every source before submitting any SDK operation.
    for (size_t peer = 0; peer < requests.size(); ++peer) {
        const auto& remote = requests[peer].remote_request();
        auto&       batch  = batches[peer];
        for (int i = 0; i < remote.block_ids_size(); ++i) {
            const auto& tag   = remote.group_tags(i);
            const int   block = remote.block_ids(i);
            batch.tags.push_back(fileTag(tag, peer));
            batch.blocks.push_back(block);
            batch.uris.push_back(remote.uris(i));
            batch.buffers.emplace_back();
            size_t bytes = 0;
            for (int layer : cache_config_.layerIdsForGroup(tag)) {
                const auto parts =
                    mapper.sliceBlockForPeer(cache_config_, tag, file_resolver_(layer, tag, block), peer);
                for (const auto& part : parts) {
                    RTP_LLM_CHECK_WITH_INFO(part.is_cuda && part.addr && part.size_bytes, "invalid EOS GPU source");
                    batch.buffers.back().iovs.push_back(
                        {kv_cache_manager::MemoryType::GPU, part.addr, part.size_bytes, false});
                    bytes += part.size_bytes;
                }
            }
            RTP_LLM_CHECK_WITH_INFO(bytes == fileWireBytes(tag), "EOS source wire size mismatch tag=%s", tag.c_str());
        }
    }
    std::vector<FunctionResponsePB> responses(requests.size());
    try {
        for (size_t peer = 0; peer < batches.size(); ++peer) {
            auto&      b        = batches[peer];
            auto&      response = *responses[peer].mutable_remote_response();
            const auto result =
                executeTagTransfers(REMOTE_OPERATION_WRITE, b.tags, b.blocks, b.uris, b.buffers, response);
            if (result == TransferResult::TIMEOUT)
                throw StorageOperationTimeout("EOS SDK transfer timeout", StorageQuarantineReason::SDK_TIMEOUT);
            if (result != TransferResult::SUCCESS)
                throw StorageOperationCompletionUnknown("EOS SDK failure cannot prove DMA completion",
                                                        StorageQuarantineReason::INVALID_RESPONSE);
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
        }
    } catch (const StorageOperationCompletionUnknown&) {
        throw;
    } catch (const std::exception& error) {
        throw StorageOperationCompletionUnknown(error.what(), StorageQuarantineReason::INVALID_RESPONSE);
    } catch (...) {
        throw StorageOperationCompletionUnknown("EOS SDK completion unknown",
                                                StorageQuarantineReason::INVALID_RESPONSE);
    }
    return responses;
}
