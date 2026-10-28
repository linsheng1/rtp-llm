// Included inside KVCMStorageBackend::Impl. EOS-only GPU source projection;
// the ordinary Prefill transaction and its PACE batching remain unchanged.

bool configureDecodePublication(const CacheTopology& topology,
                                const StorageBackend::BufferResolver& resolver,
                                const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
    if (!decode_publication_enabled_) {
        return true;
    }
    try {
        kvcm::SdkWrapperConfig sdk;
        sdk.parseBackendConfigs(kv_cache_config_.kvcm_model_sdk_config);
        if (sdk.sdk_backend_configs().size() != 1
            || !std::dynamic_pointer_cast<kvcm::TairMempoolSdkConfig>(sdk.sdk_backend_configs().front())) {
            RTP_LLM_LOG_ERROR("Decode publication requires one DRAM PACE backend");
            return false;
        }
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("invalid Decode PACE configuration: %s", error.what());
        return false;
    }
    const auto& cp = parallelism_config_.prefill_cp_config;
    target_prefill_cp_ = cp.prefill_cp_size;
    target_physical_shards_ = cp.kv_cache_sharded ? target_prefill_cp_ : 1;
    const bool prefill = parallelism_config_.role_type == RoleType::PREFILL;
    if ((!is_decode_role_ && !prefill) || target_prefill_cp_ < 1 || cache_config_.seq_size_per_block < 1
        || static_cast<size_t>(target_prefill_cp_)
               > static_cast<size_t>(std::numeric_limits<int>::max()) / cache_config_.seq_size_per_block
        || !cache_config_.mtp_sub_configs.empty() || sp_config_.type != SP_TYPE_NONE
        || autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false)
        || !kv_cache_config_.kvcm_client_config.empty() || kv_cache_config_.kvcm_instance_id_salt.empty()
        || (kv_cache_config_.kvcm_read_backend_type != 0 && kv_cache_config_.kvcm_read_backend_type != 3)
        || (is_decode_role_ && (parallelism_config_.tp_size != 1 || parallelism_config_.dp_size < 1))
        || (prefill && (parallelism_config_.tp_size != target_prefill_cp_ || parallelism_config_.dp_size != 1))) {
        RTP_LLM_LOG_ERROR("Decode publication requires P CP / D TP1-DP, DRAM, and synchronous non-speculative decoding");
        return false;
    }
    if (is_decode_role_) {
        size_t bytes = 0;
        std::unordered_set<const DeviceBlockPool*> seen;
        for (const auto& group : topology.groups()) {
            if (!group.policy.enable_prefix_reuse) {
                continue;
            }
            const auto& pool = pool_resolver(group.tag);
            if (!pool || pool->where() != MemoryType::MEMORY_GPU || !pool->getTotalSizeBytes()) {
                RTP_LLM_LOG_ERROR("Decode publication requires a native GPU pool for tag=%s", group.tag.c_str());
                return false;
            }
            if (seen.insert(pool.get()).second) {
                if (pool->getTotalSizeBytes() > std::numeric_limits<size_t>::max() - bytes) {
                    return false;
                }
                bytes += pool->getTotalSizeBytes();
            }
        }
        const auto limits = remoteCacheDecodePublicationLimits(bytes);
        if (!limits) {
            RTP_LLM_LOG_ERROR("invalid Decode publication byte/job budget");
            return false;
        }
        completion_budget_ = std::make_shared<DecodePublicationBudget>(limits->bytes, limits->jobs);
        publication_resolver_ = resolver;
        RTP_LLM_LOG_INFO("DTOP budget pool=%zu bytes=%zu jobs=%zu cp=%d shards=%d",
                         bytes, limits->bytes, limits->jobs, target_prefill_cp_, target_physical_shards_);
    }
    return true;
}

size_t publicationWireBytes(const std::string& tag) const {
    if (!decode_publication_enabled_ || !is_decode_role_ || target_physical_shards_ == 1) {
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    const auto& group = cache_config_.topology().group(tag);
    if (group.policy.group_type == CacheGroupType::FULL) {
        RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                "Decode FULL projection requires page round-robin, tag=%s", tag.c_str());
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK
                                && group.seqSizePerBlock() == cache_config_.seq_size_per_block * target_physical_shards_,
                            "Decode state/SWA checkpoint span mismatch, tag=%s", tag.c_str());
    size_t bytes = 0;
    for (int layer : cache_config_.layerIdsForGroup(tag)) {
        const auto& physical = cache_config_.physicalGroupForLayer(layer, tag);
        RTP_LLM_CHECK_WITH_INFO(physical.kvScaleStrideBytes() == 0,
                                "Decode state/SWA requires packed payload, tag=%s", tag.c_str());
        size_t payload = physical.kvBlockStrideBytes();
        if (group.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            payload = group.spec->k_block_payload_bytes();
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.policy.cp_slice == CpBlockSliceMode::EQUAL_BYTES,
                                    "Decode unsupported state/SWA slice, tag=%s", tag.c_str());
        }
        RTP_LLM_CHECK_WITH_INFO(payload && payload % target_physical_shards_ == 0,
                                "Decode indivisible state/SWA payload, tag=%s", tag.c_str());
        RTP_LLM_CHECK(payload / target_physical_shards_ <= std::numeric_limits<size_t>::max() - bytes);
        bytes += payload / target_physical_shards_;
    }
    return bytes;
}

std::string publicationPeerTag(const std::string& tag, size_t peer) const {
    return is_decode_role_ ? tag + "#cp" + std::to_string(peer) : tag;
}

StorageRequest planCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end) {
    StorageRequest request;
    if (!decode_publication_enabled_ || !is_decode_role_ || parallelism_config_.tp_rank != 0
        || completion_stopped_.load() || resource.decodeCompletionSubmitted() || resource.batchSize() != 1 || !computed_end) {
        return request;
    }
    const auto& handoff = resource.decodeCompletionLifecycle().prefillComputedEnd();
    if (!handoff || *handoff < 0 || static_cast<uint64_t>(*handoff) > computed_end) {
        // Legacy P has no assertion. Preserve serving, but do not guess which
        // positions were computed on P or publish an unvalidated D checkpoint.
        return request;
    }
    const size_t base = cache_config_.seq_size_per_block;
    const size_t canonical = base * target_physical_shards_;
    const auto& keys = resource.cacheKeys(0);
    if (keys.size() < computed_end / base) {
        return request;
    }
    size_t unit = canonical;
    for (const auto& entry : group_policy_->groups()) {
        if (entry.second.is_full) {
            continue;
        }
        const size_t span = topology_->group(entry.second.tag).seqSizePerBlock();
        const size_t quotient = unit / std::gcd(unit, span);
        RTP_LLM_CHECK(span && quotient <= std::numeric_limits<size_t>::max() / span);
        unit = quotient * span;
    }
    size_t end = computed_end / unit * unit;
    for (; end; end -= unit) {
        bool ready = true;
        const size_t count = end / canonical;
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec = entry.second;
            const auto& group = topology_->group(spec.tag);
            const auto& blocks = resource.blocks(0, spec.tag);
            if (spec.is_full) {
                for (size_t i = 0; i < end / base; ++i) {
                    if (i >= blocks.size() || isNullBlockIdx(blocks[i])) {
                        ready = false;
                        break;
                    }
                }
            } else {
                const size_t required = std::min(count, group.reuseBlockCount(count));
                for (size_t key = count - required; key < count; ++key) {
                    const size_t tokens = (key + 1) * canonical;
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
            if (!ready) {
                break;
            }
        }
        if (ready) {
            break;
        }
    }
    if (!end || end <= static_cast<uint64_t>(*handoff)) {
        return request;
    }
    CPSlotMapper mapper(0, target_physical_shards_, static_cast<int>(base));
    CacheKeysType full_keys(keys.begin(), keys.begin() + end / base);
    request.keys = std::make_shared<const CacheKeysType>(mapper.canonicalCacheKeys(full_keys));
    request.handles.resize(request.keys->size());
    request.decode_completion = true;
    request.computed_tokens = computed_end;
    request.published_tokens = end;
    request.prefill_computed_tokens = *handoff;
    size_t bytes = 0;
    std::unordered_map<const DeviceBlockPool*, std::unordered_set<BlockIdxType>> seen;
    for (size_t key = 0; key < request.keys->size(); ++key) {
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec = entry.second;
            const auto& group = topology_->group(spec.tag);
            if (!spec.is_full && request.keys->size() - key > group.reuseBlockCount(request.keys->size())) {
                continue;
            }
            const auto& blocks = resource.blocks(0, spec.tag);
            StorageBlockHandle handle;
            handle.tag = spec.tag;
            if (spec.is_full) {
                for (int peer = 0; peer < target_prefill_cp_; ++peer) {
                    handle.peer_blocks.push_back(blocks.at(target_physical_shards_ * key
                                                          + (target_physical_shards_ > 1 ? peer : 0)));
                }
                handle.block = handle.peer_blocks.front();
            } else {
                handle.block = blocks.at((key + 1) * canonical / group.seqSizePerBlock() - 1);
            }
            const auto& pool = pools_by_tag_.at(spec.tag);
            auto charge = [&](BlockIdxType block) {
                if (seen[pool.get()].insert(block).second) {
                    RTP_LLM_CHECK(pool->blockSizeBytes() <= std::numeric_limits<size_t>::max() - bytes);
                    bytes += pool->blockSizeBytes();
                }
            };
            charge(handle.block);
            for (auto peer : handle.peer_blocks) {
                charge(peer);
            }
            request.handles[key].push_back(std::move(handle));
        }
    }
    request.completion_budget_lease = completion_budget_->tryAcquire(bytes);
    if (!request.completion_budget_lease) {
        RTP_LLM_LOG_WARNING("DTOP budget rejected bytes=%zu computed=%zu", bytes, computed_end);
        return {};
    }
    // One EOS fence while the request still owns every source page. The
    // generic prepareWrite then pins the same pages before request release.
    setCudaDevice();
    check_cuda_value(cudaDeviceSynchronize());
    RTP_LLM_LOG_INFO("DTOP retained computed=%zu published=%zu tail=%zu bytes=%zu CPU_snapshot_bytes=0",
                     computed_end, end, computed_end - end, bytes);
    return request;
}

std::vector<FunctionResponsePB> transferCompletedDecode(const std::vector<FunctionRequestPB>& requests,
                                                       bool& transfer_attempted,
                                                       bool& source_may_be_touched) {
    setCudaDevice();
    struct Batch {
        std::vector<std::string> tags;
        std::vector<int32_t> blocks;
        kv_cache_manager::UriStrVec uris;
        kv_cache_manager::BlockBuffers buffers;
    };
    std::vector<Batch> batches(requests.size());
    std::vector<FunctionResponsePB> responses(requests.size());
    CPSlotMapper mapper(0, target_physical_shards_, cache_config_.seq_size_per_block);
    // Prebuild error payloads and all source descriptors before the first SDK
    // operation. A failure after any peer begins is conservatively unknown.
    const auto unknown = std::make_exception_ptr(StorageOperationCompletionUnknown(
        "Decode PACE source completion unknown", StorageQuarantineReason::INVALID_RESPONSE));
    const auto timeout = std::make_exception_ptr(StorageOperationTimeout(
        "Decode PACE transfer timed out", StorageQuarantineReason::SDK_TIMEOUT));
    for (size_t peer = 0; peer < requests.size(); ++peer) {
        const auto& remote = requests[peer].remote_request();
        (void)responses[peer].mutable_remote_response();
        RTP_LLM_CHECK(remote.block_ids_size() == remote.group_tags_size()
                      && remote.block_ids_size() == remote.uris_size());
        auto& batch = batches[peer];
        for (int i = 0; i < remote.block_ids_size(); ++i) {
            const auto& tag = remote.group_tags(i);
            const int block = remote.block_ids(i);
            batch.tags.push_back(publicationPeerTag(tag, peer));
            batch.blocks.push_back(block);
            batch.uris.push_back(remote.uris(i));
            batch.buffers.emplace_back();
            size_t bytes = 0;
            for (int layer : cache_config_.layerIdsForGroup(tag)) {
                const auto parts = mapper.sliceBlockForPeer(
                    cache_config_, tag, publication_resolver_(layer, tag, block), peer);
                for (const auto& part : parts) {
                    RTP_LLM_CHECK(part.is_cuda && part.addr && part.size_bytes);
                    batch.buffers.back().iovs.push_back(
                        {kv_cache_manager::MemoryType::GPU, part.addr, part.size_bytes, false});
                    RTP_LLM_CHECK(part.size_bytes <= std::numeric_limits<size_t>::max() - bytes);
                    bytes += part.size_bytes;
                }
            }
            RTP_LLM_CHECK_WITH_INFO(bytes == publicationWireBytes(tag), "Decode wire size mismatch tag=%s", tag.c_str());
        }
    }
    try {
        for (size_t peer = 0; peer < batches.size(); ++peer) {
            auto& batch = batches[peer];
            if (batch.tags.empty()) {
                continue;
            }
            auto& response = *responses[peer].mutable_remote_response();
            transfer_attempted = true;
            source_may_be_touched = true;
            const auto result = executeTagTransfers(
                REMOTE_OPERATION_WRITE, batch.tags, batch.blocks, batch.uris, batch.buffers, response);
            if (result == TransferResult::TIMEOUT) {
                std::rethrow_exception(timeout);
            }
            if (result != TransferResult::SUCCESS) {
                std::rethrow_exception(unknown);
            }
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
        }
    } catch (const StorageOperationCompletionUnknown&) {
        throw;
    } catch (...) {
        std::rethrow_exception(unknown);
    }
    source_may_be_touched = false;
    return responses;
}
