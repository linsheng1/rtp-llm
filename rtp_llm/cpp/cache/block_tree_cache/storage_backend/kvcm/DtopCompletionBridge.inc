// Included inside KVCMStorageBackend::Impl. Opt-in experiment only:
// D publishes complete canonical P checkpoints after FINISHED. Real PACE
// uses the native registered pools; cross-layout snapshots use CPU staging.
bool configureFileCp2(StorageBackend::BufferResolver resolver) {
    file_resolver_ = resolver;
    file_pace_ = autil::EnvUtil::getEnv("RTP_DTOP_PACE", false);
    file_cp2_ = file_pace_ || autil::EnvUtil::getEnv("RTP_DTOP_FILE_CP2", false);
    std::string backend_config = kv_cache_config_.kvcm_model_sdk_config;
    backend_config.erase(std::remove_if(backend_config.begin(), backend_config.end(),
                                       [](unsigned char c) { return std::isspace(c); }), backend_config.end());
    const bool packed_ssd=autil::EnvUtil::getEnv("RTP_DTOP_PACE_SSD_PACKED_IO", false);
    if (packed_ssd && (!file_pace_ || gdr_enabled_ || backend_config!="[{\"type\":\"pace_ssd\"}]"
                       || kv_cache_config_.kvcm_read_backend_type!=9)) {
        RTP_LLM_LOG_ERROR("explicit SSD compatibility requires backend9/pace_ssd and CPU staging, GDR disabled");
        return false;
    }
    if (!file_cp2_) {
        return true;
    }
    const int requested_tp = autil::EnvUtil::getEnv("RTP_DTOP_FILE_TARGET_P_TP", 2);
    if (!file_pace_ || (requested_tp != 2 && requested_tp != 4)
        || parallelism_config_.prefill_cp_config.prefill_cp_size != requested_tp) {
        RTP_LLM_LOG_ERROR("DTOP native candidate requires matching PACE CP2 or CP4 target (sharded or replicated)");
        return false;
    }
    file_decode_ = parallelism_config_.role_type == RoleType::DECODE;
    const bool prefill = parallelism_config_.role_type == RoleType::PREFILL;
    const auto& cp = parallelism_config_.prefill_cp_config;
    file_target_cp_ = cp.prefill_cp_size;
    file_target_tp_ = autil::EnvUtil::getEnv("RTP_DTOP_FILE_TARGET_P_TP",
                                            prefill ? static_cast<int>(parallelism_config_.tp_size) : file_target_cp_);
    file_shards_ = cp.kv_cache_sharded ? file_target_cp_ : 1;
    file_capture_blocks_ = autil::EnvUtil::getEnv("RTP_DTOP_FILE_CAPTURE_BLOCKS", file_shards_);
    if ((!file_decode_ && !prefill) || cache_config_.seq_size_per_block != 128
        || !cache_config_.mtp_sub_configs.empty() || sp_config_.type != SP_TYPE_NONE
        || !((file_target_tp_ == 2 && (cp.prefill_cp_size == 1 || cp.prefill_cp_size == 2))
             || (file_target_tp_ == 4 && (cp.prefill_cp_size == 1 || cp.prefill_cp_size == 4)))
        || (file_capture_blocks_ != 1 && file_capture_blocks_ != file_shards_ && !(file_shards_ == 1 && file_capture_blocks_ == file_target_cp_))
        || !kv_cache_config_.kvcm_client_config.empty()
        || kv_cache_config_.kvcm_instance_id_salt.empty()
        || kv_cache_config_.kvcm_instance_group.rfind("linshuo_decode_remote_", 0) != 0
        || (file_pace_ ? backend_config != (packed_ssd ? "[{\"type\":\"pace_ssd\"}]" : "[{\"type\":\"pace\"}]") : kv_cache_config_.kvcm_model_sdk_config.find("\"file\"") == std::string::npos)
        || (file_decode_ && !((file_target_tp_ == 2
                              && ((parallelism_config_.tp_size == 1 && parallelism_config_.dp_size == 2)
                                  || (parallelism_config_.tp_size == 2 && parallelism_config_.dp_size == 1)))
                              || (file_target_tp_ == 4 && parallelism_config_.tp_size == 1
                                  && parallelism_config_.dp_size == 4 && parallelism_config_.ep_size == 4)))
        || (prefill && (parallelism_config_.tp_size != file_target_tp_ || parallelism_config_.dp_size != 1))) {
        RTP_LLM_LOG_ERROR("DTOP_FILE rejects unsupported role/layout/configuration");
        return false;
    }
    RTP_LLM_LOG_INFO("DTOP_FILE layout target_cp=%d target_tp=%d shards=%d capture_blocks=%d visible_tokens=%d dp=%ld role=%d",
                     file_target_cp_, file_target_tp_, file_shards_, file_capture_blocks_,
                     128 * std::max(file_shards_, file_capture_blocks_), parallelism_config_.dp_size,
                     static_cast<int>(parallelism_config_.role_type));
    RTP_LLM_LOG_INFO("DTOP_TRANSPORT backend=%s end_only=1 target_cp=%d target_tp=%d",
                     file_pace_ ? "pace" : "file", file_target_cp_, file_target_tp_);
    file_resolver_ = std::move(resolver);
    return true;
}

size_t fileWireBytes(const std::string& tag) const {
    if (!file_cp2_ || !file_decode_ || file_shards_ == 1) {
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    const auto& group = cache_config_.topology().group(tag);
    if (group.policy.group_type == CacheGroupType::FULL) {
        RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                "DTOP_FILE requires FULL page RR, tag=%s", tag.c_str());
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK
                               && group.seqSizePerBlock() == 128 * file_shards_,
                            "DTOP_FILE requires matching canonical state/SWA span, tag=%s", tag.c_str());
    size_t bytes = 0;
    for (const int layer : cache_config_.layerIdsForGroup(tag)) {
        const auto& physical = cache_config_.physicalGroupForLayer(layer, tag);
        RTP_LLM_CHECK_WITH_INFO(physical.kvScaleStrideBytes() == 0,
                                "DTOP_FILE requires packed state/SWA, tag=%s", tag.c_str());
        size_t payload = physical.kvBlockStrideBytes();
        if (group.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            payload = group.spec->k_block_payload_bytes();
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.policy.cp_slice == CpBlockSliceMode::EQUAL_BYTES,
                                    "DTOP_FILE unsupported state slice, tag=%s", tag.c_str());
        }
        RTP_LLM_CHECK_WITH_INFO(payload > 0 && payload % file_shards_ == 0,
                                "DTOP_FILE invalid state payload, tag=%s", tag.c_str());
        bytes += payload / file_shards_;
    }
    return bytes;
}

std::string fileTag(const std::string& tag, int peer) const {
    return file_decode_ ? tag + "#cp" + std::to_string(peer) : tag;
}

void logFileRead(const std::string& tag, const std::vector<int32_t>& blocks,
                 const kv_cache_manager::UriStrVec& uris) const {
    if (file_pace_ && !autil::EnvUtil::getEnv("RTP_DTOP_PAYLOAD_AUDIT", false)) { return; }
    if (!file_cp2_ && !autil::EnvUtil::getEnv("RTP_DTOP_FILE_AUDIT_BASELINE_P", false)) { return; }
    const auto diagnostic_started_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::lock_guard<std::mutex> verification_lock(file_mutex_);
    for (size_t index = 0; index < blocks.size(); ++index) {
        // Each actual SDK Load still runs. Only the extra D2H diagnostic is
        // skipped for immutable private URIs already verified by this P rank.
        if (autil::EnvUtil::getEnv("RTP_DTOP_FILE_VERIFY_ONCE", false)
            && file_verified_uris_.count(uris.at(index))) { continue; }
        size_t bytes = 0;
        uint64_t checksum = 14695981039346656037ULL;
        for (const int layer : cache_config_.layerIdsForGroup(tag)) {
            for (const auto& part : file_resolver_(layer, tag, blocks[index])) {
                RTP_LLM_CHECK_WITH_INFO(part.is_cuda && part.addr && part.size_bytes > 0
                                           && part.size_bytes <= 32 * 1024 * 1024,
                                        "DTOP_FILE invalid read destination");
                std::vector<unsigned char> host(part.size_bytes);
                check_cuda_value(cudaMemcpy(host.data(), part.addr, part.size_bytes, cudaMemcpyDeviceToHost));
                for (const auto byte : host) { checksum ^= byte; checksum *= 1099511628211ULL; }
                bytes += part.size_bytes;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(bytes == fileWireBytes(tag), "DTOP_FILE destination byte mismatch");
        file_verified_uris_.insert(uris.at(index));
        RTP_LLM_LOG_INFO("DTOP_FILE restored tag=%s rank=%ld bytes=%zu fnv64=%lu uri=%s",
                         tag.c_str(), parallelism_config_.tp_rank, bytes, checksum, uris.at(index).c_str());
    }
    RTP_LLM_LOG_INFO("DTOP_FILE read_diagnostic_us=%ld tag=%s blocks=%zu",
        autil::TimeUtility::currentTimeInMicroSeconds() - diagnostic_started_us, tag.c_str(), blocks.size());
}

// Called synchronously before the scheduler frees any source block. The
// narrow experiment accepts only an exact completed CP boundary; it never
// labels a final partial state as an earlier complete checkpoint.
void logPrefillFileWrite(const std::string& tag, const std::vector<int32_t>& blocks,
                 const kv_cache_manager::UriStrVec& uris) const {
    if (file_pace_ && !autil::EnvUtil::getEnv("RTP_DTOP_PAYLOAD_AUDIT", false)) { return; }
    if (!file_cp2_ && !autil::EnvUtil::getEnv("RTP_DTOP_FILE_AUDIT_BASELINE_P", false)) { return; }
    const auto diagnostic_started_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::lock_guard<std::mutex> verification_lock(file_mutex_);
    for (size_t index = 0; index < blocks.size(); ++index) {
        // Each actual SDK Load still runs. Only the extra D2H diagnostic is
        // skipped for immutable private URIs already verified by this P rank.
        size_t bytes = 0;
        uint64_t checksum = 14695981039346656037ULL;
        for (const int layer : cache_config_.layerIdsForGroup(tag)) {
            for (const auto& part : file_resolver_(layer, tag, blocks[index])) {
                RTP_LLM_CHECK_WITH_INFO(part.is_cuda && part.addr && part.size_bytes > 0
                                           && part.size_bytes <= 32 * 1024 * 1024,
                                        "DTOP_FILE invalid read destination");
                std::vector<unsigned char> host(part.size_bytes);
                check_cuda_value(cudaMemcpy(host.data(), part.addr, part.size_bytes, cudaMemcpyDeviceToHost));
                for (const auto byte : host) { checksum ^= byte; checksum *= 1099511628211ULL; }
                bytes += part.size_bytes;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(bytes == fileWireBytes(tag), "DTOP_FILE destination byte mismatch");
        RTP_LLM_LOG_INFO("PREFILL_FILE source tag=%s rank=%ld bytes=%zu fnv64=%lu uri=%s",
                         tag.c_str(), parallelism_config_.tp_rank, bytes, checksum, uris.at(index).c_str());
    }
    RTP_LLM_LOG_INFO("DTOP_FILE read_diagnostic_us=%ld tag=%s blocks=%zu",
        autil::TimeUtility::currentTimeInMicroSeconds() - diagnostic_started_us, tag.c_str(), blocks.size());
}

// Called synchronously before the scheduler frees any source block. The
// narrow experiment accepts only an exact completed CP boundary; it never
// labels a final partial state as an earlier complete checkpoint.
bool captureCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end, bool decode_finished) {
    if (!file_cp2_ || !file_decode_ || parallelism_config_.tp_rank != 0) { return false; }
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (!autil::EnvUtil::getEnv("RTP_DTOP_DECODE_PUBLISH", true)) { return true; }
    RTP_LLM_CHECK_WITH_INFO(file_quarantined_snapshots_.empty(), "DTOP prior CPU transfer completion unknown");
    const size_t visible_tokens = 128 * std::max(file_shards_, file_capture_blocks_);
    try {
        const int max_end = autil::EnvUtil::getEnv("RTP_DTOP_FILE_MAX_COMPUTED_TOKENS", 2048);
        RTP_LLM_CHECK_WITH_INFO(max_end >= 1024 && max_end <= 16384 && max_end % visible_tokens == 0
                                  && resource.batchSize() == 1 && computed_end <= static_cast<size_t>(max_end),
                                "DTOP_FILE invalid staging bounds end=%zu", computed_end);
        const auto& groups = group_policy_->groups();
        const auto& full_keys = resource.cacheKeys(0);
        CPSlotMapper mapper(0, file_shards_, 128);
        auto& pending = resource.decodeCompletionSnapshot();
        if (!decode_finished) {
            if (computed_end <= 512 || computed_end % visible_tokens != 0) { return true; }
            RTP_LLM_CHECK_WITH_INFO(full_keys.size() == computed_end / 128, "DTOP_FILE snapshot key mismatch");
            auto next = std::make_shared<DecodeCompletionSnapshot>();
            next->computed_end = computed_end;
            next->keys = mapper.canonicalCacheKeys(full_keys);
            next->tags.resize(next->keys.size());
            next->payloads.resize(next->keys.size());
            const bool append = pending && pending->keys.size() <= next->keys.size()
                && std::equal(pending->keys.begin(), pending->keys.end(), next->keys.begin());
            setCudaDevice();
            check_cuda_value(cudaDeviceSynchronize());
            size_t total_bytes = 0, copied_bytes = 0;
            for (size_t key = 0; key < next->keys.size(); ++key) {
                for (const auto& [id, group] : groups) {
                    (void)id;
                    const auto& blocks = resource.blocks(0, group.tag);
                    const size_t index = group.is_full ? file_shards_ * key : key;
                    if (index >= blocks.size() || isNullBlockIdx(blocks[index])) { continue; }
                    if (group.is_full) {
                        RTP_LLM_CHECK_WITH_INFO(index + file_shards_ - 1 < blocks.size()
                            && !isNullBlockIdx(blocks[index + file_shards_ - 1]), "DTOP_FILE incomplete FULL snapshot");
                    }
                    next->tags[key].push_back(group.tag);
                    for (const auto& [spec, info] : group_policy_->spec_info_map()) {
                        if (info.group_id != id) { continue; }
                        // FULL pages are immutable; only newly computed pages need D2H.
                        DecodeCompletionSnapshot::Payload payload;
                        if (group.is_full && append && key < pending->payloads.size()) {
                            const auto old = pending->payloads[key].find(spec);
                            if (old != pending->payloads[key].end()) { payload = old->second; }
                        }
                        if (!payload) {
                            const size_t physical = group.is_full ? index + (file_shards_ > 1 ? info.tp_rank : 0) : key;
                            RTP_LLM_CHECK_WITH_INFO(info.tp_rank >= 0 && info.tp_rank < file_target_tp_
                                && physical < blocks.size() && !isNullBlockIdx(blocks[physical]), "DTOP_FILE bad snapshot source");
                            payload = std::make_shared<std::vector<unsigned char>>();
                            for (const int layer : cache_config_.layerIdsForGroup(info.tag)) {
                                auto parts = mapper.sliceBlockForPeer(cache_config_, info.tag,
                                    file_resolver_(layer, info.tag, blocks[physical]), info.tp_rank);
                                for (const auto& part : parts) {
                                    RTP_LLM_CHECK_WITH_INFO(part.is_cuda && part.addr && part.size_bytes > 0
                                        && part.size_bytes <= 32 * 1024 * 1024, "DTOP_FILE invalid snapshot part");
                                    const size_t offset = payload->size();
                                    RTP_LLM_CHECK_WITH_INFO(total_bytes + offset + part.size_bytes <= 512 * 1024 * 1024,
                                                            "DTOP_FILE per-request staging bound exceeded");
                                    payload->resize(offset + part.size_bytes);
                                    check_cuda_value(cudaMemcpy(payload->data() + offset, part.addr, part.size_bytes,
                                                                cudaMemcpyDeviceToHost));
                                    copied_bytes += part.size_bytes;
                                }
                            }
                        }
                        RTP_LLM_CHECK_WITH_INFO(payload->size() == group.block_size_bytes, "DTOP_FILE snapshot wire size");
                        total_bytes += payload->size();
                        RTP_LLM_CHECK_WITH_INFO(total_bytes <= 512 * 1024 * 1024, "DTOP_FILE per-request staging bound exceeded");
                        next->payloads[key].emplace(spec, std::move(payload));
                    }
                }
            }
            RTP_LLM_CHECK_WITH_INFO(!next->tags.empty() && next->tags.back().size() == groups.size(),
                                    "DTOP_FILE snapshot lacks state/SWA");
            pending = std::move(next);
            RTP_LLM_LOG_INFO("DTOP_FILE staged_only computed_end=%zu keys=%zu host_bytes=%zu copied_bytes=%zu remote_calls=0",
                             computed_end, pending->keys.size(), total_bytes, copied_bytes);
            return true;
        }
        // Only StreamCacheResource's FINISHED/error-free release path enters here.
        // Take ownership before starting IO; a repeat callback cannot publish twice.
        auto snapshot = std::move(pending);
        if (!snapshot) {
            RTP_LLM_LOG_INFO("DTOP_FILE completion_no_checkpoint computed_end=%zu remote_calls=0", computed_end);
            return true;
        }
        RTP_LLM_CHECK_WITH_INFO(snapshot->computed_end <= computed_end
            && computed_end - snapshot->computed_end < visible_tokens, "DTOP_FILE stale completion snapshot");
        const auto canonical = mapper.canonicalCacheKeys(full_keys);
        RTP_LLM_CHECK_WITH_INFO(snapshot->keys.size() <= canonical.size()
            && std::equal(snapshot->keys.begin(), snapshot->keys.end(), canonical.begin()), "DTOP_FILE completion prefix changed");
        StorageRequest request;
        request.keys = std::make_shared<const CacheKeysType>(snapshot->keys);
        request.handles.resize(snapshot->keys.size());
        for (size_t key = 0; key < snapshot->keys.size(); ++key) {
            // getNeedWriteGroups only uses non-null handles as a presence mask.
            // No GPU address or recycled block index is used by publication.
            for (const auto& tag : snapshot->tags[key]) { request.handles[key].push_back({tag, 0}); }
        }
        std::vector<std::string> names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, request.keys->size(), names),
                                "DTOP_FILE invalid completion group mask");
        RTP_LLM_LOG_INFO("DTOP_FILE completion_upload_begin computed_end=%zu checkpoint_end=%zu",
                         computed_end, snapshot->computed_end);
        const auto write_started_us = autil::TimeUtility::currentTimeInMicroSeconds();
        setCudaDevice();
        check_cuda_value(cudaDeviceSynchronize());
        const auto trace = nextTraceId("dtop_file_write", write_trace_sequence_);
        auto [success, location] = client_wrapper_->getWriteLocation(
            "", trace, *request.keys, {}, names, 120, kv_cache_config_.kvcm_min_replica_count);
        RTP_LLM_CHECK_WITH_INFO(success, "DTOP_FILE StartWrite failed");
        bool finish_attempted = false;
        file_quarantined_snapshots_.push_back(snapshot);
        const kv_cache_manager::Locations empty;
        try {
            const auto indices = unmaskedKeyIndices(location.block_mask, request.keys->size());
            RTP_LLM_CHECK_WITH_INFO(indices.size() == location.locations.size(), "DTOP_FILE write mask mismatch");
            size_t transferred = 0;
            for (size_t index = 0; index < indices.size(); ++index) {
                const size_t key = indices[index];
                for (auto& unit : location.locations[index]) {
                    const auto& info = group_policy_->spec_info_map().at(unit.spec_name);
                    const auto& group = groups.at(info.group_id);
                    auto payload = snapshot->payloads.at(key).at(unit.spec_name);
                    kv_cache_manager::BlockBuffers buffers(1);
                    buffers[0].iovs.push_back({kv_cache_manager::MemoryType::CPU,
                                               payload->data(), payload->size(), false});
                    const size_t bytes = payload->size();
                    uint64_t checksum = 14695981039346656037ULL;
                    const bool audit_payload = !file_pace_ || autil::EnvUtil::getEnv("RTP_DTOP_PAYLOAD_AUDIT", false);
                    if (audit_payload) {
                        for (const auto byte : *payload) { checksum ^= byte; checksum *= 1099511628211ULL; }
                    }
                    RTP_LLM_CHECK_WITH_INFO(bytes == group.block_size_bytes, "DTOP_FILE wire byte mismatch");
                    auto [saved, actual] = client_wrapper_->saveKvCachesForTag(
                        fileTag(info.tag, info.tp_rank), {unit.uri}, buffers);
                    RTP_LLM_CHECK_WITH_INFO(saved && (actual.empty() || actual.size() == 1), "DTOP_FILE payload save failed");
                    if (!actual.empty() && !actual[0].empty()) { unit.uri = actual[0]; }
                    transferred += bytes;
                    if (audit_payload) RTP_LLM_LOG_INFO("DTOP_FILE source key=%ld spec=%s bytes=%zu fnv64=%lu uri=%s computed_end=%zu",
                                     (*request.keys)[key], unit.spec_name.c_str(), bytes, checksum, unit.uri.c_str(), snapshot->computed_end);
                }
            }
            file_quarantined_snapshots_.clear();
            finish_attempted = true;
            RTP_LLM_CHECK_WITH_INFO(client_wrapper_->finishWrite(
                "", nextTraceId("dtop_file_finish", finish_write_trace_sequence_), location.write_session_id,
                static_cast<kv_cache_manager::BlockMaskOffset>(location.locations.size()), location.locations),
                "DTOP_FILE FinishWrite failed/ambiguous");
            RTP_LLM_LOG_INFO("DTOP_FILE committed computed_end=%zu canonical_keys=%zu new_locations=%zu bytes=%zu dp_rank=%ld",
                             snapshot->computed_end, request.keys->size(), location.locations.size(), transferred, parallelism_config_.dp_rank);
            RTP_LLM_LOG_INFO("DTOP_FILE write_duration_us=%ld computed_end=%zu bytes=%zu",
                             autil::TimeUtility::currentTimeInMicroSeconds() - write_started_us, snapshot->computed_end, transferred);
        } catch (...) {
            // A lost commit ACK is ambiguous. Never follow a commit attempt
            // with an abort which could revoke a successful publication.
            if (!finish_attempted) {
                try { client_wrapper_->finishWrite("", nextTraceId("dtop_file_abort", finish_write_trace_sequence_),
                                                   location.write_session_id, kv_cache_manager::BlockMaskOffset{0}, empty); }
                catch (...) { RTP_LLM_LOG_WARNING("DTOP_FILE abort failed"); }
            }
            throw;
        }
    } catch (const std::exception& error) {
        resource.decodeCompletionSnapshot().reset();
        RTP_LLM_LOG_ERROR("DTOP_FILE capture failed (no success claimed): %s", error.what());
    } catch (...) {
        resource.decodeCompletionSnapshot().reset();
        RTP_LLM_LOG_ERROR("DTOP_FILE capture failed (no success claimed): unknown exception");
    }
    // This experiment must never also publish the legacy TP1/base-128 tree
    // representation. Returning true means handled, not a successful commit.
    return true;
}
