#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

namespace {
void bindMtpLookahead(const BatchKVCacheResourcePtr& resource, const CompleteTokenIdsPtr& tokens, int unit) {
    if (!resource->cache_key_lookahead) return;
    resource->raw_prefix_keys.resize(resource->batchSize());
    for (int row = 0; row < resource->batchSize(); ++row) {
        auto keys = resource->cacheKeys(row);
        resource->raw_prefix_keys[row] = keys;
        const auto* ids = tokens->data(row);
        for (size_t key = 0; key < keys.size(); ++key) {
            const size_t h = (key + 1) * unit;
            if (h < static_cast<size_t>(tokens->seqLength())) {
                keys[key] = hashInt64Array(keys[key], ids + h, ids + h + 1);
                const auto media = tokens->multimodalCacheIdentity(h, 1);
                if (!media.empty()) keys[key] = hashInt64Array(keys[key], media.data(), media.data() + media.size());
            }
        }
        resource->setBatchCacheKeys(row, keys);
        resource->cacheResource(row).ensureLinearBlockDependencies();
    }
}
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   int64_t                 initial_hash) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    // Initial fill: compute cache_keys for all blocks, including the final partial block.
    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->clearCacheKeys(i);

        int64_t rolling_hash = initial_hash;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            const int pos       = index * seq_size_per_block;
            const int block_len = std::min(seq_size_per_block, seq_len - pos);
            rolling_hash        = rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
            const auto identity = complete_token_ids->multimodalCacheIdentity(pos, block_len);
            if (!identity.empty()) {
                rolling_hash = hashInt64Array(rolling_hash, identity.data(), identity.data() + identity.size());
            }
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
    bindMtpLookahead(batch_kv_cache_resource, complete_token_ids, seq_size_per_block);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();
    if (batch_kv_cache_resource->cache_key_lookahead) {
        if (batch_kv_cache_resource->raw_prefix_keys.size() != static_cast<size_t>(batch_size)) {
            initCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, initial_hash);
            return;
        }
        for (int row = 0; row < batch_size; ++row)
            batch_kv_cache_resource->setBatchCacheKeys(row, batch_kv_cache_resource->raw_prefix_keys[row]);
    }


    for (int i = 0; i < batch_size; ++i) {
        const auto& keys         = batch_kv_cache_resource->cacheKeys(i);
        const int   total_blocks = seq_len / seq_size_per_block;  // floor, only full blocks

        // If last_block_aligned was false previously, the last cache key corresponds to a partial block.
        // Drop it before we append new full-block cache keys.
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
            batch_kv_cache_resource->popBackCacheKey(i);
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? initial_hash : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int pos = index * seq_size_per_block;
            hash          = rtp_llm::hashInt64Array(hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block);
            const auto identity = complete_token_ids->multimodalCacheIdentity(pos, seq_size_per_block);
            if (!identity.empty()) {
                hash = hashInt64Array(hash, identity.data(), identity.data() + identity.size());
            }
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    // After incremental update we guarantee all existing keys are for full blocks.
    batch_kv_cache_resource->setLastBlockAligned(true);
    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->cacheResource(i).ensureLinearBlockDependencies();
    }
    batch_kv_cache_resource->markCacheKeysInitialized();
    bindMtpLookahead(batch_kv_cache_resource, complete_token_ids, seq_size_per_block);
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
