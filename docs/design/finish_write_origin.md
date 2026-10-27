# FinishWrite source metadata

The existing `FinishWriteCacheRequest.locations` field may supply one complete
location for each successful write-session key, in success-mask order. The mask
indexes the newly allocated session keys, not the original StartWrite input.
Empty locations and nonempty locations without any `dtop_origin` parameter retain
the legacy status-only finish behavior. Such unmarked locations were historically
ignored, including SDK sentinel fields and opaque actual URIs. No proto or SDK
signature changes are required.

If any location has the marker, validate the whole vector before claiming the session:
instance, key/location cardinality, mask, storage type, exact spec names and size,
and the original allocated core URI must agree. The only permitted URI extension
is `dtop_origin`, identical across all specs of a location:

```
v1.<32-character write session>.<handoff_end>.<begin>.<end>.<computed_end>.<sampled_end>
```

The actual SDK's `GrpcStub::GenCacheLocation` sends `ST_UNSPECIFIED` and
`spec_size=-1`, because SDK `Locations` only has spec names/URIs. These sentinels
are resolved from the saved authoritative allocation; explicit real fields must
still agree. Invalid proto enum values or other negative sizes are rejected on
the marked route. Mixed marked/unmarked locations or specs are rejected rather
than silently dropping a source assertion.

Offsets are exclusive logical computed-KV positions. `begin < end <= computed_end`,
`handoff_end <= computed_end` and `sampled_end = computed_end + 1`. They describe an
ordinary autoregressive history, not arbitrary speculative decoding. CP copies do
not multiply these logical positions. Numeric values are bounded, canonical signed
64-bit nonnegative integers. This assertion does not authenticate arbitrary writers
or prove token identity beyond a checkpoint; consumers must still validate their
actual history, layout and complete spec/rank intersection.

StartWrite retains shared immutable WRITING locations until finish/timeout. A
private optional `allocation_session_id` field binds serialized metadata to that
allocation generation; old JSON without the field remains readable. It is not a
client-controlled proto field or a KV payload. Rejected input leaves the session
cleanup-capable. Expiry and validated Finish compete for a single claim.

Under the existing per-key metadata RMW fence, compare the exact WRITING value
including its allocation generation, then publish the complete replacement specs,
source and SERVING status as one immutable CacheLocation. The local backend merges
that pointer under its item mutex. Redis serializes the entire location to one
`L#<id>` field. There is no session-wide transaction: later backend batches may
fail after earlier valid locations become visible. Aggregate failure/unknown ACK
must not be counted as confirmed publication. A stale or missing target is an
error for Finish, even where generic cleanup CAS treats absence as idempotent.
After a claimed source finish reports a commit error, cleanup uses the existing
authoritative, exact-WRITING-value guarded reclaimer; already SERVING values or a
later allocation version must not be removed. Aggregate errors still do not count
as confirmed publication.

The first extension deliberately rejects core URI remapping and unknown parameters.
Matching SDK small-payload DRAM/SSD reader compatibility and full Manager RPC tests
are deployment gates; an HTTP marker roundtrip alone is insufficient. No GPU
metadata group, tensor CPU snapshot or new transfer route is introduced.
