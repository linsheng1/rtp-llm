import os
from typing import FrozenSet, Optional

# Interpolated between named groups by the internal endpoint matcher: must stay capture-free.
OSS_REGION_ID_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)+"


def get_oss_internal_region_allowlist() -> Optional[FrozenSet[str]]:
    # None selects the group-intranet rewrite; an empty set strips every -internal marker.
    raw = os.environ.get("OSS_INTERNAL_REGION_ALLOWLIST")
    if raw is None:
        return None
    return frozenset(region.strip().lower() for region in raw.split(",") if region.strip())
