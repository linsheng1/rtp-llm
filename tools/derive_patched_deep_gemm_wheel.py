#!/usr/bin/env python3
"""Derive a wheel from the fixed ARM341b656 archive; never compile DeepGEMM.

CPU/stdlib only. The original wheel and the existing, SHA-bound private-overlay
provenance are required inputs. Output manifest publication is the completion
marker; a wheel without that verified manifest is not a completed derivation.
No installation, network, model execution, or credential access is performed.
"""

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ORIGINAL_VERSION = "2.8.2+341b656.cu132"
VERSION = ORIGINAL_VERSION + ".rtp0002.swa0003"
TAG = "cp310-cp310-linux_aarch64"
BASE_COMMIT = "c1b4b19d82f56507f89be86dddf90497501d32a6"
EPILOGUE = "deep_gemm/include/deep_gemm/epilogue/sm100_dsv41_qkv.cuh"
SIDE = "deep_gemm/include/deep_gemm/impls/dsv41_qb/side_workers.cuh"
NORMAL = "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh"
NATIVE = "deep_gemm/_C.cpython-310-aarch64-linux-gnu.so"
PATCHES = [
    ("0002-normal-gemm-pdl-trigger.patch", "45678efe0b6440a8f5506d0f5ca9259c6a1d21b230d82972e6d970298135e4a1", {NORMAL}),
    ("0003-dsv41-swa-kv-numerical-boundaries.patch", "bdb0bc6257d848d34eba99cca55f202df26deadb2bf3cec77e44f86a25a7153d", {EPILOGUE, SIDE}),
]
CONTRACT = dict(
    SHA256="ccd592d716ae83ac4908fbbdc162bf93d731587acda86cc9d36bd51307d9d262",
    bytes=5134708, members=1020,
    native_SHA256="008f525801a17269a9b6c38b6ff78438824ff452515f825c7961b156443ee3df",
    postimages={EPILOGUE: "caa34df09f5ae20044da23c2a69a9cf3a0492ce59237505e90ee47b726cf5080",
                SIDE: "e114e14156eabfbc79370df44014ba8cae887f30f7d97b1459cdc8bb8560fd2e"},
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(data):
    return dict(bytes=len(data), SHA256=hashlib.sha256(data).hexdigest())


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def safe_name(name):
    parts = name.split("/")
    require(name and not name.startswith("/") and "\\" not in name and ":" not in name
            and all(p not in ("", ".", "..") for p in parts)
            and all(ord(c) >= 32 and ord(c) != 127 for c in name), "unsafe ZIP/RECORD name")
    require(str(PurePosixPath(name)) == name, "noncanonical member name")


def read_wheel(path, data=None):
    """Validate all archive members and their complete SHA256 RECORD coverage."""
    require(path.is_file() and not path.is_symlink(), "regular wheel required")
    require(path.stat().st_size <= 64 * 1024**2, "archive budget exceeded")
    files = {}
    if data is None:
        data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(len(entries) <= 2000 and sum(e.file_size for e in entries) <= 128 * 1024**2,
                "uncompressed/member budget exceeded")
        for entry in entries:
            safe_name(entry.filename)
            mode = entry.external_attr >> 16
            require(not entry.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG)
                    and not entry.flag_bits & 1, "directory/special/encrypted ZIP member")
            require(entry.filename not in files, "duplicate ZIP member")
            require(not entry.filename.endswith(("/RECORD.jws", "/RECORD.p7s")),
                    "signed RECORD cannot be reused by a derivative")
            files[entry.filename] = archive.read(entry)
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    require(len(records) == 1, "exactly one RECORD required")
    record = records[0]
    rows = list(csv.reader(io.StringIO(files[record].decode("utf-8"), newline="")))
    seen = set()
    for row in rows:
        require(len(row) == 3, "invalid RECORD row")
        name, digest, size = row
        safe_name(name)
        require(name not in seen and name in files, "duplicate/absent RECORD member")
        seen.add(name)
        if name == record:
            require(digest == size == "", "RECORD self fields must be empty")
        else:
            expected = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b"=").decode()
            require(digest == "sha256=" + expected and size == str(len(files[name])),
                    "RECORD hash/size mismatch: " + name)
    require(seen == set(files), "RECORD does not cover every archive member")
    return files


def metadata(files, version):
    dist = "deep_gemm-" + version + ".dist-info/"
    require({n.split("/", 1)[0] for n in files if ".dist-info/" in n}
            == {dist[:-1]}, "unexpected dist-info directory")
    require(all(n.startswith(("deep_gemm/", dist)) for n in files), "unexpected top-level package")
    message = BytesParser().parsebytes(files[dist + "METADATA"])
    require(message.get_all("Name") == ["deep_gemm"] and message.get_all("Version") == [version],
            "distribution Name/Version mismatch")
    wheel = BytesParser().parsebytes(files[dist + "WHEEL"])
    require(wheel.get_all("Wheel-Version") == ["1.0"] and wheel.get_all("Tag") == [TAG]
            and wheel.get_all("Root-Is-Purelib") == ["false"],
            "wheel ABI/platform tag mismatch")
    return dist, message


def patch_hunks(data):
    """Parse the two pinned unified patches, retaining every context byte."""
    changes = {}
    path = None
    current = None
    for line in data.splitlines(keepends=True):
        if line.startswith(b"diff --git "):
            path = None
            current = None
        elif line.startswith(b"+++ b/"):
            path = line[6:].decode().strip()
            safe_name(path)
        elif line.startswith(b"@@ "):
            require(path is not None, "hunk without path")
            match = re.match(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            require(match is not None, "invalid hunk header")
            current = [[], [], int(match[2] or b"1"), int(match[4] or b"1")]
            changes.setdefault(path, []).append(current)
        elif current is not None:
            require(line[:1] in (b" ", b"-", b"+"), "unsupported patch line")
            if line[:1] in (b" ", b"-"):
                current[0].append(line[1:])
            if line[:1] in (b" ", b"+"):
                current[1].append(line[1:])
    for hunks in changes.values():
        for old, new, old_count, new_count in hunks:
            require(len(old) == old_count and len(new) == new_count and old and new,
                    "hunk line count mismatch")
    return {name: [(b"".join(old), b"".join(new)) for old, new, _, _ in hunks]
            for name, hunks in changes.items()}


def patched(data, hunks, expected_sha):
    if pin(data)["SHA256"] == expected_sha:
        return data, "already_applied_exact_full_posthash"
    spans = []
    for old, new in hunks:
        require(data.count(old) == 1 and new not in data, "patch preimage missing/ambiguous/mixed")
        offset = data.index(old)
        spans.append((offset, offset + len(old), new))
    spans.sort()
    require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "overlapping patch hunks")
    result = data
    for start, end, new in reversed(spans):
        result = result[:start] + new + result[end:]
    require(pin(result)["SHA256"] == expected_sha, "full patch posthash differs")
    return result, "applied_exact_preimage"


def make_record(files, record):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(files.items()):
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record, "", ""])
    return output.getvalue().encode()


def archive_write(path, files):
    # ZIP_STORED fixes compression-library differences as well as timestamps.
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(item, data)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def derive(original, vendor_provenance, vendor_sha, patch_dir, output, source_commit,
           contract=CONTRACT, patches=PATCHES):
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "full product source commit required")
    require(original.is_file() and not original.is_symlink() and original.stat().st_size <= 64 * 1024**2,
            "bounded regular original wheel required")
    original_data = original.read_bytes()
    require(pin(original_data) == {k: contract[k] for k in ("bytes", "SHA256")}, "original wheel fingerprint differs")
    require(original.name == "deep_gemm-" + ORIGINAL_VERSION + "-" + TAG + ".whl", "original filename differs")
    files = read_wheel(original, original_data)
    require(len(files) == contract["members"], "original member count differs")
    old_dist, old_metadata = metadata(files, ORIGINAL_VERSION)
    require(pin(files[NATIVE])["SHA256"] == contract["native_SHA256"], "original native binary differs")
    require(vendor_provenance.stat().st_size <= 8 * 1024**2 and not vendor_provenance.is_symlink(), "vendor evidence budget/path")
    vendor_data = vendor_provenance.read_bytes()
    require(pin(vendor_data)["SHA256"] == vendor_sha, "vendor provenance SHA differs")
    vendor = json.loads(vendor_data)
    require(vendor["state"] == "CLUSTER1_PRIVATE_DEEP_GEMM_OVERLAY_DERIVED"
            and vendor["source_commit"] == BASE_COMMIT and vendor["native_rebuilt"] is False
            and vendor["original_source_unchanged"] is True and vendor["version"] == ORIGINAL_VERSION,
            "unexpected vendor derivation provenance")
    require(set(vendor["source_members"]) == set(vendor["destination_members"]), "vendor source/destination member set differs")
    vendor_modified = {name for name in vendor["source_members"]
                       if vendor["source_members"][name] != vendor["destination_members"][name]}
    require(vendor_modified == {"site-packages/" + EPILOGUE, "site-packages/" + SIDE},
            "unexpected vendor modification outside numerical headers")
    before = dict(files)
    # The tested Bazel vendor retains exactly one patch-0002 backup. Recreate
    # it from the original archive bytes, rather than ignoring extra members
    # or trusting unverified bytes from an installed directory.
    vendor_package = {n[len("site-packages/"):] for n in vendor["destination_members"]
                      if n.startswith("site-packages/deep_gemm/")}
    missing = vendor_package - {n for n in files if n.startswith("deep_gemm/")}
    added = {}
    if missing:
        backup = NORMAL + ".orig"
        require(missing == {backup}, "tested vendor member set differs: unknown addition")
        fingerprint = pin(before[NORMAL])
        require(vendor["source_members"]["site-packages/" + backup]
                == vendor["destination_members"]["site-packages/" + backup]
                == fingerprint, "patch backup fingerprint differs from original header")
        files[backup] = before[NORMAL]
        added[backup] = dict(source_original_member=NORMAL, **fingerprint)
    patch_receipts = []
    for index, (name, digest, targets) in enumerate(patches):
        patch_data = (patch_dir / name).read_bytes()
        require(pin(patch_data)["SHA256"] == digest, "patch SHA differs: " + name)
        changes = patch_hunks(patch_data)
        require(set(changes) == targets, "unexpected patch target set")
        source = vendor["source_members" if index == 0 else "destination_members"]
        states = {}
        for target, hunks in changes.items():
            expected = source["site-packages/" + target]["SHA256"]
            if target in contract["postimages"]:
                require(expected == contract["postimages"][target], "known tested header posthash differs")
            files[target], states[target] = patched(files[target], hunks, expected)
        patch_receipts.append(dict(name=name, SHA256=digest, targets=states))
    require(vendor["patch_SHA256"] == patches[1][1], "vendor numerical patch identity differs")
    # Compare every code/data member with the already tested private dependency,
    # not just the two patched headers. Bazel-generated dist-info is not reused.
    require({n for n in files if n.startswith("deep_gemm/")}
            == {n[len("site-packages/"):] for n in vendor["destination_members"]
                if n.startswith("site-packages/deep_gemm/")}, "tested vendor member set differs")
    for name, data in files.items():
        if name.startswith("deep_gemm/"):
            require(pin(data) == vendor["destination_members"]["site-packages/" + name],
                    "tested vendor code/data differs: " + name)
    allowed = set().union(*(p[2] for p in patches))
    modified = sorted(n for n in before if files[n] != before[n])
    require(set(modified) <= allowed, "unexpected original member modification")
    binaries = {n: pin(data) for n, data in files.items()
                if re.search(r"\.(?:so(?:\.[^/]+)?|a|cubin)$", n)}
    require(all(files[n] == before[n] for n in binaries), "native binary was modified")
    headers = {n: pin(data) for n, data in files.items() if n.startswith("deep_gemm/include/")}
    provenance = dict(schema="rtp-deep-gemm-wheel-derivation-v1", source_commit=source_commit,
                      source_commit_binding="explicit caller-supplied product identity",
                      original_wheel=pin(original_data), original_version=ORIGINAL_VERSION,
                      derived_version=VERSION, native_rebuilt=False, binaries=binaries, headers=headers,
                      original_headers={n: pin(before[n]) for n in headers if n in before},
                      added_members_from_original=added,
                      patches=patch_receipts, modified_original_members=modified,
                      vendor_provenance_SHA256=vendor_sha, vendor_source_commit=BASE_COMMIT,
                      tool_SHA256=pin(Path(__file__).read_bytes())["SHA256"],
                      RECORD_signatures="absent_required", runtime_module_version="upstream unchanged")
    new_dist = "deep_gemm-" + VERSION + ".dist-info/"
    files = {(new_dist + n[len(old_dist):] if n.startswith(old_dist) else n): data for n, data in files.items()}
    key = new_dist + "METADATA"
    old_version = ("Version: " + ORIGINAL_VERSION).encode()
    require(files[key].count(old_version + b"\n") == 1, "noncanonical Version header")
    files[key] = files[key].replace(old_version + b"\n", ("Version: " + VERSION + "\n").encode(), 1)
    _, new_metadata = metadata(files, VERSION)
    require(new_metadata.get_all("Requires-Dist") == old_metadata.get_all("Requires-Dist"), "Requires-Dist changed")
    files[new_dist + "rtp_derivation.json"] = canonical(provenance)
    record = new_dist + "RECORD"
    files[record] = make_record(files, record)
    output.mkdir(mode=0o700)  # Existing outputs, including incomplete attempts, are never overwritten.
    name = "deep_gemm-" + VERSION + "-" + TAG + ".whl"
    temporary = output / (name + ".partial")
    archive_write(temporary, files)
    verified = read_wheel(temporary)
    require(verified == files, "derived archive verification differs")
    metadata(verified, VERSION)
    manifest = dict(provenance, wheel=dict(filename=name, **pin(temporary.read_bytes())),
                    internal_provenance=pin(files[new_dist + "rtp_derivation.json"]),
                    state="DERIVED_WHEEL_VERIFIED_NOT_DEPLOYED", compression="ZIP_STORED")
    manifest_temp = output / "derivation-manifest.json.partial"
    with manifest_temp.open("xb") as stream:
        stream.write(canonical(manifest)); stream.flush(); os.fsync(stream.fileno())
    # Hard links publish only verified files atomically and fail on any collision.
    os.link(temporary, output / name)
    os.link(manifest_temp, output / "derivation-manifest.json")
    temporary.unlink(); manifest_temp.unlink()
    fd = os.open(str(output), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("original-wheel", "vendor-provenance", "patch-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vendor-provenance-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = derive(args.original_wheel, args.vendor_provenance, args.vendor_provenance_sha256,
                    args.patch_dir, args.output_dir, args.source_commit)
    print(json.dumps(dict(state=result["state"], wheel=result["wheel"], native_rebuilt=False), sort_keys=True))


if __name__ == "__main__":
    main()
