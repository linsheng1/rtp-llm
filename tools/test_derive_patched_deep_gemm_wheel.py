"""Small CPU contracts for the fixed DeepGEMM wheel derivation tool.

Fixtures contain inert synthetic binary bytes; no original ARM wheel, compiler,
CUDA, GPU, or network is needed. Real deployment input pins remain CLI constants.
"""

import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile


PATH = Path(__file__).with_name("derive_patched_deep_gemm_wheel.py")
SPEC = importlib.util.spec_from_file_location("derive_dg", PATH)
dg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dg)


class DeriveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_dir = self.root / "patches"
        self.patch_dir.mkdir()
        self.dist = "deep_gemm-" + dg.ORIGINAL_VERSION + ".dist-info/"
        self.data = {
            "deep_gemm/__init__.py": b"__version__ = '2.8.2'\n",
            dg.NATIVE: b"\x7fELF\0inert ARM fixture, not loadable\xff",
            dg.NORMAL: b"normal line\nbefore normal\nend\n",
            dg.EPILOGUE: b"epilogue line\nbefore epilogue\nend\n",
            dg.SIDE: b"side line\nbefore side\nend\n",
            "deep_gemm/include/keep.h": b"// untouched header\n",
            self.dist + "licenses/LICENSE": b"fixture license\n",
            self.dist + "METADATA": ("Metadata-Version: 2.1\nName: deep_gemm\nVersion: "
                + dg.ORIGINAL_VERSION + "\nRequires-Dist: torch>=2\n\nUntouched description\n").encode(),
            self.dist + "WHEEL": ("Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: false\nTag: "
                + dg.TAG + "\n").encode(),
        }
        self.patches = []
        self.expected = dict(self.data)
        for index, targets in enumerate([{dg.NORMAL}, {dg.EPILOGUE, dg.SIDE}], 2):
            patch = b""
            for target in sorted(targets):
                old = self.data[target].splitlines(keepends=True)
                new = old[1].replace(b"before", b"after")
                patch += ("diff --git a/{0} b/{0}\n--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n".format(target)).encode()
                patch += b" " + old[0] + b"-" + old[1] + b"+" + new + b" " + old[2]
                self.expected[target] = old[0] + new + old[2]
            name = "000%d.patch" % index
            (self.patch_dir / name).write_bytes(patch)
            self.patches.append((name, dg.pin(patch)["SHA256"], targets))
        self.original = self.root / ("deep_gemm-" + dg.ORIGINAL_VERSION + "-" + dg.TAG + ".whl")
        self.write_original(self.data)
        source = dict(self.data)
        source[dg.NORMAL] = self.expected[dg.NORMAL]
        self.vendor = dict(state="CLUSTER1_PRIVATE_DEEP_GEMM_OVERLAY_DERIVED", source_commit=dg.BASE_COMMIT,
            native_rebuilt=False, original_source_unchanged=True, version=dg.ORIGINAL_VERSION,
            patch_SHA256=self.patches[1][1],
            source_members={"site-packages/" + k: dg.pin(v) for k, v in source.items()},
            destination_members={"site-packages/" + k: dg.pin(v) for k, v in self.expected.items()})
        self.proof = self.root / "existing-overlay-provenance.json"

    def write_original(self, files):
        files = dict(files)
        record = self.dist + "RECORD"
        files[record] = dg.make_record(files, record)
        self.original.unlink(missing_ok=True)
        dg.archive_write(self.original, files)
        self.contract = dict(dg.pin(self.original.read_bytes()), members=len(files),
            native_SHA256=dg.pin(files[dg.NATIVE])["SHA256"],
            postimages={k: dg.pin(self.expected[k])["SHA256"] for k in (dg.EPILOGUE, dg.SIDE)})

    def run_derive(self, out="out"):
        self.proof.write_bytes(dg.canonical(self.vendor))
        return dg.derive(self.original, self.proof, dg.pin(self.proof.read_bytes())["SHA256"],
                         self.patch_dir, self.root / out, "1" * 40,
                         contract=self.contract, patches=self.patches)

    def test_deterministic_complete_record_binary_license_and_metadata_preserved(self):
        original = self.original.read_bytes()
        a = self.run_derive("a")
        b = self.run_derive("b")
        self.assertEqual(a, b)
        wheel_a = self.root / "a" / a["wheel"]["filename"]
        wheel_b = self.root / "b" / b["wheel"]["filename"]
        self.assertEqual(wheel_a.read_bytes(), wheel_b.read_bytes())
        self.assertEqual(self.original.read_bytes(), original)
        contents = dg.read_wheel(wheel_a)
        dist, metadata = dg.metadata(contents, dg.VERSION)
        self.assertEqual(metadata.get_all("Requires-Dist"), ["torch>=2"])
        self.assertEqual(contents[dist + "licenses/LICENSE"], self.data[self.dist + "licenses/LICENSE"])
        for path, data in self.expected.items():
            if path.startswith("deep_gemm/"):
                self.assertEqual(contents[path], data)
        self.assertFalse(a["native_rebuilt"])
        self.assertEqual(contents[dg.NATIVE], self.data[dg.NATIVE])
        internal = json.loads(contents[dist + "rtp_derivation.json"])
        self.assertNotIn("wheel", internal)  # No circular final-wheel fingerprint.
        self.assertEqual(a["wheel"]["SHA256"], dg.pin(wheel_a.read_bytes())["SHA256"])
        self.assertEqual(a["tool_SHA256"], dg.pin(PATH.read_bytes())["SHA256"])
        with self.assertRaises(FileExistsError):
            self.run_derive("a")

    def test_accepts_only_complete_known_posthash_for_already_applied_patch(self):
        data = dict(self.data)
        data[dg.NORMAL] = self.expected[dg.NORMAL]
        self.write_original(data)
        result = self.run_derive()
        self.assertEqual(result["patches"][0]["targets"][dg.NORMAL], "already_applied_exact_full_posthash")
        hunks = dg.patch_hunks((self.patch_dir / self.patches[0][0]).read_bytes())[dg.NORMAL]
        with self.assertRaisesRegex(ValueError, "preimage"):
            dg.patched(data[dg.NORMAL] + b"unknown tail\n", hunks, dg.pin(data[dg.NORMAL])["SHA256"])

    def test_rejects_broken_or_ambiguous_patch_preimage_even_with_valid_record(self):
        for replacement in [b"unexpected\n", self.data[dg.EPILOGUE] * 2]:
            with self.subTest(replacement=replacement):
                data = dict(self.data)
                data[dg.EPILOGUE] = replacement
                self.write_original(data)
                with self.assertRaisesRegex(ValueError, "preimage"):
                    self.run_derive()

    def test_rejects_changed_patch_proof_native_and_extra_vendor_member(self):
        with self.assertRaisesRegex(ValueError, "provenance SHA"):
            self.proof.write_bytes(dg.canonical(self.vendor))
            dg.derive(self.original, self.proof, "0" * 64, self.patch_dir, self.root / "out", "1" * 40,
                      contract=self.contract, patches=self.patches)
        patch = self.patch_dir / self.patches[0][0]
        old = patch.read_bytes()
        patch.write_bytes(old + b"\n")
        with self.assertRaisesRegex(ValueError, "patch SHA"):
            self.run_derive()
        patch.write_bytes(old)
        self.contract["native_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native binary"):
            self.run_derive()
        self.contract["native_SHA256"] = dg.pin(self.data[dg.NATIVE])["SHA256"]
        self.vendor["destination_members"]["site-packages/deep_gemm/extra.py"] = dg.pin(b"extra")
        with self.assertRaisesRegex(ValueError, "member set"):
            self.run_derive()

    def test_rejects_mixed_hunks_and_vendor_binary_change(self):
        with self.assertRaisesRegex(ValueError, "preimage"):
            dg.patched(b"after A\nbefore B\n", [(b"before A\n", b"after A\n"),
                       (b"before B\n", b"after B\n")], dg.pin(b"after A\nafter B\n")["SHA256"])
        self.vendor["destination_members"]["site-packages/" + dg.NATIVE] = dg.pin(b"changed native")
        with self.assertRaisesRegex(ValueError, "vendor modification"):
            self.run_derive()

    def test_record_detects_tampered_missing_size_and_duplicate_rows(self):
        valid = dg.read_wheel(self.original)
        record = self.dist + "RECORD"
        cases = []
        tampered = dict(valid); tampered[dg.NATIVE] += b"tampered"; cases.append(tampered)
        missing = dict(valid); missing[record] = b""; cases.append(missing)
        duplicate = dict(valid); duplicate[record] += duplicate[record].splitlines(keepends=True)[0]; cases.append(duplicate)
        size = dict(valid); size[record] = size[record].replace(b",259", b",260")
        # A guaranteed size-only alteration, independent of fixture byte length.
        lines = size[record].splitlines(); first = lines[0].rsplit(b",", 1)
        lines[0] = first[0] + b",99999"; size[record] = b"\n".join(lines) + b"\n"; cases.append(size)
        for index, value in enumerate(cases):
            with self.subTest(case=index):
                path = self.root / ("bad%d.whl" % index)
                dg.archive_write(path, value)
                with self.assertRaisesRegex(ValueError, "RECORD"):
                    dg.read_wheel(path)

    def test_rejects_traversal_duplicate_symlink_and_signatures(self):
        for index, name in enumerate(["../escape", "/absolute", "a\\b", "a//b", "a/./b", "a/../b", "C:drive"]):
            with self.subTest(name=name):
                path = self.root / ("unsafe%d.whl" % index)
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(name, b"bad")
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    dg.read_wheel(path)
        path = self.root / "duplicate.whl"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("deep_gemm/a", b"a"); archive.writestr("deep_gemm/a", b"b")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            dg.read_wheel(path)
        path = self.root / "symlink.whl"
        with zipfile.ZipFile(path, "w") as archive:
            entry = zipfile.ZipInfo("deep_gemm/link"); entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16; archive.writestr(entry, "target")
        with self.assertRaisesRegex(ValueError, "special"):
            dg.read_wheel(path)
        for name in ("RECORD.jws", "RECORD.p7s"):
            path = self.root / (name + ".whl")
            dg.archive_write(path, {self.dist + name: b"stale signature"})
            with self.assertRaisesRegex(ValueError, "signed RECORD"):
                dg.read_wheel(path)

    def test_pinned_real_patch_files_have_expected_sha_targets_and_valid_hunks(self):
        directory = PATH.parent.parent / "patches" / "deep_gemm"
        for name, digest, targets in dg.PATCHES:
            data = (directory / name).read_bytes()
            self.assertEqual(dg.pin(data)["SHA256"], digest)
            self.assertEqual(set(dg.patch_hunks(data)), targets)

    def test_rejects_wrong_distribution_version_and_abi_tag(self):
        cases = [("METADATA", b"Name: deep_gemm", b"Name: unrelated", "Name/Version"),
                 ("METADATA", dg.ORIGINAL_VERSION.encode(), b"9.9.9", "Name/Version"),
                 ("WHEEL", dg.TAG.encode(), b"cp310-cp310-linux_x86_64", "tag mismatch")]
        for member, old, new, error in cases:
            with self.subTest(member=member, replacement=new):
                data = dict(self.data)
                data[self.dist + member] = data[self.dist + member].replace(old, new)
                self.write_original(data)
                with self.assertRaisesRegex(ValueError, error):
                    self.run_derive()

    def test_real_cli_contract_cannot_accept_synthetic_archive(self):
        self.proof.write_bytes(dg.canonical(self.vendor))
        with self.assertRaisesRegex(ValueError, "original wheel fingerprint"):
            dg.derive(self.original, self.proof, dg.pin(self.proof.read_bytes())["SHA256"],
                      self.patch_dir, self.root / "out", "1" * 40)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
