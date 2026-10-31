"""CPU policy checks; no Torch import, model weights or CUDA context required."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "_mhc_split_policy", Path(__file__).resolve().parents[1] / "mhc_split.py"
)
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)


class MHCSplitPolicyTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"DSV4_MHC_SPLIT_MODE": "fixed", "DSV4_MHC_NUM_SPLITS": ""}
        )
        env.start()
        self.addCleanup(env.stop)

    def test_same_geometry_pins_split_across_reuse_and_decode_sizes(self):
        for tokens in (1, 32, 64, 90, 130, 192, 512, 513, 4096, 200000):
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    policy.mhc_prenorm_num_splits(20480, tokens=tokens, num_sms=152),
                    50,
                )
        self.assertEqual(policy.mhc_prenorm_num_splits(16384, num_sms=148), 49)

    def test_legacy_policy_is_explicit_and_does_not_stale_cache(self):
        self.assertEqual(
            policy.mhc_prenorm_num_splits(20480, tokens=90, num_sms=152), 50
        )
        with patch.dict(os.environ, {"DSV4_MHC_SPLIT_MODE": "token"}):
            self.assertEqual(
                policy.mhc_prenorm_num_splits(20480, tokens=90, num_sms=152), 76
            )
            self.assertEqual(
                policy.mhc_prenorm_num_splits(20480, tokens=130, num_sms=152), 50
            )
        self.assertEqual(
            policy.mhc_prenorm_num_splits(20480, tokens=90, num_sms=152), 50
        )

    def test_override_and_small_k_remain_valid(self):
        with patch.dict(os.environ, {"DSV4_MHC_NUM_SPLITS": "76"}):
            self.assertEqual(policy.mhc_prenorm_num_splits(20480, num_sms=152), 76)
            self.assertEqual(policy.mhc_prenorm_num_splits(64, num_sms=152), 1)
        self.assertEqual(policy.mhc_prenorm_num_splits(64, num_sms=1), 1)
        for override in ("0", "-1", "invalid"):
            with self.subTest(override=override), patch.dict(
                os.environ, {"DSV4_MHC_NUM_SPLITS": override}
            ):
                with self.assertRaises(ValueError):
                    policy.mhc_prenorm_num_splits(20480, num_sms=152)

    def test_device_sm_count_is_not_global_device_zero(self):
        with patch.object(
            policy, "_device_num_sms", side_effect=lambda index: (148, 152)[index]
        ) as sms:
            self.assertEqual(policy.mhc_prenorm_num_splits(20480, device=0), 49)
            self.assertEqual(policy.mhc_prenorm_num_splits(20480, device=1), 50)
        self.assertEqual([call.args[0] for call in sms.call_args_list], [0, 1])


if __name__ == "__main__":
    unittest.main()
