# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
import sys
import unittest
from unittest.mock import patch

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

import cuvslam_app


class _FakeFuture:
    def __init__(self, stat):
        self._stat = stat

    def result(self):
        return self._stat


class _FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, sequence, *args):
        del function, args
        return _FakeFuture(cuvslam_app.Stat(sequence_title=sequence["sequence_title"]))


class RunParallelTrackingTest(unittest.TestCase):
    def test_returns_stats_in_config_order_when_futures_finish_out_of_order(self):
        config = {
            "dataset_folder": "dataset",
            "sequence_cfgs": [
                {"enable": True, "sequence_title": "first"},
                {"enable": False, "sequence_title": "disabled"},
                {"enable": True, "sequence_title": "second"},
                {"enable": True, "sequence_title": "third"},
            ],
        }

        def reverse_completion_order(futures):
            return reversed(list(futures))

        with patch.object(cuvslam_app, "ProcessPoolExecutor", _FakeExecutor), \
                patch.object(cuvslam_app, "as_completed", reverse_completion_order):
            stats = cuvslam_app.run_parallel_tracking(config, object(), "/datasets", max_workers=2)

        self.assertEqual(["first", "second", "third"], [stat.sequence_title for stat in stats])


if __name__ == "__main__":
    unittest.main()
