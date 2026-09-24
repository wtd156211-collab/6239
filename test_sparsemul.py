"""sparsemul 的验收测试：python -m unittest discover"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")

CASES = [
    "basic",
    "zero-rows",
    "diagonal",
    "expand",
    "empty",
    "bigint",
    "single-row",
    "single-column",
    "medium",
]


def run_cli(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "sparsemul", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"退出码 {proc.returncode}: {proc.stderr}"
    return proc


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


class MulTest(unittest.TestCase):
    def test_mul_matches_expected_byte_for_byte(self):
        for case in CASES:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "c.txt")
                run_cli(
                    "mul",
                    os.path.join(SAMPLES, f"mat-{case}.a.txt"),
                    os.path.join(SAMPLES, f"mat-{case}.b.txt"),
                    out,
                )
                self.assertEqual(
                    read_bytes(out),
                    read_bytes(os.path.join(SAMPLES, f"expected-{case}.c.txt")),
                )

    def test_mul_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for i in range(2):
                out = os.path.join(tmp, f"c{i}.txt")
                run_cli(
                    "mul",
                    os.path.join(SAMPLES, "mat-expand.a.txt"),
                    os.path.join(SAMPLES, "mat-expand.b.txt"),
                    out,
                )
                digests.append(hashlib.sha256(read_bytes(out)).hexdigest())
            self.assertEqual(digests[0], digests[1])


class MatvecTest(unittest.TestCase):
    def test_matvec_on_expected_c_matches_expected_y(self):
        for case in CASES:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "y.txt")
                run_cli(
                    "matvec",
                    os.path.join(SAMPLES, f"expected-{case}.c.txt"),
                    os.path.join(SAMPLES, f"vec-{case}.txt"),
                    out,
                )
                self.assertEqual(
                    read_bytes(out),
                    read_bytes(os.path.join(SAMPLES, f"expected-{case}.y.txt")),
                )

    def test_matvec_on_own_c_matches_expected_y(self):
        for case in CASES:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                c_path = os.path.join(tmp, "c.txt")
                y_path = os.path.join(tmp, "y.txt")
                run_cli(
                    "mul",
                    os.path.join(SAMPLES, f"mat-{case}.a.txt"),
                    os.path.join(SAMPLES, f"mat-{case}.b.txt"),
                    c_path,
                )
                run_cli(
                    "matvec",
                    c_path,
                    os.path.join(SAMPLES, f"vec-{case}.txt"),
                    y_path,
                )
                self.assertEqual(
                    read_bytes(y_path),
                    read_bytes(os.path.join(SAMPLES, f"expected-{case}.y.txt")),
                )


class RunTest(unittest.TestCase):
    def test_run_outputs_and_page_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                os.path.join(SAMPLES, "mat-basic.a.txt"),
                os.path.join(SAMPLES, "mat-basic.b.txt"),
                os.path.join(SAMPLES, "vec-basic.txt"),
                tmp,
            )
            # c.txt / y.txt 与期望输出逐字节一致
            self.assertEqual(
                read_bytes(os.path.join(tmp, "c.txt")),
                read_bytes(os.path.join(SAMPLES, "expected-basic.c.txt")),
            )
            self.assertEqual(
                read_bytes(os.path.join(tmp, "y.txt")),
                read_bytes(os.path.join(SAMPLES, "expected-basic.y.txt")),
            )
            # stdout 那行 JSON 固定三字段
            line = proc.stdout.strip().splitlines()[-1]
            summary = json.loads(line)
            self.assertEqual(
                sorted(summary), ["matvec_sec", "mul_sec", "total_sec"]
            )
            # page-data.json：键固定、数字与引擎结果一致
            with open(os.path.join(tmp, "page-data.json"), encoding="ascii") as fh:
                data = json.load(fh)
            self.assertEqual(sorted(data), ["elapsed_sec", "heatmap", "nnz", "shape"])
            self.assertEqual(data["shape"], [4, 3])
            self.assertEqual(data["nnz"], 10)
            self.assertEqual(sorted(data["elapsed_sec"]), ["matvec", "mul", "total"])
            hm = data["heatmap"]
            self.assertEqual(
                sorted(hm),
                ["block_cols", "block_height", "block_rows", "block_width", "cells"],
            )
            self.assertLessEqual(hm["block_rows"] * hm["block_cols"], 262144)
            keys = [(br, bc) for br, bc, _ in hm["cells"]]
            self.assertEqual(keys, sorted(keys))
            self.assertTrue(all(count > 0 for _, _, count in hm["cells"]))
            self.assertEqual(sum(count for _, _, count in hm["cells"]), data["nnz"])

class FormatTest(unittest.TestCase):
    def test_empty_result_is_single_header_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "c.txt")
            run_cli(
                "mul",
                os.path.join(SAMPLES, "mat-empty.a.txt"),
                os.path.join(SAMPLES, "mat-empty.b.txt"),
                out,
            )
            self.assertEqual(read_bytes(out), b"5 4 0\n")

    def test_bad_args_exit_nonzero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "sparsemul"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
