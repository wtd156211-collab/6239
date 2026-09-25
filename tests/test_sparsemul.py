import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        timeout=120,
    )
    assert proc.returncode == 0, "exit=%d stderr=%s" % (proc.returncode, proc.stderr)
    return proc


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


class MulMatvecTest(unittest.TestCase):
    def test_samples_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case):
                    a = os.path.join(SAMPLES, "mat-%s.a.txt" % case)
                    b = os.path.join(SAMPLES, "mat-%s.b.txt" % case)
                    v = os.path.join(SAMPLES, "vec-%s.txt" % case)
                    c_out = os.path.join(tmp, "c.txt")
                    y_out = os.path.join(tmp, "y.txt")
                    run_cli("mul", a, b, c_out)
                    self.assertEqual(
                        read_bytes(c_out),
                        read_bytes(os.path.join(SAMPLES, "expected-%s.c.txt" % case)),
                    )
                    run_cli("matvec", c_out, v, y_out)
                    self.assertEqual(
                        read_bytes(y_out),
                        read_bytes(os.path.join(SAMPLES, "expected-%s.y.txt" % case)),
                    )

    def test_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in ("basic", "expand", "medium"):
                with self.subTest(case=case):
                    digests = []
                    for n in range(2):
                        out = os.path.join(tmp, "c%d.txt" % n)
                        run_cli(
                            "mul",
                            os.path.join(SAMPLES, "mat-%s.a.txt" % case),
                            os.path.join(SAMPLES, "mat-%s.b.txt" % case),
                            out,
                        )
                        digests.append(hashlib.sha256(read_bytes(out)).hexdigest())
                    self.assertEqual(digests[0], digests[1])


class RunCommandTest(unittest.TestCase):
    def _run_case(self, case, tmp):
        out_dir = os.path.join(tmp, "out")
        proc = run_cli(
            "run",
            os.path.join(SAMPLES, "mat-%s.a.txt" % case),
            os.path.join(SAMPLES, "mat-%s.b.txt" % case),
            os.path.join(SAMPLES, "vec-%s.txt" % case),
            out_dir,
        )
        return out_dir, proc

    def test_run_outputs_match_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case):
                    out_dir, _ = self._run_case(case, tmp)
                    self.assertEqual(
                        read_bytes(os.path.join(out_dir, "c.txt")),
                        read_bytes(os.path.join(SAMPLES, "expected-%s.c.txt" % case)),
                    )
                    self.assertEqual(
                        read_bytes(os.path.join(out_dir, "y.txt")),
                        read_bytes(os.path.join(SAMPLES, "expected-%s.y.txt" % case)),
                    )

    def test_run_logs_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, proc = self._run_case("basic", tmp)
            m = re.search(
                r'^\{"mul_sec": \d+\.\d{3}, "matvec_sec": \d+\.\d{3}, '
                r'"total_sec": \d+\.\d{3}\}$',
                proc.stderr,
                re.MULTILINE,
            )
            self.assertIsNotNone(m, proc.stderr)

    def _check_page_data(self, case, out_dir):
        raw = read_bytes(os.path.join(out_dir, "page-data.json")).decode("ascii")
        data = json.loads(raw)
        self.assertEqual(
            set(data), {"shape", "nnz", "elapsed_sec", "heatmap"}
        )
        header = read_bytes(os.path.join(out_dir, "c.txt")).split(b"\n", 1)[0]
        rows, cols, nnz = (int(t) for t in header.split())
        self.assertEqual(data["shape"], [rows, cols])
        self.assertEqual(data["nnz"], nnz)

        elapsed = data["elapsed_sec"]
        self.assertEqual(set(elapsed), {"mul", "matvec", "total"})
        for key in ("mul", "matvec", "total"):
            self.assertRegex(raw, r'"%s": \d+\.\d{3}' % key)
        self.assertAlmostEqual(elapsed["mul"] + elapsed["matvec"], elapsed["total"], places=2)

        hm = data["heatmap"]
        self.assertEqual(
            set(hm), {"block_rows", "block_cols", "block_height", "block_width", "cells"}
        )
        self.assertLessEqual(hm["block_rows"] * hm["block_cols"], 262144)
        keys = [(c[0], c[1]) for c in hm["cells"]]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(all(c[2] > 0 for c in hm["cells"]))
        self.assertEqual(sum(c[2] for c in hm["cells"]), nnz)
        for br, bc, _ in hm["cells"]:
            self.assertTrue(0 <= br < hm["block_rows"])
            self.assertTrue(0 <= bc < hm["block_cols"])

    def test_page_data_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case):
                    out_dir, _ = self._run_case(case, tmp)
                    self._check_page_data(case, out_dir)


class HeatmapGeometryTest(unittest.TestCase):
    def test_block_bounds(self):
        from sparsemul.core import heatmap_grid

        for rows, cols in [
            (0, 0),
            (0, 5),
            (1, 1),
            (1, 5000),
            (5000, 1),
            (5000, 5000),
            (500000, 500000),
            (499999, 3),
        ]:
            with self.subTest(shape=(rows, cols)):
                br, bc, bh, bw = heatmap_grid(rows, cols)
                self.assertLessEqual(br * bc, 262144)
                self.assertGreaterEqual(bh, 1)
                self.assertGreaterEqual(bw, 1)
                if rows:
                    self.assertEqual((rows + bh - 1) // bh, br)
                if cols:
                    self.assertEqual((cols + bw - 1) // bw, bc)


if __name__ == "__main__":
    unittest.main()
