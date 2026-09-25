"""CLI: python -m sparsemul {mul|matvec|run} ..."""

import os
import sys
import time

from . import core


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def cmd_mul(a_path, b_path, out_path):
    t0 = time.perf_counter()
    mat = core.multiply(a_path, b_path)
    core.write_matrix(out_path, mat)
    elapsed = time.perf_counter() - t0
    _log(
        "mul: %d x %d, nnz=%d, %.3f sec"
        % (mat.nrows, mat.ncols, mat.nnz, elapsed)
    )
    return mat, elapsed


def cmd_matvec(mat_path, vec_path, out_path):
    t0 = time.perf_counter()
    y = core.matvec(mat_path, vec_path)
    core.write_vector(out_path, y)
    elapsed = time.perf_counter() - t0
    _log("matvec: n=%d, %.3f sec" % (len(y), elapsed))
    return y, elapsed


def cmd_run(a_path, b_path, vec_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c_path = os.path.join(out_dir, "c.txt")
    y_path = os.path.join(out_dir, "y.txt")
    mat, mul_sec = cmd_mul(a_path, b_path, c_path)
    _, matvec_sec = cmd_matvec(c_path, vec_path, y_path)
    total_sec = mul_sec + matvec_sec
    heatmap = core.compute_heatmap(mat)
    with open(
        os.path.join(out_dir, "page-data.json"), "w", encoding="ascii", newline="\n"
    ) as f:
        f.write(core.page_data_json(mat, mul_sec, matvec_sec, heatmap))
    _log(
        '{"mul_sec": %.3f, "matvec_sec": %.3f, "total_sec": %.3f}'
        % (mul_sec, matvec_sec, total_sec)
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _log(__doc__.strip())
        return 2
    cmd, rest = args[0], args[1:]
    try:
        if cmd == "mul" and len(rest) == 3:
            cmd_mul(*rest)
        elif cmd == "matvec" and len(rest) == 3:
            cmd_matvec(*rest)
        elif cmd == "run" and len(rest) == 4:
            cmd_run(*rest)
        else:
            _log(__doc__.strip())
            return 2
    except (OSError, ValueError) as exc:
        _log("error: %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
