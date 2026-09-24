"""命令行入口：python -m sparsemul <mul|matvec|run> ..."""

import os
import sys
import time

from .core import matvec, multiply

USAGE = (
    "用法：\n"
    "  python -m sparsemul mul <A> <B> <C输出>\n"
    "  python -m sparsemul matvec <矩阵> <向量> <y输出>\n"
    "  python -m sparsemul run <A> <B> <v> <输出目录>"
)


def _cmd_mul(a_path, b_path, c_path):
    t0 = time.perf_counter()
    result = multiply(a_path, b_path, c_path)
    elapsed = time.perf_counter() - t0
    print(
        f"[mul] C: {result.nrows}x{result.ncols} nnz={result.nnz} "
        f"{elapsed:.3f}s -> {c_path}",
        file=sys.stderr,
    )
    return 0


def _cmd_matvec(m_path, v_path, y_path):
    t0 = time.perf_counter()
    n = matvec(m_path, v_path, y_path)
    elapsed = time.perf_counter() - t0
    print(f"[matvec] y: 长 {n} {elapsed:.3f}s -> {y_path}", file=sys.stderr)
    return 0


def _cmd_run(a_path, b_path, v_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    c_path = os.path.join(outdir, "c.txt")
    y_path = os.path.join(outdir, "y.txt")
    data_path = os.path.join(outdir, "page-data.json")

    t0 = time.perf_counter()
    result = multiply(a_path, b_path, c_path, want_heatmap=True)
    t1 = time.perf_counter()
    matvec(c_path, v_path, y_path)
    t2 = time.perf_counter()
    mul_sec = t1 - t0
    matvec_sec = t2 - t1
    total_sec = mul_sec + matvec_sec

    heatmap = result.heatmap
    cells = ",".join(f"[{br},{bc},{count}]" for br, bc, count in heatmap.cells())
    page_data = (
        f'{{"shape": [{result.nrows}, {result.ncols}], "nnz": {result.nnz}, '
        f'"elapsed_sec": {{"mul": {mul_sec:.3f}, "matvec": {matvec_sec:.3f}, '
        f'"total": {total_sec:.3f}}}, '
        f'"heatmap": {{"block_rows": {heatmap.block_rows}, '
        f'"block_cols": {heatmap.block_cols}, '
        f'"block_height": {heatmap.block_height}, '
        f'"block_width": {heatmap.block_width}, "cells": [{cells}]}}}}'
    )
    with open(data_path, "w", encoding="ascii", newline="") as fh:
        fh.write(page_data)
        fh.write("\n")

    print(
        f"[run] C: {result.nrows}x{result.ncols} nnz={result.nnz} "
        f"mul={mul_sec:.3f}s matvec={matvec_sec:.3f}s -> {outdir}",
        file=sys.stderr,
    )
    print(
        f'{{"mul_sec": {mul_sec:.3f}, "matvec_sec": {matvec_sec:.3f}, '
        f'"total_sec": {total_sec:.3f}}}'
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        if cmd == "mul" and len(args) == 3:
            return _cmd_mul(*args)
        if cmd == "matvec" and len(args) == 3:
            return _cmd_matvec(*args)
        if cmd == "run" and len(args) == 4:
            return _cmd_run(*args)
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
