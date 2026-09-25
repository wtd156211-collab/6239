"""Sparse integer matrix engine: streaming SpMM and matvec, stdlib only.

Representation: CSR with array('q') row pointers / column indices and a
plain list of arbitrary-precision ints for values. A is streamed row by
row, B stays resident, C is accumulated one row at a time so transient
memory stays proportional to a single result row.
"""

from array import array

MAX_HEATMAP_BLOCKS = 262144
_HEATMAP_TARGET = 512
_WRITE_CHUNK = 65536


class SparseMatrix:
    __slots__ = ("nrows", "ncols", "rowptr", "colidx", "values")

    def __init__(self, nrows, ncols, rowptr, colidx, values):
        self.nrows = nrows
        self.ncols = ncols
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values

    @property
    def nnz(self):
        return len(self.values)


def _parse_header(line):
    rows, cols, nnz = (int(tok) for tok in line.split())
    return rows, cols, nnz


class MatrixReader:
    """Sequentially yields (row_index, cols, vals) for non-empty rows."""

    def __init__(self, path):
        self._f = open(path, "r", encoding="ascii", newline="\n")
        self.nrows, self.ncols, self.nnz = _parse_header(self._f.readline())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._f.close()

    def rows_iter(self):
        row = -1
        cols = []
        vals = []
        for line in self._f:
            r, c, v = line.split()
            r = int(r)
            if r != row:
                if vals:
                    yield row, cols, vals
                row = r
                cols = [int(c)]
                vals = [int(v)]
            else:
                cols.append(int(c))
                vals.append(int(v))
        if vals:
            yield row, cols, vals


def load_csr(path):
    """Load a matrix fully into CSR (used for the right operand B)."""
    with open(path, "r", encoding="ascii", newline="\n") as f:
        nrows, ncols, nnz = _parse_header(f.readline())
        rowptr = array("q", bytes(8 * (nrows + 1)))
        colidx = array("q")
        values = []
        idx = 0
        last_row = -1
        for line in f:
            r, c, v = line.split()
            r = int(r)
            if r != last_row:
                for rr in range(last_row + 1, r + 1):
                    rowptr[rr] = idx
                last_row = r
            colidx.append(int(c))
            values.append(int(v))
            idx += 1
        for rr in range(last_row + 1, nrows + 1):
            rowptr[rr] = idx
    return SparseMatrix(nrows, ncols, rowptr, colidx, values)


def multiply(a_path, b_path):
    """C = A * B. A is streamed row by row; B and C are held as CSR."""
    b = load_csr(b_path)
    with MatrixReader(a_path) as reader:
        if reader.ncols != b.nrows:
            raise ValueError(
                "inner dimensions differ: A has %d cols, B has %d rows"
                % (reader.ncols, b.nrows)
            )
        nrows = reader.nrows
        c_rowptr = array("q", bytes(8 * (nrows + 1)))
        c_colidx = array("q")
        c_values = []
        b_rowptr = b.rowptr
        b_colidx = b.colidx
        b_values = b.values
        count = 0
        last = 0
        for i, a_cols, a_vals in reader.rows_iter():
            while last < i:
                last += 1
                c_rowptr[last] = count
            acc = {}
            acc_get = acc.get
            for k, a_ik in zip(a_cols, a_vals):
                for p in range(b_rowptr[k], b_rowptr[k + 1]):
                    col = b_colidx[p]
                    acc[col] = acc_get(col, 0) + a_ik * b_values[p]
            for col, val in sorted(acc.items()):
                if val != 0:
                    c_colidx.append(col)
                    c_values.append(val)
                    count += 1
            last += 1
            c_rowptr[last] = count
        while last < nrows:
            last += 1
            c_rowptr[last] = count
    return SparseMatrix(nrows, b.ncols, c_rowptr, c_colidx, c_values)


def read_vector(path):
    with open(path, "r", encoding="ascii", newline="\n") as f:
        n = int(f.readline())
        vec = [int(line) for line in f]
    if len(vec) != n:
        raise ValueError("vector length mismatch: header %d, got %d" % (n, len(vec)))
    return vec


def matvec(mat_path, vec_path):
    """y = M * v, streaming M row by row."""
    vec = read_vector(vec_path)
    with MatrixReader(mat_path) as reader:
        if reader.ncols != len(vec):
            raise ValueError(
                "dimension mismatch: matrix has %d cols, vector has %d"
                % (reader.ncols, len(vec))
            )
        y = [0] * reader.nrows
        for i, cols, vals in reader.rows_iter():
            total = 0
            for c, v in zip(cols, vals):
                total += v * vec[c]
            y[i] = total
    return y


def _write_lines(f, line_iter):
    buf = []
    append = buf.append
    for line in line_iter:
        append(line)
        if len(buf) >= _WRITE_CHUNK:
            f.write("".join(buf))
            buf.clear()
    if buf:
        f.write("".join(buf))


def write_matrix(path, mat):
    def lines():
        for i in range(mat.nrows):
            for p in range(mat.rowptr[i], mat.rowptr[i + 1]):
                yield "%d %d %d\n" % (i, mat.colidx[p], mat.values[p])

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("%d %d %d\n" % (mat.nrows, mat.ncols, mat.nnz))
        _write_lines(f, lines())


def write_vector(path, y):
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("%d\n" % len(y))
        _write_lines(f, ("%d\n" % v for v in y))


def heatmap_grid(nrows, ncols):
    """Pick block geometry: at most _HEATMAP_TARGET blocks per axis."""
    block_height = max(1, -(-nrows // _HEATMAP_TARGET))
    block_width = max(1, -(-ncols // _HEATMAP_TARGET))
    block_rows = -(-nrows // block_height) if nrows else 0
    block_cols = -(-ncols // block_width) if ncols else 0
    assert block_rows * block_cols <= MAX_HEATMAP_BLOCKS
    return block_rows, block_cols, block_height, block_width


def compute_heatmap(mat):
    block_rows, block_cols, block_height, block_width = heatmap_grid(
        mat.nrows, mat.ncols
    )
    counts = {}
    if block_rows and block_cols:
        rowptr = mat.rowptr
        colidx = mat.colidx
        for i in range(mat.nrows):
            base = (i // block_height) * block_cols
            for p in range(rowptr[i], rowptr[i + 1]):
                key = base + colidx[p] // block_width
                counts[key] = counts.get(key, 0) + 1
    cells = [
        [key // block_cols, key % block_cols, cnt]
        for key, cnt in sorted(counts.items())
    ]
    return {
        "block_rows": block_rows,
        "block_cols": block_cols,
        "block_height": block_height,
        "block_width": block_width,
        "cells": cells,
    }


def page_data_json(mat, mul_sec, matvec_sec, heatmap):
    import json

    cells = json.dumps(heatmap["cells"], separators=(",", ":"))
    return (
        '{\n'
        '"shape": [%d, %d],\n'
        '"nnz": %d,\n'
        '"elapsed_sec": {"mul": %.3f, "matvec": %.3f, "total": %.3f},\n'
        '"heatmap": {"block_rows": %d, "block_cols": %d, '
        '"block_height": %d, "block_width": %d, "cells": %s}\n'
        '}\n'
    ) % (
        mat.nrows,
        mat.ncols,
        mat.nnz,
        mul_sec,
        matvec_sec,
        mul_sec + matvec_sec,
        heatmap["block_rows"],
        heatmap["block_cols"],
        heatmap["block_height"],
        heatmap["block_width"],
        cells,
    )
