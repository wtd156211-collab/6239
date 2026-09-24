"""核心实现。

口径（与 README 一致）：
- A 按行流式读入，不在内存里整存；B 以 CSR（行指针 + 列下标 + 值）常驻。
- 对 A 第 i 行的每个非零 (i, k)，把 B 第 k 行归并累加进第 i 行的累加器；
  累加器按列升序整理、丢掉相加为 0 的位置后按规范形写出。
- 整数全程任意精度，不用浮点；同一输入跑两遍输出逐字节相同。
- 任何时刻都没有与 行数 x 列数 成正比的结构；中间数据只与
  输入非零数、结果非零数成正比。
"""

import array

# 热力图每个方向最多 512 块：512 * 512 = 262144，满足 block_rows x block_cols 上限。
HEATMAP_MAX_BLOCKS = 512


def load_csr(path):
    """把规范形文本矩阵读成 CSR：返回 (行数, 列数, indptr, indices, values)。"""
    with open(path, "r", encoding="ascii") as fh:
        nrows, ncols, _ = map(int, fh.readline().split())
        indptr = array.array("q", [0]) * (nrows + 1)
        indices = array.array("q")
        values = []
        count = 0
        filled = 1
        for line in fh:
            row, col, val = line.split()
            row = int(row)
            while filled <= row:
                indptr[filled] = count
                filled += 1
            indices.append(int(col))
            values.append(int(val))
            count += 1
        while filled <= nrows:
            indptr[filled] = count
            filled += 1
    return nrows, ncols, indptr, indices, values


def _block_dim(size):
    """返回 (块数, 块长)：块数 <= HEATMAP_MAX_BLOCKS，空维度块数为 0。"""
    if size <= 0:
        return 0, 1
    step = -(-size // HEATMAP_MAX_BLOCKS)
    return -(-size // step), step


class Heatmap:
    """按块统计结果非零个数；键为 br * block_cols + bc，排序后天然按 (br, bc) 升序。"""

    def __init__(self, nrows, ncols):
        self.block_rows, self.block_height = _block_dim(nrows)
        self.block_cols, self.block_width = _block_dim(ncols)
        self.counts = {}

    def add(self, row, col):
        key = (row // self.block_height) * self.block_cols + col // self.block_width
        self.counts[key] = self.counts.get(key, 0) + 1

    def cells(self):
        return [
            [key // self.block_cols, key % self.block_cols, count]
            for key, count in sorted(self.counts.items())
        ]


class MultiplyResult:
    def __init__(self, nrows, ncols, nnz, heatmap):
        self.nrows = nrows
        self.ncols = ncols
        self.nnz = nnz
        self.heatmap = heatmap


def multiply(a_path, b_path, c_path, want_heatmap=False):
    """C = A x B，按规范形写出；返回 MultiplyResult。"""
    b_rows, b_cols, indptr, indices, values = load_csr(b_path)
    body = []
    nnz = 0
    with open(a_path, "r", encoding="ascii") as fa:
        a_rows, a_cols, _ = map(int, fa.readline().split())
        if a_cols != b_rows:
            raise ValueError(f"维度不匹配：A 是 {a_rows}x{a_cols}，B 是 {b_rows}x{b_cols}")
        heatmap = Heatmap(a_rows, b_cols) if want_heatmap else None
        acc = {}
        cur = -1

        def emit():
            nonlocal nnz
            parts = []
            for col, val in sorted(acc.items()):
                if val:
                    parts.append(f"{cur} {col} {val}\n")
                    nnz += 1
                    if heatmap is not None:
                        heatmap.add(cur, col)
            body.append("".join(parts))
            acc.clear()

        for line in fa:
            row, k, aval = line.split()
            row = int(row)
            if row != cur:
                if cur >= 0:
                    emit()
                cur = row
            k = int(k)
            aval = int(aval)
            get = acc.get
            for p in range(indptr[k], indptr[k + 1]):
                col = indices[p]
                acc[col] = get(col, 0) + aval * values[p]
        if cur >= 0:
            emit()
    with open(c_path, "w", encoding="ascii", newline="") as fc:
        fc.write(f"{a_rows} {b_cols} {nnz}\n")
        fc.write("".join(body))
    return MultiplyResult(a_rows, b_cols, nnz, heatmap)


def matvec(m_path, v_path, y_path):
    """y = M x v，按 4.2 格式写出（全零分量也照写）；返回向量长度。"""
    with open(v_path, "r", encoding="ascii") as fv:
        tokens = fv.read().split()
    n = int(tokens[0])
    vec = list(map(int, tokens[1 : 1 + n]))
    out = []
    with open(m_path, "r", encoding="ascii") as fm:
        rows, cols, _ = map(int, fm.readline().split())
        if cols != n:
            raise ValueError(f"维度不匹配：矩阵是 {rows}x{cols}，向量长 {n}")
        acc = 0
        cur = 0
        for line in fm:
            row, col, val = line.split()
            row = int(row)
            while cur < row:
                out.append(str(acc))
                acc = 0
                cur += 1
            acc += int(val) * vec[int(col)]
        while cur < rows:
            out.append(str(acc))
            acc = 0
            cur += 1
    with open(y_path, "w", encoding="ascii", newline="") as fy:
        fy.write(f"{rows}\n")
        if out:
            fy.write("\n".join(out))
            fy.write("\n")
    return rows
