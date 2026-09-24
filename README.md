# wt-031 sparsemul 稀疏矩阵运算（内存预算内，从 0 实现）

起始环境没有代码：`samples/` 是输入与期望输出，`web/` 空着。本文写清格式、口径、预算与验收。

## 1. 范围

做的：稀疏乘稀疏（4.1 进、出）、结果矩阵乘稠密向量（4.2）、`web/index.html` 热力图页面、标准库 `unittest` 测试
（`python -m unittest discover` 跑通、覆盖 `samples/`）；入口固定 `python -m sparsemul <子命令>`（4.3），页面固定
`web/index.html`。

不做：稠密矩阵输入；加减、转置、点乘、求逆、分解、幂；多进程/多线程/GPU/向量化；增量更新与缓存复用；第三方
库/CDN/外部资源、构建打包；图片文件；数据库与结果落库；输入容错（`samples/` 保证合法）。

## 2. 口径与公式

A 是 `Ra×Ka`、B 是 `Kb×Cb`（必须 `Kb == Ka`），C = A·B 是 `Ra×Cb`；v 长 `Cb`，y = C·v 长 `Ra`。公式一
`C[i][j] = Σk A[i][k]·B[k][j]`，公式二 `y[i] = Σj C[i][j]·v[j]`（v 长 = 列数）。整数全程任意精度，不许浮点中间
量、不许溢出。稀疏表示只存非零；若干贡献相加后为 0 的位置不写进结果、不算非零个数。同一输入跑两遍输出
逐字节相同。`mul`、`matvec` 的秒数从读入算到写出结束（含读写），墙钟时间。

## 3. 数据结构与处理口径

文本一律是**规范形**（输入输出共用）：行主序、行内列升序、同一坐标只出现一次、值非零；一种矩阵只有一种写法，
「输出逐字节相同」就等于「个数、位置、数值、顺序全对」。

处理：按行流式读入；对 A 第 `i` 行的每个非零 `(i, k)`，把 B 第 `k` 行按列归并累加到第 `i` 行的累加器；累加器按
列升序整理、丢掉相加为 0 的位置后按规范形写出；`y[i]` 是结果第 `i` 行与 v 的点乘。中间数据只允许与「输入非零
数与结果非零数之和」成正比。

## 4. 输入输出与文件格式

### 4.1 稀疏矩阵文本（`mat-*`、`expected-*.c.txt`）

首行 `<rows> <cols> <nnz>`，随后恰好 `nnz` 行 `<row> <col> <value>`：

- 纯 ASCII、单 `\n` 结尾（无 `\r`/BOM/空行/注释/行尾空格），分隔符是单个半角空格。
- `rows`、`cols`、`nnz ≥ 0`；`nnz` 等于条目行数，`rows` 或 `cols` 为 0 时 `nnz` 必须为 0。
- 下标从 0 起且在界内：`0 ≤ row < rows`、`0 ≤ col < cols`；`(row, col)` 严格递增，不重复。
- `value` 非零，写法 `-?(0|[1-9][0-9]*)`：无 `+`、无前导 0、位数不限。全零矩阵只写 `<rows> <cols> 0`。

### 4.2 稠密向量文本（`vec-*`、`expected-*.y.txt`）

首行 `<n>`，随后恰好 `n` 行，第 `i` 行是第 `i` 个分量；0 也照写、不能省行；整数写法同 4.1；`n = 0` 时只有一行
`0`。

### 4.3 命令行与产物

```
python -m sparsemul mul <A> <B> <C输出>          # 写 4.1 格式的 C
python -m sparsemul matvec <矩阵> <向量> <y输出>  # 写 4.2 格式的 y
python -m sparsemul run <A> <B> <v> <输出目录>     # 写 c.txt、y.txt、page-data.json
```

都在仓库根目录执行，输出目录自动创建，退出码 0 表示成功，日志走 stderr。`run` 那行 JSON 固定三字段、秒数三位
小数：`{"mul_sec": 0.412, "matvec_sec": 0.058, "total_sec": 0.470}`；输出目录用 `var/`，页面读 `var/page-data.json`。

### 4.4 页面数据 `var/page-data.json`

由 `run` 写出，页面只读不算，键固定：

- `shape`：结果 `[行数, 列数]`；`nnz`：结果非零个数（抵消不算）。
- `elapsed_sec`：`{"mul": …, "matvec": …, "total": …}`，两运算各自与合计的秒数。
- `heatmap`：`block_rows`、`block_cols`、`block_height`、`block_width`、`cells`；`cells` 每项 `[br, bc, count]`，
  按 `(br, bc)` 升序、只列 `count > 0` 的块。

`block_rows × block_cols ≤ 262144`；第 `(br, bc)` 块覆盖行区间 `[br×block_height, min((br+1)×block_height, 行数))`、
列区间同理；`nnz` 始终是整矩阵总数。

### 4.5 展示这一侧

- 打开：仓库根目录 `python -m http.server 8000`，浏览器开 `http://127.0.0.1:8000/web/index.html`；只用原生 HTML
  + Canvas + JavaScript（`file://` 打不开，页面要读 JSON）。
- 数据：先跑 `python -m sparsemul run … var/`，页面用相对路径 `fetch('../var/page-data.json')` 取回它，取不到就提示
  一行；页面里不重算矩阵。
- 必须体现：① 规模「行数 × 列数」；② 非零个数；③ 两运算各自与合计耗时（秒，三位小数）；④ 非零分布热力图
  （Canvas 按 `heatmap` 分块计数着色，非零越多越深、`count` 为 0 用底色）。

## 5. 性能与验收口径

### 5.1 内存预算

两个输入行列数各可达 `5e5`、非零数合计可达 `2e8`；除输入与结果的表示外，额外峰值分配 ≤ **256 MiB**，常驻表示
≤ `48 × (输入非零数 + 结果非零数) + 1 MiB`，任何时刻不得有与 `行数 × 列数` 成正比的结构。量法：`tracemalloc`
包住一次 `run` 取峰值，扣掉 `48 × (输入非零数 + 结果非零数)`。

### 5.2 时间预算

`samples/mat-medium.a.txt` × `samples/mat-medium.b.txt` 再加一步乘向量，两运算合计 ≤ **60 秒**（单进程、单机、
含读写）；真实规模不设硬时限，但耗时须与「输入非零数 + 结果非零数」同阶。

### 5.3 验收口径（逐条）

1. 环境：Python 3.13、只用标准库，无第三方依赖与构建步骤；非法输入报错、内部结构、日志都不查。
2. 正确性：`mul`、`matvec` 的输出与对应 `expected-*` 逐字节相同（空格、顺序、结尾换行都算）。
3. 确定性：同一输入跑两遍，输出 sha256 一致。
4. 内存与时间：额外开销 ≤ 256 MiB、没有与 `行数 × 列数` 成正比的缓冲；medium 两运算 ≤ 60 秒。
5. 页面与测试：按 4.5 能打开、数字与 JSON 一致（改 JSON 再刷新就跟着变）；`python -m unittest discover` 跑通。

## 6. 样例说明

每个样例五个文件：`mat-<case>.a.txt`（A）、`mat-<case>.b.txt`（B）、`vec-<case>.txt`（v）、
`expected-<case>.c.txt`（C = A·B）、`expected-<case>.y.txt`（y = C·v，行数 = C 的行数）。样例（`形状/非零个数`）：
`basic` 4×5/8、5×3/10 → 4×3/10，y 4/3；`zero-rows` 5×4/5、4×6/11 → 5×6/9，y 5/2；
`diagonal` 6×6/4、6×6/4 → 6×6/2，y 6/2；`expand` 64×64/128、64×64/252 → 64×64/4096，y 64/64；`empty` 5×7/7、
7×4/7 → 5×4/0，y 5/0；`bigint` 4×4/8、4×4/14 → 4×4/12，y 4/2；`single-row` 1×5000/40、5000×1/40 → 1×1/1，y 1/1；
`single-column` 5000×1/60、1×5000/60 → 5000×5000/3600，y 5000/0；`medium` 20000×20000/39993、
20000×20000/39995 → 20000×20000/79981，y 20000/20000。

覆盖点：`basic` 非方阵带负值、抵消位置不写；`zero-rows` 全零行/列；`diagonal` 纯对角有缺位；`expand` 380 个非零 →
满矩阵 4096（B 62 行不参与）；`empty` 两边不相交，C 写成一行 `5 4 0`、y 全零；`bigint` 37 位大数相消成空行；
`single-row`/`single-column` 极端形状（C 分别 1×1、5000×5000）；`medium` 按 5.2 的时限。

`samples/` 合计约 2.9 MiB。逐条核对（`basic`）：`python -m sparsemul mul samples/mat-basic.a.txt
samples/mat-basic.b.txt var/c-basic.txt`、`python -m sparsemul matvec samples/expected-basic.c.txt
samples/vec-basic.txt var/y-basic.txt`，再用 `Get-FileHash` 比对 `samples/expected-basic.*.txt`，一致即逐字节相同。

## 7. 待补的文档

1. 真实规模的输入与内存实测口径：上亿非零的数据放不进仓库，要现场自备；在哪一点取 `tracemalloc` 峰值、要不要连
   页面数据生成一起量，还没定稿。
2. 其它：客户机器规格、页面配色与刻度、结果装不下时要不要分片落盘、现场 Python 版本、退出码与错误信息格式
   都没定。
