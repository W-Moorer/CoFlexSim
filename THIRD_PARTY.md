# 依赖与许可

仓库沿用 Apache-2.0。未包含 Abaqus、第三方作者 baseline 程序、商业数据库、服务器配置或凭据。外部依赖由使用者安装／构建时下载，保留上游许可：

| 依赖 | 用途 | 上游 |
|---|---|---|
| Eigen 3.4.0 | C++ 矩阵 | [Eigen](https://eigen.tuxfamily.org/)，MPL-2.0 等声明 |
| pybind11 | Python/C++ | [pybind11](https://github.com/pybind/pybind11)，BSD-3-Clause |
| CUDA Toolkit | 驱动、NVRTC、cuBLAS、图 | [NVIDIA](https://developer.nvidia.com/cuda-toolkit)，NVIDIA 许可 |
| CuPy | GPU 数组与初始化 | [CuPy](https://github.com/cupy/cupy)，MIT |
| NumPy / SciPy | 离线与参照 | [NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)，BSD |
| IPC Toolkit 1.4 | 历史公共模块几何依赖 | [IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit)，MIT |
| Numba / Rtree | CPU 辅助与空间索引 | [Numba](https://numba.pydata.org/)、[Rtree](https://rtree.readthedocs.io/) |
| Matplotlib | Python 图形 | [Matplotlib](https://matplotlib.org/)，上游许可 |

当前公开契约使用有限法向柔度接触，不用 IPC 障碍势；`ipctk` 依赖不代表本例采用 IPC 接触模型。字体由本机提供，仓库不分发 Times New Roman 等字体。

目前没有对应的正式论文 DOI。引用时注明版本和提交哈希；同模型后端加速不等同于对未运行的作者程序或商业产品的直接比较。

## 参考论文

Release 中的姚、田、米三篇原始 PDF 保留其作者／出版社版权及原文件声明，不纳入代码 Apache-2.0 授权范围。文献信息、DOI 和逐文件 SHA-256 见 [references/papers.json](references/papers.json)，引用格式见 [README](README.md#参考论文与引用)。
