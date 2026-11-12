# 结果与口径

`measurements.json` 保留重复计时、内存、全程／末段误差、3090 复验和不利结果。绝对路径已去除，来源文件名与哈希保留。`provenance.json` 对应 6 份完整响应 CSV：齿轮各 4,000 点，凸轮各 20,000 点。

CSV 的 `response`：齿轮为 `theta1 + (20/30) theta0`，单位 rad；凸轮为从动件参考坐标，单位 m。`contact_force_N` 为第一物体接触合力模长，`axial_force_N` 为 z 分量。两物体参考坐标保留于 `reference0`、`reference1`。

相对误差为 `||y_new − y_ref||₂ / max(||y_ref||₂, 1e−30)`。不做时间平移或幅值拟合。完整向量力误差来自原始记录，不能用模长差替代。这是后端一致性验证，不替代独立 FEM 工程验证。

v0.2.0 的新旧网格对照在 `mesh_update/`：两例均保留完整物理时间序列，以及全程和末段差异、独立在线时间、预计算及显存诊断。旧网格响应保留为 `previous.csv`，更新网格为 `updated.csv`；这些差异属于空间离散变化，不是后端数值偏差。

执行 `python tools/plot_mesh_update.py` 重画 v0.2.0 响应图；`plot_publication.py` 保留历史性能图的复现入口。图片使用 Python、英文与 Times New Roman，性能数字以表格列出。
