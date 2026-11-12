# 结果与口径

`measurements.json` 保留重复计时、内存、全程／末段误差、3090 复验和不利结果。绝对路径已去除，来源文件名与哈希保留。`provenance.json` 对应 6 份完整响应 CSV：齿轮各 4,000 点，凸轮各 20,000 点。

CSV 的 `response`：齿轮为 `theta1 + (20/30) theta0`，单位 rad；凸轮为从动件参考坐标，单位 m。`contact_force_N` 为第一物体接触合力模长，`axial_force_N` 为 z 分量。两物体参考坐标保留于 `reference0`、`reference1`。

相对误差为 `||y_new − y_ref||₂ / max(||y_ref||₂, 1e−30)`。不做时间平移或幅值拟合。完整向量力误差来自原始记录，不能用模长差替代。这是后端一致性验证，不替代独立 FEM 工程验证。

执行 `python tools/plot_results.py` 重画 README 图。使用 Python、英文与 Times New Roman；SVG 保留可编辑文字，PNG 检查图输出到 `outputs/figure_qa/`。
