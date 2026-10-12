# 历史实现归档

正式入口已切换为仓库根目录的 `quantrade` 命令，正式包为 `src/quantrade`。

- `legacy_v2/`：根目录原 ETF 系统、评价、因子和 Streamlit 页面。
- `legacy_signal/quant_system/`：原个股机器人、模型、数据工具与旧数据库。
- `legacy_research/`：原根目录 v1.1 研究脚本。
- 既有 `experiments/`、`old_versions/` 等目录保留原内容。

移动记录见 `v3_migration_manifest.json`，单文件保存移动前 SHA256。已有 v2.0.2c 未提交修改随文件原样移动，v2.2.0 的既有删除状态没有恢复。

历史库和关键根入口另有 `backups/v3-pre-refactor/` 副本。恢复时按迁移清单把文件复制到原位置，在独立目录恢复旧环境；不要覆盖 v3 工作区。旧脚本存在记账、时点、评价等已知缺陷，不作为正式运行入口。

历史测试在 `tests/legacy/`，需要 legacy 可选依赖；`python -m pytest tests/legacy -q` 验证保留行为。正式默认测试只运行 v3。
