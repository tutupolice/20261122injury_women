# 女足运动损伤日级预测研究代码

本项目研究：使用女足运动员第 t 日监测结束时及此前的信息，预测次日起 7 个自然日内的自报告运动损伤；再检验职业女足队列开发的模型在本地高校女足队伍中的表现。代码供论文复核，不是已经验证可用于训练决策的工具。

`src/` 保留原路径，因为部分现行脚本直接导入其他脚本，运行回执也记录了这些路径。脚本编号与文件名中的 `final`、`locked` 不等于论文主线；以下顺序才是本次审查的主线。

## 研究设计与现有结果

| 阶段 | 研究对象与用途 | 已生成的结果 |
| --- | --- | --- |
| 源域开发与内部评价 | SoccerMon 50 人，按运动员分组比较八类模型 | 7 天有效风险人日 19,266，阳性窗口 264；LR 折外汇总 AP 0.2771、AUC 0.7851 |
| 冻结模型跨队评价 | 源域 LR 用于本地 24 人 | 本地有效风险人日 1,315，阳性窗口 225；直接映射 AUC/AP 0.5771/0.1897，无监督目标域标准化后 0.5813/0.2315 |
| 目标队伍模型更新 | 另一套源域专家集成和本地更新模型 | 2b 源域集成 AUC/AP 0.6895/0.3259；3c 本地更新 AUC/AP 0.6605/0.3061 |

AP 评价的是相关的**阳性风险人日**排序，不是独立伤病事件的检出率。源域 264 个阳性窗口围绕 63 个报告发作日形成；本地 225 个阳性窗口来自 64 条登记合并后的 61 个上报人日。两队都使用自报告运动损伤，但源域按上报间隔大于 3 天识别新发簇，本地按同日登记归并，结局判定和人群仍有差异。阶段 2 是跨队列迁移评价；阶段 3 使用本地结局更新另一个模型，不能把两者合称为同一次外部验证。

预测时点是第 t 日监测结束后，允许使用截至 t 日的特征；主结局为 `(t, t+7]`，补充结局为 `(t, t+3]`。评价样本须满足 `active_day == 1`、`is_rehab == 0`、`is_injured == 0`，且未来窗口有随访依据。本地特征监测截至 2026-06-18，独立伤病随访截至 2026-06-25。详细规则见[研究协议](config/study_protocol.json)。

## 私有环境中的执行顺序

在项目根目录安装依赖。版本是本机环境记录，其他平台尚未验证端到端复现。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

获授权的 SoccerMon 原始文件置于 `data/raw/soccermon/`，本地训练、问卷和伤病文件置于 `data/raw/local_24/`。公开仓库只有空目录占位符，不能单靠克隆仓库重算结果。在**私有环境**中从根目录按顺序运行：

```bash
python src/step02_build_local_dataset.py
python src/step02b_build_soccermon_dataset.py
python src/step02c_fix_calendar_and_targets.py
python src/step03_feature_engineering.py
python src/step14_soccermon_8models_benchmark.py
python src/step13_shap_analysis.py
python src/step05_freeze_external_validation.py
python src/step11_locked_paper_suite.py
python src/step12_final_figures.py
python src/step16_verify_and_receipt.py
```

`step09_final_clean_pipeline.py` 向 `step11`、`step12` 等提供函数，不需要在上述主线单独执行。`step16` 生成 `outputs/run_receipt.json`，记录指定主线代码、四个处理后面板、八张核心表、五张图的哈希及运行时 Git 状态。`all_artifacts_verified: true` 只说明清单中的文件存在，不证明方法正确、补充分析已登记，或**当前**工作区与回执生成时相同。

## 补充分析

主线完成后可分别运行：

```bash
python src/step17_journal_sensitivity.py
python src/step18_chapter4_split_and_timescale.py
python src/step19_local_shap_and_consistency.py
```

- `step17` 比较伤病史简单基线和本地结局聚类规则。本地 5 折伤病史基线 AP 0.3520，3c 更新模型 AP 0.3061；当前结果未证明多源模型优于该简单基线。
- `step18` 对照记录级与运动员不相交划分，并将源域 LR 特征按时间尺度归类。记录级划分 AUC 较高，但汇总 AP 较低，不能写成所有指标都升高。该脚本与 `step14` 的嵌套调参过程并不完全相同。
- `step19` 对**本地另行拟合的 38 特征 LR**做按运动员分组的折外归因，再与源域 LR 排序比较；它不是 `step11` 的 2b 或 3c 模型的 SHAP。源域归因基于全数据拟合，本地归因基于折外预测，比例和秩相关只宜作描述性比较，不能证明伤病机制或微调必要性。

`step17`—`step19` 及其输出尚未列入 `step16` 的主线回执。论文若使用这些结果，应另外固定版本并核验输入、代码和输出哈希。

## 给同行复核的关键提醒

源域 0.2771 是八模型比较过程的折外汇总 AP，不是获选后另一个独立测试集的分数；全源域解释模型保留 38 个特征，五个外层训练折分别保留 39、37、35、34、34 个。代码先按单变量 Spearman 绝对相关性排列特征，再用 `np.corrcoef` 做成对去共线，L1 筛选固定 `C=0.05`，论文方法应按实际实现描述。

`step05` 的冻结 LR 直接映射、目标域无监督标准化，以及 `step11` 的源域专家集成和本地更新是不同方案，不能合称为同一模型“微调后显著提升”。本地伤病史简单基线 AP 为 0.3520，3c 更新模型 AP 为 0.3061。源域 LR 的 Brier 为 0.1021，AP 排名第一不等于绝对风险概率校准良好。

`step13` 是全源域 LR 的 LinearSHAP；`step19` 是本地另行拟合的 38 特征 LR 的折外 LinearSHAP，不是 3c 更新模型的 SHAP。两队 38 项贡献秩相关约为 0.126、系数方向一致 18/38，只能作描述性比较，不能证明伤病机制或微调必要性。源域近 7 日睡眠不佳特征的系数为负，本地同名特征方向不同，不能写成稳定的致伤因素。直接迁移 AUC 0.5771 的区间包含 0.5，不宜写成统计意义上高于随机。

本地原始伤病表有 64 条登记；新版“功能影响”中“带痛坚持”为 29 条，另有 1 条仅写“轻度”，旧表的 30 条需在论文表格中重新核对。SHAP 和模型评分是预测归因，不证明干预可以减少损伤，也不等于已经形成可直接用于每日调量的工具。

## 目录、隐私与历史代码

- `src/`：正式主线、补充分析和必要的伤病复核工具；脚本职责见下表。
- `config/`：研究协议；`data/raw/`、`data/processed/`：获授权的私有数据。
- `outputs/`：主线与补充结果，其中部分文件含姓名、逐人记录或轨迹，**不能整体发送或上传**。
- 伦理、数据使用和隐私边界见下文，历史审查记录不再作为仓库文件保留。

`.gitignore` 默认排除原始数据、处理后数据和全部输出，请勿用 `git add -f` 绕过。当前没有开放代码许可证，也没有公开合成数据包，因此不应把私有工作目录直接标成“完全可复现的论文仓库”。与师哥共享时，先共享代码、协议、README 及人工检查后的汇总表；私人数据与逐人输出只在授权条件下传递。

## 脚本职责速查

| 顺序 | 脚本 | 作用 |
| --- | --- | --- |
| 1 | `step02_build_local_dataset.py` | 构建本地训练、问卷、GPS/IMU 日级面板 |
| 2 | `step02b_build_soccermon_dataset.py` | 构建 SoccerMon 日级面板 |
| 3 | `step02c_fix_calendar_and_targets.py` | 补连续自然日、显式标记风险集状态、建立严格未来标签 |
| 4 | `step03_feature_engineering.py` | 建立截至预测日可用的时序特征 |
| 5 | `step14_soccermon_8models_benchmark.py` | 源域八模型按运动员分组比较并选择 LR |
| 6 | `step13_shap_analysis.py` | 源域 LR 的 LinearSHAP 描述性解释 |
| 7 | `step05_freeze_external_validation.py` | 冻结源域 LR 的本地直接迁移评价 |
| 8 | `step11_locked_paper_suite.py` | 源域专家集成、目标队伍更新与分组评价 |
| 9 | `step12_final_figures.py` | 生成主线图件 |
| 10 | `step16_verify_and_receipt.py` | 生成主线运行回执和哈希 |
| 补充 | `step02d_export_local_adjudication.py` | 导出仅限私有环境的 64 条伤病复核清单 |
| 补充 | `step17_journal_sensitivity.py` | 伤病史基线和连续上报规则敏感性分析 |
| 补充 | `step18_chapter4_split_and_timescale.py` | 划分方式和特征时间尺度对照 |
| 补充 | `step19_local_shap_and_consistency.py` | 本地另拟合 LR 与源域 LR 的归因描述性比较 |

`step09_final_clean_pipeline.py` 是被 `step11`、`step12`、`step17` 调用的函数库，需要保留但不作为独立执行步骤。`step16` 当前不登记 `step17`–`step19`；如果论文引用补充结果，应先把这些脚本、输入和输出纳入新的固定回执。
