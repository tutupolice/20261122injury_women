# 女足运动损伤风险评估项目 — PyCharm 工程框架与论文章节映射留档手册

> **适用项目**：`womens_soccer_injury_risk`
> **对应论文**：东北师范大学硕士学位论文《基于机器学习的女足运动损伤风险因素评估模型构建与验证研究》
> **核心原则**：全流程严格遵循 `shift(1)` 因果时序、按运动员分组防泄露（`StratifiedGroupKFold`）及折内预处理与校准，所有图表与数据均一一对应论文章节。

---

## 一、PyCharm 目录总览

```text
womens_soccer_injury_risk/
├── data/
│   ├── raw/
│   │   ├── soccermon/                 # 公开源域：SoccerMon（subjective 19 CSV + 1 JSON；客观 GPS 9,994 Parquet）
│   │   └── local_24/                  # 本队目标域：24名女足队员（00_原始客观数据 736 xlsx；01_原始主观数据；02_基线数据）
│   └── processed/                     # 清洗、联结与因果特征工程后的标准面板
│       ├── soccermon_daily_panel.csv  # 源域日级面板（19,681 活跃日，50 人）
│       ├── local_daily_panel.csv      # 本队日级面板（1,421 活跃日，24 人）
│       ├── soccermon_features.csv     # 源域时序特征表（125 列，含 114 个因果派生特征）
│       └── local_features.csv         # 本队时序特征表（133 列，含 120 个因果派生特征）
├── src/                               # 按 step01 -> step15 组织的模块化脚本
└── outputs/                           # 所有论文表格（CSV/JSON）与高清配图（300 DPI PNG）
    ├── figures_final/                 # 论文终稿配图目录（fig1 ~ fig7）
    └── *.csv / *.json                 # 各阶段对比表、参数文件与 SHAP 排名表
```

---

## 二、脚本模块（`src/stepXX`）功能与论文章节对应表

### 1. 数据清洗、域偏移诊断与特征工程（对应论文第 3 章、第 4 章 4.1–4.3 节）

| 脚本名称 | 核心作用与处理逻辑 | 关键产出 | 对应论文章节 |
| :--- | :--- | :--- | :--- |
| `step01` – `step01d` | 数据探查与跨队列域偏移诊断：定位 SoccerMon 100 Hz vs 本地 1 Hz、$g$ vs 12-bit ADC 原始整数、PmSys (1–5 正向) vs Hooper (1–7 反向)、重复文件与时间对齐问题 | 诊断报告（指导后续对齐规则） | 第 3 章 3.1–3.2 节、第 5 章 5.1 节 |
| `step02_build_local_dataset.py` | 清洗本地 24 人晨间/训练后问卷、Possiball GPS/IMU 与队医损伤登记表，按自然日联结 | `local_daily_panel.csv`（1,421 活跃日，7d 阳性 260 / 18.30%，3d 阳性 146 / 10.27%） | 第 3 章 3.2 节、第 5 章 5.1 节 |
| `step02b_build_soccermon_dataset.py` | 清洗 SoccerMon 主观报告与客观 GPS 数据，过滤未监控空窗期并按运动员-日联结 | `soccermon_daily_panel.csv`（19,681 活跃日，50 人） | 第 3 章 3.1–3.2 节、第 4 章 4.1–4.2 节 |
| `step03_feature_engineering.py` | 严格使用 `shift(1)` 历史窗口构造多时间窗均值/标准差、ACWR、Monotony、Strain、个体 Z-score 与伤病指数衰减特征 | `soccermon_features.csv`（114 派生特征）、`local_features.csv`（120 派生特征） | 第 3 章 3.3 节、第 4 章 4.3 节 |

### 2. 论文主线：源域 8 模型选优、SHAP 解释与冻结外部验证（对应第 4 章 4.4–4.9 节、第 5 章 5.2 节）

| 脚本名称 | 核心作用与处理逻辑 | 关键产出与核心结论 | 对应论文章节 |
| :--- | :--- | :--- | :--- |
| `step14_soccermon_8models_benchmark.py` | 在 SoccerMon 17,005 活跃监控日（7 天窗口，313 阳性，1.84%）上执行跨队列初筛（70 维）$\rightarrow$ Spearman 去共线（38 维）$\rightarrow$ LASSO 稀疏回归客观筛出 **27 维**；5 折 `StratifiedGroupKFold` 对比 **LR、GNB、RF、XGBoost、LightGBM、CatBoost、SVM-RBF、MLP**（AP 优先） | 选定唯一源域最优基准模型：**Logistic Regression (`C=0.01, penalty='l2'`)**，Pooled AP = **0.2874**（15.6×），ROC-AUC = **0.8286** | 第 4 章 4.3、4.5–4.6 节（表 4-5） |
| `step14b_next1d_benchmark.py` | 对比**次日新发（1 天窗口：`inj_all_onset`，61 阳性，0.359%）**下 8 模型表现 | 1 天窗口最高 AP 仅 **0.0350**（GNB），实证支撑采用“每日输出风险评分 + 7 天滚动窗口” | 第 4 章 4.4 节（窗口对比论证） |
| `step13_shap_analysis.py`（主线 LR 版） | 对最优 LR 基准模型（27 特征）计算精确 `LinearExplainer` SHAP 贡献度与风险方向 | 主观恢复 **67.14%**、内部负荷 **19.46%**、伤病史 **11.43%**、外部穿戴 **1.98%**；揭示自由线性模型存在共线符号补偿 | 第 4 章 4.8 节（表 4-10、图 4-4） |
| `step05_freeze_external_validation.py` | **100% 冻结**源域最优 LR 模型与 27 项特征，直接在本地 24 人上做 Zero-Shot 外部验证（7d 与 3d）及 Top 10%–30% 预警预算敏感性 | 7d ROC-AUC **0.5651**（Lift **1.55×**），3d ROC-AUC **0.5749**（Lift **1.80×**），验证跨队直接冻结的局限性 | 第 5 章 5.2 节（直接评分外部验证） |

### 3. 深化改进与严格无泄露验证套件（对应第 5 章 5.3 节）

| 脚本名称 | 核心作用与处理逻辑 | 关键产出与核心结论 | 对应论文章节 |
| :--- | :--- | :--- | :--- |
| `step06` – `step10` | 方法学演进与审稿修正式中间脚本（逐步修复 GPS 缺失误填 0、康复期污染、源域专家泄露与折叠交叉问题） | 历史迭代留档（论文正文不直接引用其旧数值） | 研究过程留档（不入正文主表） |
| `step11_locked_paper_suite.py` | 论文定稿级深化验证：交叉拟合三专家（Overload / 1Hz Biomech / Wellness）+ 生理单调双模型集成（LR + HistGBDT）+ 折内 Platt 校准 + `StratifiedGroupKFold`（5/5 有效折）+ Purged 时间前滚 + 内层 OOF 锁定 Top 10%–30% 预算阈值 + P01–P24 匿名折明细 | 7d 跨队单调迁移 ROC-AUC **0.7082**（AP **0.3924** / 2.14×），本地适配 **0.6991**，时间前滚 **0.6759**；3d 本地适配 **0.6999**（AP **0.2700** / 2.63×） | 第 5 章 5.3.1 节（主表、预算敏感性表、匿名折明细表） |
| `step12_final_figures.py` | 基于 Step 11 锁定管线绘制 4 张 300-DPI 论文高清组合图 | 生成 `fig1` ~ `fig4` | 第 5 章 5.3 节配图 |
| `step15_triage_three_tier_table.py` | 按论文式 (5-5) 计算近 14 天个人滚动风险百分位，检验红区（P90+）/ 黄区（P75–P89）/ 绿区（P0–P74）命中率 | 7d 红区阳性率 **18.12%**（Lift **1.25×**），证明纯个人对比会抹平伤病史基线差异，支撑“全队分位 + 个人轨迹”双轨复核 | 第 5 章 5.3.3 节（三级复核对照） |

---

## 三、已生成的输出文件与高清图表标注清单（`outputs/`）

### 1. 数据表与参数文件（`outputs/*.csv` & `*.json`）

| 文件名 | 内容说明 | 建议在论文中的用途 |
| :--- | :--- | :--- |
| `step14_soccermon_8models_cv.csv` | SoccerMon 7 天窗口下 8 大模型 5 折交叉验证完整指标表 | **第 4 章 表 4-5**（八模型源域对比） |
| `step14_best_source_model_params.json` | 特征筛选 5 步账本及唯一胜出模型 LR (`C=0.01, L2`) 的完整超参数 | **第 4 章 4.3 & 4.6 节**（特征数与模型参数表） |
| `step14b_next1d_8models_cv.csv` | SoccerMon 1 天次日窗口下 8 大模型表现对比表 | **第 4 章 4.4 节**（1 天 vs 7 天窗口对比依据） |
| `step13_shap_soccermon_best_lr_ranking.csv` | 源域最优 LR 模型 27 项特征的 LinearSHAP 贡献度、$\beta$ 系数与 $OR$ 值 | **第 4 章 表 4-10**（源域核心风险指标排序） |
| `step05_frozen_lr_external_validation.csv` | 冻结源域 LR 在本地 24 人（7d 与 3d）的 Zero-Shot 外部验证表 | **第 5 章 5.2 节**（冻结直接评分表现） |
| `step05_frozen_lr_budget_sensitivity.csv` | 冻结源域 LR 在本地 Top 10%–30% 预警预算下的敏感性表 | **第 5 章 5.2 节**（直接评分预警预算表现） |
| `step11_table1_main_results.csv` | 本地 24 人深化验证主表（未对齐基线、2a、2b、3c 分层适配、4 时间前滚） | **第 5 章 5.3.1 节**（改进后主性能对比表） |
| `step11_table2_budget_sensitivity.csv` | 折内 OOF 锁定阈值下 Top 10%–30% 目标预算与实际报警率对照表 | **第 5 章 5.3 节**（预警预算敏感性表） |
| `step11_table3_anonymized_folds.csv` | `P01–P24` 匿名运动员分组外层各折及时间前滚各期明细表 | **第 5 章 5.3 节 / 附录**（稳健性明细） |
| `shap_local_features_ranking.csv` | 本地适配全特征 SHAP 排名（`recent_inj_decay` 18.70%、`impact_ewma_acwr` 15.27% 等） | **第 5 章 5.3.2 节**（本地重要变量对照） |
| `step15_table5_7_three_tier_triage.csv` | 个人近 14 天滚动百分位红/黄/绿三级复核统计表（7d 与 3d） | **第 5 章 5.3.3 节**（双轨复核依据） |

### 2. 论文高清配图清单（`outputs/figures_final/*.png`，均为 300 DPI）

| 图片文件名 | 图表内容说明 | 建议论文图号与章节标注 |
| :--- | :--- | :--- |
| `fig5_shap_soccermon_best_lr.png` | **(A)** 源域最优 LR 模型 Top 15 特征平均 `|SHAP|` 占比与正/负风险方向；**(B)** 四大生理维度累计 SHAP 贡献 | **第 4 章 图 4-4**（4.8 节源域模型 SHAP 解释） |
| `fig1_roc_pr_validation.png` | 本地 24 人 7 天与 3 天任务下各验证路径（未对齐基线、2a、2b、3c、4）的 ROC 曲线与 PR 曲线对比 | **第 5 章 图 5-1**（5.3.1 节模型区分度对比） |
| `fig2_calibration_and_quintiles.png` | Platt 概率校准前后对比曲线及风险五分位（Quintile）实际损伤发生率单调递增图 | **第 5 章 图 5-2**（5.3.1 节概率校准与风险分层） |
| `fig3_alert_budget_and_folds.png` | Top 10%–30% 预警预算敏感性曲线及外层各折（`StratifiedGroupKFold` / 时间前滚）稳定性分布 | **第 5 章 图 5-3**（5.3.1 节预算敏感性与折间稳定性） |
| `fig4_anonymized_trajectories.png` | 匿名高风险多发队员 vs 低风险队员的每日风险评分轨迹与实际损伤发生时点对照 | **第 5 章 图 5-4**（5.3.3 节个例风险预警轨迹） |
| `fig6_shap_5_pillars_adapted.png` | 本地适配后两阶段五大支柱 SHAP 贡献占比（伤病史 39.01% + 赛程与生理期 29.43% + 源域三专家 31.56%） | **第 5 章 图 5-5**（5.3.2 节本地五大风险支柱归因） |
| `fig7_shap_local_top15_indicators.png` | 本地适配模型 Top 15 单项特征 SHAP 贡献排名 | **第 5 章 图 5-6**（5.3.2 节本地核心单项指标排名） |
