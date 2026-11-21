# src/step13_shap_analysis.py
"""
Step 13 (论文主线正式版):
针对 Step 14 客观筛选出的 27 个非零特征与唯一胜出的源域最优基准模型
[Logistic Regression (C=0.01, penalty='l2', class_weight='balanced')]，
使用 shap.LinearExplainer 计算精确的 SHAP 特征贡献度、损伤风险方向（正/负相关），
并生成论文级 300-DPI 高清特征重要度条形图与 Beeswarm 蜂群图。
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures_final"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_step14_spec_and_soccermon():
    json_path = OUTPUTS_DIR / "step14_best_source_model_params.json"
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 {json_path}，请先运行 step14_soccermon_8models_benchmark.py")

    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    selected_features = spec["feature_audit"]["lasso_selected_features"]
    tuned_params = spec["tuned_hyperparameters"]
    target_col = spec["target_col"]

    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv")

    if "active_day" in df_sm.columns:
        df_sm = df_sm[df_sm["active_day"] == 1].copy()
    elif "has_monitoring" in df_sm.columns:
        df_sm = df_sm[df_sm["has_monitoring"] == 1].copy()
    else:
        mon_cols = [c for c in ["fatigue_01", "soreness_01", "uload_ewma_acwr", "dist_ewma_acwr"] if c in df_sm.columns]
        if mon_cols:
            df_sm = df_sm[df_sm[mon_cols].notna().any(axis=1)].copy()

    if "is_rehab" in df_sm.columns:
        df_sm = df_sm[df_sm["is_rehab"] == 0].copy()
    if "is_injured" in df_sm.columns:
        df_sm = df_sm[df_sm["is_injured"] == 0].copy()

    group_col = "player_id" if "player_id" in df_sm.columns else "athlete_id"
    df_sm = df_sm.dropna(subset=[target_col, group_col]).reset_index(drop=True)

    X = df_sm[selected_features].copy()
    y = df_sm[target_col].astype(int).values
    return X, y, selected_features, tuned_params, spec


def categorize_feature(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ["inj", "days_since"]):
        return "既往伤病史与恢复周期 (Injury History)"
    if any(k in c for k in ["sleep", "fatigue", "soreness", "stress", "mood", "wellness", "readiness"]):
        return "主观疲劳/睡眠/酸痛恢复 (Subjective Wellness)"
    if any(k in c for k in ["hsr", "impact", "decel", "accel", "imu", "sprint", "dist"]):
        return "外部穿戴跑动与冲击负荷 (External GPS/IMU Load)"
    return "内部训练负荷与急慢性比 (Internal sRPE/ACWR Load)"


def main():
    X, y, selected_features, tuned_params, spec = load_step14_spec_and_soccermon()

    print("=" * 105)
    print(f"🔍 Step 13: 正在对唯一源域最优基准模型 [{spec['best_model']}] 进行 SHAP 贡献度分析")
    print(f"   样本数 = {len(y)}, 阳性 = {y.sum()} ({y.mean()*100:.2f}%), 客观筛选特征数 = {len(selected_features)}")
    print(f"   最优超参数 = {tuned_params}")
    print("=" * 105)

    # 1. 在源域拟合完整预处理与最优 LR 基准模型
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_std = scaler.fit_transform(X_imp)
    X_std_df = pd.DataFrame(X_std, columns=selected_features)

    clf = LogisticRegression(
        C=tuned_params.get("C", 0.01),
        penalty=tuned_params.get("penalty", "l2"),
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X_std_df, y)

    # 2. 使用 LinearExplainer 计算精确的 SHAP 值（对数几率空间 log-odds）
    explainer = shap.LinearExplainer(clf, X_std_df)
    shap_values = explainer.shap_values(X_std_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values, dtype=float)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = np.sum(mean_abs_shap)
    shap_share_pct = (mean_abs_shap / (total_shap + 1e-12)) * 100.0
    lr_coefs = clf.coef_[0]

    ranking_df = pd.DataFrame({
        "Rank": range(1, len(selected_features) + 1),
        "Feature": selected_features,
        "Domain_Category": [categorize_feature(c) for c in selected_features],
        "Mean_Abs_SHAP": mean_abs_shap,
        "Contribution_Pct": shap_share_pct,
        "Standardized_LR_Beta": lr_coefs,
        "Odds_Ratio_per_SD": np.exp(lr_coefs),
        "Risk_Direction": [
            "↑ 正向促伤 (值越高风险越高)" if b > 0 else "↓ 负向保护 (值越高风险越低)"
            for b in lr_coefs
        ],
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)

    csv_path = OUTPUTS_DIR / "step13_shap_soccermon_best_lr_ranking.csv"
    ranking_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 汇总四大生理维度的累计贡献占比
    cat_summary = (
        ranking_df.groupby("Domain_Category", as_index=False)["Contribution_Pct"]
        .sum()
        .sort_values("Contribution_Pct", ascending=False)
    )

    print("\n📊 1. SoccerMon 最优源域基准模型 (LR, C=0.01) Top 20 核心损伤风险指标 SHAP 排名:")
    display_cols = ["Rank", "Feature", "Mean_Abs_SHAP", "Contribution_Pct", "Standardized_LR_Beta", "Odds_Ratio_per_SD", "Risk_Direction"]
    print(ranking_df[display_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n📊 2. 四大运动生理维度累计 SHAP 贡献占比:")
    for _, row in cat_summary.iterrows():
        print(f"   - {row['Domain_Category']:42s}: {row['Contribution_Pct']:.2f}%")

    # 3. 绘制论文级组合图 (左：Top 15 平均|SHAP|贡献占比与风险方向；右：SHAP Beeswarm 分布图)
    top_n = min(15, len(ranking_df))
    top_df = ranking_df.head(top_n).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)

    # 左图：带风险方向配色的水平条形图
    colors = ["#D55E00" if b > 0 else "#0072B2" for b in top_df["Standardized_LR_Beta"]]
    bars = axes[0].barh(top_df["Feature"], top_df["Contribution_Pct"], color=colors, alpha=0.88, edgecolor="black", height=0.68)
    for bar, pct, beta in zip(bars, top_df["Contribution_Pct"], top_df["Standardized_LR_Beta"]):
        sign_str = "(+)" if beta > 0 else "(-)"
        axes[0].text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% {sign_str}",
            va="center", ha="left", fontsize=9, fontweight="bold"
        )
    axes[0].set_xlim(0, max(top_df["Contribution_Pct"]) * 1.25)
    axes[0].set_xlabel("Relative SHAP Contribution (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Top 15 Injury Risk Predictors (Mean |SHAP| %)\n[Red (+): Risk Factor | Blue (-): Protective Factor]", fontsize=11.5, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    # 右图：四大生理维度贡献饼图/水平汇总条形图
    cat_rev = cat_summary.iloc[::-1]
    cat_colors = ["#CC79A7", "#009E73", "#E69F00", "#56B4E9"][:len(cat_rev)]
    bars2 = axes[1].barh(cat_rev["Domain_Category"], cat_rev["Contribution_Pct"], color=cat_colors, alpha=0.88, edgecolor="black", height=0.55)
    for bar, pct in zip(bars2, cat_rev["Contribution_Pct"]):
        axes[1].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=10, fontweight="bold"
        )
    axes[1].set_xlim(0, max(cat_rev["Contribution_Pct"]) * 1.25)
    axes[1].set_xlabel("Cumulative SHAP Share (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("(B) Cumulative SHAP Contribution by Physiological Domain", fontsize=11.5, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig5_shap_soccermon_best_lr.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 105)
    print(f"✅ Step 13 SHAP 分析完成！")
    print(f"   - 排名表已保存至: {csv_path}")
    print(f"   - 论文配图已保存至: {fig_path}")
    print("=" * 105)


if __name__ == "__main__":
    main()