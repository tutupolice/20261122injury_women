# src/step15_triage_three_tier_table.py
"""
Step 15: 导出论文第 5.3.3 节 [表 5-7 个人风险百分位三级复核统计表]
- 基于本地 24 名女足队员 (1,421 活跃监控日)，采用 StratifiedGroupKFold (5折) 折外预测 + 折内 Platt 校准
- 严格按论文式 (5-5) 计算每位队员当日之前最近 14 个记录日 (不少于 7 个) 的个人风险百分位
- 统计红区 (P90-P100)、黄区 (P75-P89)、绿区 (P0-P74) 在未来 7 天与 3 天非接触性损伤上的命中率与富集倍数
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_local_active():
    df = pd.read_csv(PROCESSED_DIR / "local_features.csv")
    if "active_day" in df.columns:
        df = df[df["active_day"] == 1].copy()
    if "is_rehab" in df.columns:
        df = df[df["is_rehab"] == 0].copy()
    if "inj_all_onset" in df.columns and "is_injured" in df.columns:
        df = df[(df["is_injured"] == 0) | (df["inj_all_onset"] == 1)].copy()

    group_col = "player_id" if "player_id" in df.columns else "athlete_id"
    if "date" in df.columns:
        df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    # 匿名化编号 P01 ~ P24
    unique_players = sorted(df[group_col].astype(str).unique())
    anon_map = {pid: f"P{idx+1:02d}" for idx, pid in enumerate(unique_players)}
    df["anon_id"] = df[group_col].astype(str).map(anon_map)
    return df, group_col


def get_clean_features(df, group_col):
    ban_exact = {
        group_col, "anon_id", "date", "team", "session_id", "injury_id",
        "is_rehab", "is_injured", "active_day", "has_monitoring",
        "inj_all_onset", "inj_timeloss_onset", "inj_non_contact_onset",
        "injury_onset", "non_contact_onset",
    }
    cols = [
        c for c in df.columns
        if c not in ban_exact
        and not c.startswith(("target_", "future_", "next_", "label_"))
        and not c.endswith("_onset")
        and "timeloss" not in c
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def compute_oof_calibrated_probs(df, feature_cols, target_col, group_col):
    sub = df.dropna(subset=[target_col, group_col]).reset_index(drop=True)
    X = sub[feature_cols].copy()
    y = sub[target_col].astype(int).values
    groups = sub[group_col].astype(str).values

    cv = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y))

    for tr_idx, va_idx in cv.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X.iloc[tr_idx])
        X_va = imp.transform(X.iloc[va_idx])

        scl = StandardScaler()
        X_tr_s = scl.fit_transform(X_tr)
        X_va_s = scl.transform(X_va)

        # 双模型集成 (LR + 正则化 HistGBDT) + 折内 Platt 校准
        lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000, random_state=42)
        gb = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.03, max_depth=3,
            min_samples_leaf=30, l2_regularization=10.0,
            class_weight="balanced", random_state=42
        )
        lr.fit(X_tr_s, y[tr_idx])
        gb.fit(X_tr, y[tr_idx])

        raw_tr = 0.5 * lr.predict_proba(X_tr_s)[:, 1] + 0.5 * gb.predict_proba(X_tr)[:, 1]
        raw_va = 0.5 * lr.predict_proba(X_va_s)[:, 1] + 0.5 * gb.predict_proba(X_va)[:, 1]

        eps = 1e-6
        logit_tr = np.log(np.clip(raw_tr, eps, 1 - eps) / (1 - np.clip(raw_tr, eps, 1 - eps))).reshape(-1, 1)
        logit_va = np.log(np.clip(raw_va, eps, 1 - eps) / (1 - np.clip(raw_va, eps, 1 - eps))).reshape(-1, 1)

        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(logit_tr, y[tr_idx])
        oof_probs[va_idx] = platt.predict_proba(logit_va)[:, 1]

    sub["risk_prob"] = oof_probs
    return sub


def compute_personal_percentile(sub_df):
    """严格按论文式 (5-5) 计算个人近 14 个记录日 (不少于 7 个) 的风险百分位"""
    pctls = np.full(len(sub_df), np.nan)
    for _, grp in sub_df.groupby("anon_id", sort=False):
        idxs = grp.index.values
        probs = grp["risk_prob"].values
        for k in range(len(idxs)):
            start = max(0, k - 14)
            window = probs[start:k]
            if len(window) < 7:
                continue
            r_cur = probs[k]
            less = np.sum(window < r_cur)
            eq = np.sum(window == r_cur)
            pctls[idxs[k]] = 100.0 * (less + 0.5 * eq) / len(window)
    sub_df["personal_pctl"] = pctls
    return sub_df.dropna(subset=["personal_pctl"]).reset_index(drop=True)


def summarize_three_tiers(eval_df, target_col, task_name):
    y = eval_df[target_col].astype(int).values
    pctl = eval_df["personal_pctl"].values
    total_n = len(eval_df)
    total_pos = int(y.sum())
    base_rate = total_pos / max(total_n, 1)

    tiers = [
        ("红区，优先医学复核", "P90—P100", pctl >= 90.0),
        ("黄区，重点关注", "P75—P89", (pctl >= 75.0) & (pctl < 90.0)),
        ("绿区，常规监控", "P0—P74", pctl < 75.0),
        ("红区+黄区合计 (预警合集)", "P75—P100", pctl >= 75.0),
    ]

    rows = []
    for label, rng, mask in tiers:
        n_days = int(mask.sum())
        pos_days = int(y[mask].sum())
        pos_rate = pos_days / max(n_days, 1)
        rows.append({
            "预测任务": task_name,
            "等级": label,
            "百分位区间": rng,
            "运动员日": n_days,
            "占比(%)": round(n_days / total_n * 100, 1),
            "阳性日": pos_days,
            "占全部阳性日比例(%)": round(pos_days / max(total_pos, 1) * 100, 1),
            "区内阳性率(%)": round(pos_rate * 100, 2),
            "相对基线倍数(Lift)": f"{(pos_rate / max(base_rate, 1e-6)):.2f}x",
        })
    return pd.DataFrame(rows)


def main():
    df, group_col = load_local_active()
    feature_cols = get_clean_features(df, group_col)

    t7 = "target_non_contact_next_7d"
    t3 = "target_non_contact_next_3d"

    df7 = compute_oof_calibrated_probs(df, feature_cols, t7, group_col)
    eval7 = compute_personal_percentile(df7)
    tab7 = summarize_three_tiers(eval7, t7, "未来 7 天非接触性损伤")

    df3 = compute_oof_calibrated_probs(df, feature_cols, t3, group_col)
    eval3 = compute_personal_percentile(df3)
    tab3 = summarize_three_tiers(eval3, t3, "未来 3 天非接触性损伤")

    out_df = pd.concat([tab7, tab3], ignore_index=True)
    csv_path = OUTPUTS_DIR / "step15_table5_7_three_tier_triage.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 110)
    print("📊 [表 5-7] 个人风险百分位三级复核统计表 (本地 24 名女足队员，折外预测 + 14天个人滚动百分位):")
    print("=" * 110)
    print(out_df.to_string(index=False))

    # 筛选一个典型高风险匿名案例队员 (阳性日>=3 且红区命中多)
    p_stats = eval7.groupby("anon_id").agg(
        total_days=(t7, "count"),
        pos_days=(t7, "sum"),
        red_days=("personal_pctl", lambda s: int((s >= 90).sum())),
        yellow_days=("personal_pctl", lambda s: int(((s >= 75) & (s < 90)).sum())),
    ).sort_values(["pos_days", "red_days"], ascending=False)

    top_case = p_stats.index[0]
    row = p_stats.loc[top_case]
    print("\n" + "-" * 110)
    print(f"📌 典型高风险匿名个例推荐 ({top_case}): 有效评分日 {int(row['total_days'])} 天, "
          f"阳性日 {int(row['pos_days'])} 天, 红区 {int(row['red_days'])} 天, 黄区 {int(row['yellow_days'])} 天。")
    print(f"💾 结果已保存至: {csv_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()