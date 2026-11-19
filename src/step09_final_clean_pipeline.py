# src/step09_final_clean_pipeline.py
"""
女足运动损伤风险跨数据集迁移与小样本验证 —— 终极零泄露审稿级管线 (Step 09)
已严格修正全部四项方法学细节：
1. 源域三专家采用源域按球员分组的 OOF (Out-of-Fold) 预测训练源域融合器 (杜绝源域 Stacking 过拟合)；
2. 零样本 (2a/2b) 的分类阈值 100% 在 SoccerMon 源域 OOF 上确定 (真正零本地信息)；
3. 跨运动员盲测 (3a/3b) 采用【训练折内部嵌套的内层 GroupKFold OOF】确定分类阈值；
   时间序列前滚 (4) 采用【截至隔离日之前的纯历史前滚 OOF】确定分类阈值；
4. 95% 置信区间采用【按运动员整人聚类重采样 (Player-Cluster Bootstrap)】与【7日时间块重采样 (Block Bootstrap)】。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def add_causal_ewma_and_context_features(df: pd.DataFrame, is_local: bool = False) -> pd.DataFrame:
    """严格因果特征工程：仅使用每位运动员截至当日或前一日的个人历史数据"""
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    pkey = df["player_id"]

    def by_p(s: pd.Series, fn):
        return s.groupby(pkey, sort=False, group_keys=False).transform(fn)

    new_cols = {}
    for col, prefix in [
        ("unified_daily_load", "uload"),
        ("total_distance_m", "dist"),
        ("hsr_distance_m", "hsr"),
        ("high_decel_sec", "decel"),
        ("imu_high_impact_sec", "impact"),
    ]:
        p_med = by_p(df[col], lambda s: s.shift(1).expanding(min_periods=1).median()).fillna(1.0)
        proxy = df["unified_daily_load"].fillna(0.0) * (p_med if col != "unified_daily_load" else 1.0)
        s_filled = df[col].where(df[col].notna(), proxy).fillna(0.0)

        ewma_acute = by_p(s_filled, lambda s: s.ewm(span=7, min_periods=1).mean())
        ewma_chronic = by_p(s_filled, lambda s: s.ewm(span=28, min_periods=3).mean())
        new_cols[f"{prefix}_ewma_atl7"] = ewma_acute
        new_cols[f"{prefix}_ewma_ctl28"] = ewma_chronic
        new_cols[f"{prefix}_ewma_acwr"] = (ewma_acute / (ewma_chronic + 1e-3)).clip(0.0, 4.0)
        new_cols[f"{prefix}_ewma_spike"] = (ewma_acute - ewma_chronic) / (ewma_chronic + 1e-3)

    new_cols["ewma_overload_composite"] = (
        0.35 * new_cols["uload_ewma_acwr"]
        + 0.25 * new_cols["dist_ewma_acwr"]
        + 0.20 * new_cols["hsr_ewma_acwr"]
        + 0.20 * new_cols["decel_ewma_acwr"]
    )
    new_cols["subj_strain_composite"] = (
        0.35 * df["wellness_poor_mean"].fillna(0.3)
        + 0.35 * (df["wellness_poor_mean_z"].fillna(0.0).clip(-3, 3) / 3.0)
        + 0.30 * df["soreness_01_roll7d"].fillna(0.3)
    )
    new_cols["recovery_deficit_index"] = new_cols["ewma_overload_composite"] * (
        1.0 + df["soreness_01"].fillna(0.2) + df["fatigue_01"].fillna(0.2) + df["stress_01"].fillna(0.2)
    )

    if is_local:
        new_cols["ctx_hist_inj_12m"] = df["hist_inj_12m"].fillna(0.0)
        new_cols["ctx_hist_major_inj"] = df["hist_major_inj"].fillna(0.0)
        new_cols["ctx_is_menstruation"] = df["is_menstruation"].fillna(0.0)
        new_cols["ctx_dysmenorrhea"] = (df["dysmenorrhea_score"].fillna(0.0) / 10.0).clip(0.0, 1.0)
        new_cols["ctx_mens_x_overload"] = new_cols["ctx_is_menstruation"] * new_cols["ewma_overload_composite"]
        new_cols["ctx_is_match_day"] = df["is_match_day"].fillna(0.0)
        new_cols["ctx_match_days_7d"] = by_p(
            new_cols["ctx_is_match_day"], lambda s: s.rolling(7, min_periods=1).sum()
        )
        new_cols["ctx_rpe_norm"] = (df["rpe"].fillna(3.0) / 10.0).clip(0.0, 1.0)
        new_cols["ctx_rpe_roll7d"] = by_p(new_cols["ctx_rpe_norm"], lambda s: s.rolling(7, min_periods=1).mean())
    else:
        for c in ["ctx_hist_inj_12m", "ctx_hist_major_inj", "ctx_is_menstruation",
                  "ctx_dysmenorrhea", "ctx_mens_x_overload", "ctx_is_match_day",
                  "ctx_match_days_7d", "ctx_rpe_norm", "ctx_rpe_roll7d"]:
            new_cols[c] = pd.Series(0.0, index=df.index)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()


FEAT_EXPERT_OVERLOAD = [
    "uload_ewma_acwr", "dist_ewma_acwr", "hsr_ewma_acwr",
    "uload_ewma_spike", "dist_ewma_spike",
    "uload_monotony7", "active_train_days_7d", "ewma_overload_composite",
]

FEAT_EXPERT_WELLNESS = [
    "soreness_01", "fatigue_01", "sleep_poor_01", "stress_01", "mood_poor_01",
    "wellness_poor_mean", "wellness_poor_mean_z", "soreness_01_z", "fatigue_01_z",
    "soreness_01_roll7d", "fatigue_01_roll7d", "stress_01_roll7d",
    "subj_strain_composite", "recovery_deficit_index",
]

FEAT_EXPERT_BIOMECH = [
    "decel_ewma_acwr", "decel_ewma_spike", "impact_ewma_acwr",
    "decel_accel_ratio", "hsr_ratio", "imu_dyn_p95_g_z",
    "high_decel_sec_z", "hsr_distance_m_z",
]

FEAT_LOCAL_FEMALE = [
    "ctx_match_days_7d", "ctx_is_match_day", "ctx_rpe_roll7d",
    "ctx_is_menstruation", "ctx_dysmenorrhea", "ctx_mens_x_overload",
]

FEAT_INJ_HISTORY = [
    "recent_inj_decay", "prev_inj_30d", "prev_inj_14d", "ctx_hist_inj_12m",
]


def select_threshold_from_oof(y_oof: np.ndarray, prob_oof: np.ndarray, min_alert: float = 0.15, max_alert: float = 0.35) -> float:
    """严格在折外预测 (OOF) 上选择最大化 F1 的分类阈值"""
    prec, rec, ths = precision_recall_curve(y_oof, prob_oof)
    if len(ths) == 0:
        return 0.5
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    q_low = np.quantile(prob_oof, 1.0 - max_alert)
    q_high = np.quantile(prob_oof, 1.0 - min_alert)
    valid_mask = (ths >= q_low) & (ths <= q_high)
    if valid_mask.any():
        best_idx = int(np.argmax(np.where(valid_mask, f1s, -1.0)))
        return float(ths[best_idx])
    return float(ths[int(np.argmax(f1s))])


class OOFSourceExpertBranch:
    """
    源域单分支专家：
    1) 通过 GroupKFold 在 SoccerMon 上生成严格的源域 OOF 预测分数，供源域二层融合器使用；
    2) 在 SoccerMon 全集上 fit 最终转换器与模型，对目标域进行零本地统计量的纯归纳预测。
    """
    def __init__(self, feat_cols: list[str], require_gps: bool = False):
        self.feat_cols = feat_cols
        self.require_gps = require_gps
        self.imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.lr = None
        self.gb = None
        self.oof_mean_ = 0.5
        self.oof_std_ = 0.1

    def _fit_single_fold(self, X_tr: np.ndarray, y_tr: np.ndarray):
        imp = SimpleImputer(strategy="median").fit(X_tr)
        sc = StandardScaler().fit(imp.transform(X_tr))
        X_tr_z = sc.transform(imp.transform(X_tr))

        lr = LogisticRegression(C=0.03, class_weight="balanced", random_state=42).fit(X_tr_z, y_tr)
        lr.coef_ = np.maximum(lr.coef_, 0.02)
        gb = HistGradientBoostingClassifier(
            max_iter=80, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=8.0, monotonic_cst=[1] * len(self.feat_cols),
            class_weight="balanced", random_state=42
        ).fit(X_tr_z, y_tr)
        return imp, sc, lr, gb

    def fit_and_get_source_oof(self, sm_df: pd.DataFrame, target_col: str) -> np.ndarray:
        """返回与 sm_df 同长度的源域 OOF 专家风险分 (0~1)"""
        gkf = GroupKFold(n_splits=5)
        raw_oof = np.full(len(sm_df), np.nan)

        if self.require_gps:
            valid_mask = sm_df["total_distance_m"].notna().values
            sub_df = sm_df[valid_mask].reset_index(drop=True)
        else:
            valid_mask = np.ones(len(sm_df), dtype=bool)
            sub_df = sm_df.reset_index(drop=True)

        X_sub = sub_df[self.feat_cols].values
        y_sub = sub_df[target_col].values
        g_sub = sub_df["player_id"].values
        sub_oof = np.zeros(len(sub_df))

        for tr_i, te_i in gkf.split(X_sub, y_sub, groups=g_sub):
            imp, sc, lr, gb = self._fit_single_fold(X_sub[tr_i], y_sub[tr_i])
            X_te_z = sc.transform(imp.transform(X_sub[te_i]))
            sub_oof[te_i] = 0.60 * lr.predict_proba(X_te_z)[:, 1] + 0.40 * gb.predict_proba(X_te_z)[:, 1]

        self.oof_mean_ = float(np.mean(sub_oof))
        self.oof_std_ = float(np.std(sub_oof) + 1e-6)

        # 对全量 sm_df 计算 OOF 映射值 (若 require_gps 且某行无 GPS，则在各折内用中位数 0.5 填充)
        z_sub = np.clip((sub_oof - self.oof_mean_) / self.oof_std_, -4.0, 4.0)
        raw_oof[valid_mask] = 1.0 / (1.0 + np.exp(-z_sub))
        raw_oof[~valid_mask] = 0.50

        # 最后在源域全集上 fit 归纳模型，用于对本地目标域做纯粹的 transform 预测
        self.imp, self.scaler, self.lr, self.gb = self._fit_single_fold(X_sub, y_sub)
        return raw_oof

    def predict_inductive_score(self, df_eval: pd.DataFrame) -> np.ndarray:
        X = self.imp.transform(df_eval[self.feat_cols].values)
        X_z = self.scaler.transform(X)
        raw = 0.60 * self.lr.predict_proba(X_z)[:, 1] + 0.40 * self.gb.predict_proba(X_z)[:, 1]
        z = np.clip((raw - self.oof_mean_) / self.oof_std_, -4.0, 4.0)
        return 1.0 / (1.0 + np.exp(-z))


class NestedOOFAdapter:
    """
    严格嵌套式的二阶段校准器：
    1) 缺失值填补、标准化与权重学习 100% 在当前训练折 (df_tr, y_tr) 内部完成；
    2) 分类阈值 (threshold_) 严格通过训练折内部的【内层 5-Fold GroupKFold OOF 预测】选定，
       彻底消除“训练内拟合预测选阈值偏乐观”的问题！
    """
    def __init__(self, include_history: bool = True):
        self.include_history = include_history
        self.imp_female = SimpleImputer(strategy="median")
        self.sc_female = StandardScaler()
        self.imp_hist = SimpleImputer(strategy="median")
        self.sc_hist = StandardScaler()
        self.pillar_scaler = StandardScaler()
        self.weights_ = None
        self.gb_ = None
        self.threshold_ = 0.5

    def _extract_raw_pillars(self, df_sub: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        p1 = df_sub["sm_expert_overload"].values
        p2 = df_sub["sm_expert_biomech"].values
        p3 = df_sub["sm_expert_wellness"].values

        X_fem = df_sub[FEAT_LOCAL_FEMALE].values
        if fit_mode:
            X_fem_z = self.sc_female.fit_transform(self.imp_female.fit_transform(X_fem))
        else:
            X_fem_z = self.sc_female.transform(self.imp_female.transform(X_fem))
        p4 = 1.0 / (1.0 + np.exp(-np.clip(X_fem_z @ np.array([0.35, 0.20, 0.20, 0.05, 0.05, 0.15]), -4, 4)))

        if not self.include_history:
            return np.column_stack([p1, p2, p3, p4])

        X_his = df_sub[FEAT_INJ_HISTORY].values
        if fit_mode:
            X_his_z = self.sc_hist.fit_transform(self.imp_hist.fit_transform(X_his))
        else:
            X_his_z = self.sc_hist.transform(self.imp_hist.transform(X_his))
        p5 = 1.0 / (1.0 + np.exp(-np.clip(X_his_z @ np.array([0.40, 0.25, 0.15, 0.20]), -4, 4)))
        return np.column_stack([p1, p2, p3, p4, p5])

    def _fit_core(self, df_tr: pd.DataFrame, y_tr: np.ndarray):
        P_raw_tr = self._extract_raw_pillars(df_tr, fit_mode=True)
        P_tr = self.pillar_scaler.fit_transform(P_raw_tr)
        P_tr_01 = 1.0 / (1.0 + np.exp(-np.clip(P_tr, -3.5, 3.5)))

        n_p = P_tr_01.shape[1]
        auc_gains = np.array([max(roc_auc_score(y_tr, P_tr_01[:, j]) - 0.50, 0.01) for j in range(n_p)])
        auc_w = auc_gains / (auc_gains.sum() + 1e-8)

        nnls_w, _ = nnls(P_tr_01, y_tr)
        nnls_w = nnls_w / (nnls_w.sum() + 1e-8) if nnls_w.sum() > 0 else auc_w

        w = np.clip(0.55 * auc_w + 0.45 * nnls_w, 0.10, 0.38)
        self.weights_ = w / w.sum()

        self.gb_ = HistGradientBoostingClassifier(
            max_iter=45, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=10.0, monotonic_cst=[1] * n_p,
            class_weight="balanced", random_state=42
        ).fit(P_tr_01, y_tr)

    def _predict_prob_only(self, df_te: pd.DataFrame) -> np.ndarray:
        P_raw_te = self._extract_raw_pillars(df_te, fit_mode=False)
        P_te = self.pillar_scaler.transform(P_raw_te)
        P_te_01 = 1.0 / (1.0 + np.exp(-np.clip(P_te, -3.5, 3.5)))
        return 0.70 * (P_te_01 @ self.weights_) + 0.30 * self.gb_.predict_proba(P_te_01)[:, 1]

    def fit_with_inner_group_oof_threshold(self, df_tr: pd.DataFrame, y_tr: np.ndarray):
        """通过训练折内部的嵌套 GroupKFold OOF 预测来选定分类阈值"""
        players_tr = df_tr["player_id"].values
        n_groups = len(np.unique(players_tr))
        inner_oof = np.zeros(len(df_tr))

        if n_groups >= 3:
            inner_gkf = GroupKFold(n_splits=min(5, n_groups))
            for i_tr, i_val in inner_gkf.split(df_tr, y_tr, groups=players_tr):
                if len(np.unique(y_tr[i_tr])) < 2:
                    continue
                sub_adapter = NestedOOFAdapter(include_history=self.include_history)
                sub_adapter._fit_core(df_tr.iloc[i_tr], y_tr[i_tr])
                inner_oof[i_val] = sub_adapter._predict_prob_only(df_tr.iloc[i_val])

        # 在全训练折上拟合最终参数
        self._fit_core(df_tr, y_tr)
        if np.any(inner_oof > 0):
            self.threshold_ = select_threshold_from_oof(y_tr, inner_oof)
        else:
            self.threshold_ = 0.50
        return self

    def predict(self, df_te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prob_te = self._predict_prob_only(df_te)
        pred_label_te = (prob_te >= self.threshold_).astype(int)
        return prob_te, pred_label_te


def cluster_bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, cluster_ids: np.ndarray, metric_fn, n_boot: int = 500, seed: int = 42
) -> str:
    """
    聚类 Bootstrap 95% 置信区间 (Cluster Bootstrap by Player or 7-Day Date Block)：
    整人或整时间块重采样，充分考虑同一运动员多日记录的自相关性。
    """
    val = metric_fn(y_true, y_prob)
    unique_clusters = np.unique(cluster_ids)
    n_c = len(unique_clusters)
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_c, replace=True)
        idx = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(metric_fn(y_true[idx], y_prob[idx]))
    low, high = np.percentile(boots, [2.5, 97.5])
    return f"{val:.4f} ({low:.3f}-{high:.3f})"


def format_row(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, clusters: np.ndarray) -> dict:
    return {
        "验证方案 (零泄露方法学闭环)": name,
        "ROC_AUC (Cluster 95% CI)": cluster_bootstrap_ci(y_true, y_prob, clusters, roc_auc_score),
        "AP (Cluster 95% CI)": cluster_bootstrap_ci(y_true, y_prob, clusters, average_precision_score),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "AP": round(average_precision_score(y_true, y_prob), 4),
        "F1 (嵌套OOF定阈值)": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
    }


def run_step09_evaluation(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    print("\n" + "=" * 108)
    print(f"🏆 终极零泄露评估 (Source OOF + Inner OOF Threshold + Cluster Bootstrap) —— [{title}]")
    print("=" * 108)

    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_sm_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_sm_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].reset_index(drop=True)
    y_sm = sm_clean[target_col].values

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    dates_loc = loc_active["date"]
    # 构造 7 日时间块 ID，供时间序列验证的 Block Bootstrap 使用
    date_blocks = ((dates_loc - dates_loc.min()).dt.days // 7).values

    # 1. 在 SoccerMon 上用 GroupKFold 获取三大专家的严格源域 OOF 分数，并拟合全量归纳专家
    exp_overload = OOFSourceExpertBranch(FEAT_EXPERT_OVERLOAD, require_gps=False)
    exp_biomech = OOFSourceExpertBranch(FEAT_EXPERT_BIOMECH, require_gps=True)
    exp_wellness = OOFSourceExpertBranch(FEAT_EXPERT_WELLNESS, require_gps=False)

    sm_clean["sm_expert_overload"] = exp_overload.fit_and_get_source_oof(sm_clean, target_col)
    sm_clean["sm_expert_biomech"] = exp_biomech.fit_and_get_source_oof(sm_clean, target_col)
    sm_clean["sm_expert_wellness"] = exp_wellness.fit_and_get_source_oof(sm_clean, target_col)

    loc_active["sm_expert_overload"] = exp_overload.predict_inductive_score(loc_active)
    loc_active["sm_expert_biomech"] = exp_biomech.predict_inductive_score(loc_active)
    loc_active["sm_expert_wellness"] = exp_wellness.predict_inductive_score(loc_active)

    # --- 方案 1: 传统未对齐 Baseline (默认 0.5 阈值) ---
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0).values, sm_active[target_col].values)
    prob_1_base = rf_base.predict_proba(loc_active[base_cols].fillna(0).values)[:, 1]
    pred_1_base = (prob_1_base >= 0.50).astype(int)

    # --- 方案 2a: 真正零本地信息的纯生理/穿戴/主观 Zero-Shot (阈值 100% 由 SoccerMon OOF 确定！) ---
    sm_oof_2a = (
        0.40 * sm_clean["sm_expert_overload"].values
        + 0.30 * sm_clean["sm_expert_biomech"].values
        + 0.30 * sm_clean["sm_expert_wellness"].values
    )
    th_2a_from_sm = select_threshold_from_oof(y_sm, sm_oof_2a)
    prob_2a_physio = (
        0.40 * loc_active["sm_expert_overload"].values
        + 0.30 * loc_active["sm_expert_biomech"].values
        + 0.30 * loc_active["sm_expert_wellness"].values
    )
    pred_2a_physio = (prob_2a_physio >= th_2a_from_sm).astype(int)

    # --- 方案 2b: 真正零本地信息的全维度源域 Zero-Shot (基于源域 OOF Stacking + 源域 OOF 阈值) ---
    zs_cols = ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness", "recent_inj_decay", "prev_inj_30d"]
    sm_gkf = GroupKFold(n_splits=5)
    sm_zs_oof = np.zeros(len(sm_clean))
    X_sm_raw = sm_clean[zs_cols].values
    g_sm = sm_clean["player_id"].values

    for s_tr, s_te in sm_gkf.split(X_sm_raw, y_sm, groups=g_sm):
        imp_s = SimpleImputer(strategy="median").fit(X_sm_raw[s_tr])
        sc_s = StandardScaler().fit(imp_s.transform(X_sm_raw[s_tr]))
        lr_s = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(
            sc_s.transform(imp_s.transform(X_sm_raw[s_tr])), y_sm[s_tr]
        )
        lr_s.coef_ = np.maximum(lr_s.coef_, 0.05)
        sm_zs_oof[s_te] = lr_s.predict_proba(sc_s.transform(imp_s.transform(X_sm_raw[s_te])))[:, 1]

    th_2b_from_sm = select_threshold_from_oof(y_sm, sm_zs_oof)

    # 拟合源域全量融合器并盲测本地数据
    zs_imp_full = SimpleImputer(strategy="median").fit(X_sm_raw)
    zs_sc_full = StandardScaler().fit(zs_imp_full.transform(X_sm_raw))
    lr_zs_full = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(
        zs_sc_full.transform(zs_imp_full.transform(X_sm_raw)), y_sm
    )
    lr_zs_full.coef_ = np.maximum(lr_zs_full.coef_, 0.05)

    prob_2b_zs = lr_zs_full.predict_proba(zs_sc_full.transform(zs_imp_full.transform(loc_active[zs_cols].values)))[:, 1]
    pred_2b_zs = (prob_2b_zs >= th_2b_from_sm).astype(int)

    # --- 方案 3a & 3b: 跨运动员盲测 GroupKFold (内层嵌套 GroupKFold OOF 定阈值) ---
    gkf = GroupKFold(n_splits=6)
    oof_prob_grp_nohist = np.zeros(len(loc_active))
    oof_pred_grp_nohist = np.zeros(len(loc_active), dtype=int)
    oof_prob_grp_full = np.zeros(len(loc_active))
    oof_pred_grp_full = np.zeros(len(loc_active), dtype=int)

    for tr_idx, te_idx in gkf.split(loc_active, y_loc, groups=players_loc):
        df_tr, y_tr = loc_active.iloc[tr_idx], y_loc[tr_idx]
        df_te = loc_active.iloc[te_idx]

        ad_nh = NestedOOFAdapter(include_history=False).fit_with_inner_group_oof_threshold(df_tr, y_tr)
        oof_prob_grp_nohist[te_idx], oof_pred_grp_nohist[te_idx] = ad_nh.predict(df_te)

        ad_full = NestedOOFAdapter(include_history=True).fit_with_inner_group_oof_threshold(df_tr, y_tr)
        oof_prob_grp_full[te_idx], oof_pred_grp_full[te_idx] = ad_full.predict(df_te)

    # --- 方案 4: 严格时间序列前滚 Purged Temporal CV (先验锚定平滑 + 纯历史前滚 OOF 定阈值) ---
    unique_dates = np.sort(dates_loc.unique())
    n_dates = len(unique_dates)
    oof_prob_temp = np.copy(prob_2b_zs)
    oof_pred_temp = np.copy(pred_2b_zs)
    warmup_cut = int(n_dates * 0.25)

    for d_idx in range(warmup_cut, n_dates):
        curr_date = unique_dates[d_idx]
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        te_mask = (dates_loc == curr_date).values

        n_hist = int(tr_mask.sum())
        if n_hist >= 120 and y_loc[tr_mask].sum() >= 8:
            ad_temp = NestedOOFAdapter(include_history=True)
            ad_temp._fit_core(loc_active.iloc[tr_mask], y_loc[tr_mask])
            p_local_te = ad_temp._predict_prob_only(loc_active.iloc[te_mask])

            # 先验锚定收缩权重 alpha: 随历史样本量从 0.15 平滑增长至 0.50，防止跨周尺度跳变
            alpha = float(np.clip((n_hist - 100) / 1000.0, 0.15, 0.50))
            oof_prob_temp[te_mask] = (1.0 - alpha) * prob_2b_zs[te_mask] + alpha * p_local_te

            # 阈值严格使用截至 purge_cutoff 之前已经产生的前滚历史 OOF 预测值选定！
            th_hist_oof = select_threshold_from_oof(y_loc[tr_mask], oof_prob_temp[tr_mask])
            oof_pred_temp[te_mask] = (oof_prob_temp[te_mask] >= th_hist_oof).astype(int)

    rows = [
        format_row("1. 传统直接迁移 Baseline (未对齐+默认0.5阈值)", y_loc, prob_1_base, pred_1_base, players_loc),
        format_row("2a. Zero-Shot 纯生理+穿戴+主观 (无伤病史/源域OOF定阈值)", y_loc, prob_2a_physio, pred_2a_physio, players_loc),
        format_row("2b. Zero-Shot 全维度源域迁移 (源域OOF融合+源域OOF定阈值)", y_loc, prob_2b_zs, pred_2b_zs, players_loc),
        format_row("3a. 跨运动员盲测 GroupKFold (无伤病史+内层嵌套OOF定阈值)", y_loc, oof_prob_grp_nohist, oof_pred_grp_nohist, players_loc),
        format_row("3b. 跨运动员盲测 GroupKFold (全维度5支柱+内层嵌套OOF定阈值)", y_loc, oof_prob_grp_full, oof_pred_grp_full, players_loc),
        format_row("4. 时间序列前滚 Purged Temporal (历史前滚OOF定阈值+Block CI)", y_loc, oof_prob_temp, oof_pred_temp, date_blocks),
    ]
    res_df = pd.DataFrame(rows)
    cols_show = [
        "验证方案 (零泄露方法学闭环)",
        "ROC_AUC (Cluster 95% CI)",
        "AP (Cluster 95% CI)",
        "F1 (嵌套OOF定阈值)",
        "Recall",
        "Precision",
    ]
    print(res_df[cols_show].to_string(index=False))
    return res_df


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    res_7d = run_step09_evaluation(
        df_sm, df_loc, "target_non_contact_next_7d", horizon_days=7, title="未来 7 天非接触性损伤预警 (阳性率 18.30%)"
    )
    res_3d = run_step09_evaluation(
        df_sm, df_loc, "target_non_contact_next_3d", horizon_days=3, title="未来 3 天非接触性损伤预警 (阳性率 10.27%)"
    )

    out_csv = OUTPUTS_DIR / "step09_bulletproof_paper_table.csv"
    pd.concat([
        res_7d.assign(预测窗口="未来7天非接触损伤"),
        res_3d.assign(预测窗口="未来3天非接触损伤"),
    ], ignore_index=True).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 终极零泄露论文主表已保存至: {out_csv}")
    print("🎉 步骤 09 完成！请把控制台结果发给我（也可以直接发给 ChatGPT 做终审验证）！")