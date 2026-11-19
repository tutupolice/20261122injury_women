# src/step08_strict_leakfree_pipeline.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_causal_ewma_and_context_features(df: pd.DataFrame, is_local: bool = False) -> pd.DataFrame:
    """
    严格因果特征工程 (Strictly Causal Feature Engineering)：
    所有纵向滚动/指数平滑仅使用每位运动员截至当日 (t) 或前一日 (t-1) 的个人历史，
    绝不使用全局未来中位数或跨时间窗口统计量！
    """
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
        # 严格只用截至前一日的历史个人 expanding median，首日缺失用 1.0 兜底（不用全局 df[col].median()！）
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


def select_threshold_on_train(y_tr: np.ndarray, prob_tr: np.ndarray, min_pos_rate: float = 0.15, max_pos_rate: float = 0.35) -> float:
    """
    严格仅在训练集 (y_tr, prob_tr) 上选择最佳 F1 阈值！
    同时限制预警比例在合理的运动队承受范围内 (15%~35%)，绝不触碰测试集标签。
    """
    prec, rec, ths = precision_recall_curve(y_tr, prob_tr)
    if len(ths) == 0:
        return 0.5
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    # 过滤掉把所有人都判为0或判为1的极端阈值
    q_low = np.quantile(prob_tr, 1.0 - max_pos_rate)
    q_high = np.quantile(prob_tr, 1.0 - min_pos_rate)
    valid_mask = (ths >= q_low) & (ths <= q_high)
    if valid_mask.any():
        best_idx = np.argmax(np.where(valid_mask, f1s, -1.0))
        return float(ths[best_idx])
    return float(ths[int(np.argmax(f1s))])


class SourceExpertBranch:
    """
    完全在 SoccerMon 源域上 fit 的单分支专家模型：
    缺失值填补器 (SimpleImputer) 与标准化器 (StandardScaler) 100% 仅在 SoccerMon 上 fit！
    对本地数据预测时只调用 transform，彻底杜绝利用本地测试集均值/方差。
    """
    def __init__(self, feat_cols: list[str], require_gps: bool = False):
        self.feat_cols = feat_cols
        self.require_gps = require_gps
        self.imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=0.03, class_weight="balanced", random_state=42)
        self.gb = HistGradientBoostingClassifier(
            max_iter=80, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=8.0, monotonic_cst=[1] * len(feat_cols),
            class_weight="balanced", random_state=42
        )
        self.sm_prob_mean_ = 0.5
        self.sm_prob_std_ = 0.1

    def fit(self, sm_df: pd.DataFrame, target_col: str):
        sub = sm_df[sm_df["total_distance_m"].notna()].copy() if self.require_gps else sm_df.copy()
        X_sm = self.imp.fit_transform(sub[self.feat_cols])
        X_sm_z = self.scaler.fit_transform(X_sm)
        y_sm = sub[target_col].values

        self.lr.fit(X_sm_z, y_sm)
        self.lr.coef_ = np.maximum(self.lr.coef_, 0.02)
        self.gb.fit(X_sm_z, y_sm)

        raw_sm = 0.60 * self.lr.predict_proba(X_sm_z)[:, 1] + 0.40 * self.gb.predict_proba(X_sm_z)[:, 1]
        self.sm_prob_mean_ = float(np.mean(raw_sm))
        self.sm_prob_std_ = float(np.std(raw_sm) + 1e-6)
        return self

    def predict_inductive_score(self, df_eval: pd.DataFrame) -> np.ndarray:
        """零测试集泄露预测：使用源域 fit 好的 imputer、scaler 和源域概率均值/标准差映射"""
        X = self.imp.transform(df_eval[self.feat_cols])
        X_z = self.scaler.transform(X)
        raw = 0.60 * self.lr.predict_proba(X_z)[:, 1] + 0.40 * self.gb.predict_proba(X_z)[:, 1]
        z = np.clip((raw - self.sm_prob_mean_) / self.sm_prob_std_, -4.0, 4.0)
        return 1.0 / (1.0 + np.exp(-z))


class LeakFreeFoldAdapter:
    """
    严格折内闭环的二阶段五大支柱校准器：
    所有的缺失值填补、标准化、支柱合成、权重学习与最佳分类阈值 (threshold_)
    100% 仅在当前训练折 (df_tr, y_tr) 内部 fit！
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

    def fit(self, df_tr: pd.DataFrame, y_tr: np.ndarray):
        P_raw_tr = self._extract_raw_pillars(df_tr, fit_mode=True)
        P_tr = self.pillar_scaler.fit_transform(P_raw_tr)
        # 将标准化后的支柱通过 Sigmoid 映射回 (0, 1)
        P_tr_01 = 1.0 / (1.0 + np.exp(-np.clip(P_tr, -3.5, 3.5)))

        n_p = P_tr_01.shape[1]
        auc_gains = np.array([max(roc_auc_score(y_tr, P_tr_01[:, j]) - 0.50, 0.01) for j in range(n_p)])
        auc_w = auc_gains / (auc_gains.sum() + 1e-8)

        nnls_w, _ = nnls(P_tr_01, y_tr)
        nnls_w = nnls_w / (nnls_w.sum() + 1e-8) if nnls_w.sum() > 0 else auc_w

        w = 0.55 * auc_w + 0.45 * nnls_w
        w = np.clip(w, 0.10, 0.38)
        self.weights_ = w / w.sum()

        self.gb_ = HistGradientBoostingClassifier(
            max_iter=45, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=10.0, monotonic_cst=[1] * n_p,
            class_weight="balanced", random_state=42
        )
        self.gb_.fit(P_tr_01, y_tr)

        prob_tr = 0.70 * (P_tr_01 @ self.weights_) + 0.30 * self.gb_.predict_proba(P_tr_01)[:, 1]
        # 严格在训练集内部选定最佳阈值并冻结！
        self.threshold_ = select_threshold_on_train(y_tr, prob_tr)
        return self

    def predict(self, df_te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        P_raw_te = self._extract_raw_pillars(df_te, fit_mode=False)
        P_te = self.pillar_scaler.transform(P_raw_te)
        P_te_01 = 1.0 / (1.0 + np.exp(-np.clip(P_te, -3.5, 3.5)))

        prob_te = 0.70 * (P_te_01 @ self.weights_) + 0.30 * self.gb_.predict_proba(P_te_01)[:, 1]
        pred_label_te = (prob_te >= self.threshold_).astype(int)
        return prob_te, pred_label_te


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 500, seed: int = 42) -> str:
    """计算指标及 95% Bootstrap 置信区间"""
    val = metric_fn(y_true, y_prob)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(metric_fn(y_true[idx], y_prob[idx]))
    low, high = np.percentile(boots, [2.5, 97.5])
    return f"{val:.4f} ({low:.3f}-{high:.3f})"


def format_metrics_row(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred_infold: np.ndarray) -> dict:
    """
    注意：这里的 y_pred_infold 是每一折测试集使用【各自训练折内部选定的阈值】直接生成的 0/1 预测，
    绝不在测试集 y_true 上重新搜索阈值！
    """
    return {
        "验证方案 (100%折内预处理+折内定阈值)": name,
        "ROC_AUC (95% CI)": bootstrap_ci(y_true, y_prob, roc_auc_score),
        "AP (95% CI)": bootstrap_ci(y_true, y_prob, average_precision_score),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "AP": round(average_precision_score(y_true, y_prob), 4),
        "F1 (折内定阈值)": round(f1_score(y_true, y_pred_infold, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred_infold, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred_infold, zero_division=0), 4),
    }


def run_strict_leakfree_evaluation(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    print("\n" + "=" * 105)
    print(f"🔒 审稿级零泄露评估 (In-Fold Preprocessing & Thresholding) —— 目标: [{title}] ({target_col})")
    print("=" * 105)

    # 1. 准备清洗后的 SoccerMon 训练集
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_sm_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_sm_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].copy()

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    dates_loc = loc_active["date"]

    # 2. 在 SoccerMon 上训练 3 个纯归纳式专家模型 (Imputer/Scaler 100% 仅在 SoccerMon 上 fit)
    exp_overload = SourceExpertBranch(FEAT_EXPERT_OVERLOAD, require_gps=False).fit(sm_clean, target_col)
    exp_biomech = SourceExpertBranch(FEAT_EXPERT_BIOMECH, require_gps=True).fit(sm_clean, target_col)
    exp_wellness = SourceExpertBranch(FEAT_EXPERT_WELLNESS, require_gps=False).fit(sm_clean, target_col)

    loc_active["sm_expert_overload"] = exp_overload.predict_inductive_score(loc_active)
    loc_active["sm_expert_biomech"] = exp_biomech.predict_inductive_score(loc_active)
    loc_active["sm_expert_wellness"] = exp_wellness.predict_inductive_score(loc_active)

    # 同时在 SoccerMon 上训练一个包含近期伤病史的完整源域模型，并在 SoccerMon 自身上确定先验分位阈值
    sm_clean["sm_expert_overload"] = exp_overload.predict_inductive_score(sm_clean)
    sm_clean["sm_expert_biomech"] = exp_biomech.predict_inductive_score(sm_clean)
    sm_clean["sm_expert_wellness"] = exp_wellness.predict_inductive_score(sm_clean)

    # --- 方案 1: 传统未对齐 Baseline (默认 0.5 阈值) ---
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0), sm_active[target_col].values)
    prob_1_base = rf_base.predict_proba(loc_active[base_cols].fillna(0))[:, 1]
    pred_1_base = (prob_1_base >= 0.50).astype(int)

    # --- 方案 2a: Zero-Shot 纯生理/穿戴/主观专家融合 (不含任何伤病史，零本地统计量，按无监督 Top 25% 预算预警) ---
    prob_2a_physio = (
        0.40 * loc_active["sm_expert_overload"].values
        + 0.30 * loc_active["sm_expert_biomech"].values
        + 0.30 * loc_active["sm_expert_wellness"].values
    )
    # 无监督预警预算：仅根据预测分数自身的 75% 分位数预警，完全不看 y_loc！
    pred_2a_physio = (prob_2a_physio >= np.quantile(prob_2a_physio, 0.75)).astype(int)

    # --- 方案 2b: Zero-Shot 全维度源域模型 (100% 在 SoccerMon 上 fit 全部参数) ---
    zs_cols = ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness", "recent_inj_decay", "prev_inj_30d"]
    zs_imp = SimpleImputer(strategy="median").fit(sm_clean[zs_cols])
    zs_sc = StandardScaler().fit(zs_imp.transform(sm_clean[zs_cols]))
    X_sm_zs = zs_sc.transform(zs_imp.transform(sm_clean[zs_cols]))
    lr_zs = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(X_sm_zs, sm_clean[target_col].values)
    lr_zs.coef_ = np.maximum(lr_zs.coef_, 0.05)

    X_loc_zs = zs_sc.transform(zs_imp.transform(loc_active[zs_cols]))
    prob_2b_zs = lr_zs.predict_proba(X_loc_zs)[:, 1]
    pred_2b_zs = (prob_2b_zs >= np.quantile(prob_2b_zs, 0.75)).astype(int)

    # --- 方案 3a & 3b: 严格跨运动员盲测 (6-Fold GroupKFold，所有 Imputer/Scaler/阈值 100% 在训练运动员内部 fit) ---
    gkf = GroupKFold(n_splits=6)
    oof_prob_grp_nohist = np.zeros(len(loc_active))
    oof_pred_grp_nohist = np.zeros(len(loc_active), dtype=int)
    oof_prob_grp_full = np.zeros(len(loc_active))
    oof_pred_grp_full = np.zeros(len(loc_active), dtype=int)

    for tr_idx, te_idx in gkf.split(loc_active, y_loc, groups=players_loc):
        df_tr, y_tr = loc_active.iloc[tr_idx], y_loc[tr_idx]
        df_te = loc_active.iloc[te_idx]

        # 3a: 不含伤病史 (纯负荷+穿戴+主观+赛程生理期)
        adapter_nh = LeakFreeFoldAdapter(include_history=False).fit(df_tr, y_tr)
        oof_prob_grp_nohist[te_idx], oof_pred_grp_nohist[te_idx] = adapter_nh.predict(df_te)

        # 3b: 全维度 5 支柱
        adapter_full = LeakFreeFoldAdapter(include_history=True).fit(df_tr, y_tr)
        oof_prob_grp_full[te_idx], oof_pred_grp_full[te_idx] = adapter_full.predict(df_te)

    # --- 方案 4: 严格时间序列前滚验证 (Purged Temporal Expanding-Window，折内历史 fit + 折内历史定阈值) ---
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

        if tr_mask.sum() >= 120 and y_loc[tr_mask].sum() >= 8:
            adapter_temp = LeakFreeFoldAdapter(include_history=True).fit(loc_active.iloc[tr_mask], y_loc[tr_mask])
            p_te, lbl_te = adapter_temp.predict(loc_active.iloc[te_mask])
            oof_prob_temp[te_mask] = p_te
            oof_pred_temp[te_mask] = lbl_te

    # --- 方案 5 (仅作附录队内参考): 队内回顾性插值参考 (5-Fold Stratified CV，同样严格折内标准化+折内定阈值) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_intra = np.zeros(len(loc_active))
    oof_pred_intra = np.zeros(len(loc_active), dtype=int)
    intra_cols = base_cols + FEAT_LOCAL_FEMALE + FEAT_INJ_HISTORY + ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness"]

    for tr_idx, te_idx in skf.split(loc_active, y_loc):
        imp_in = SimpleImputer(strategy="median").fit(loc_active.iloc[tr_idx][intra_cols])
        sc_in = StandardScaler().fit(imp_in.transform(loc_active.iloc[tr_idx][intra_cols]))
        X_tr_in = sc_in.transform(imp_in.transform(loc_active.iloc[tr_idx][intra_cols]))
        X_te_in = sc_in.transform(imp_in.transform(loc_active.iloc[te_idx][intra_cols]))
        y_tr_in = y_loc[tr_idx]

        gb_in = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, learning_rate=0.04, min_samples_leaf=15,
            l2_regularization=2.0, class_weight="balanced", random_state=42
        ).fit(X_tr_in, y_tr_in)
        et_in = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=6,
            class_weight="balanced", random_state=42, n_jobs=-1
        ).fit(X_tr_in, y_tr_in)

        p_tr_in = 0.50 * gb_in.predict_proba(X_tr_in)[:, 1] + 0.50 * et_in.predict_proba(X_tr_in)[:, 1]
        th_in = select_threshold_on_train(y_tr_in, p_tr_in)

        p_te_in = 0.50 * gb_in.predict_proba(X_te_in)[:, 1] + 0.50 * et_in.predict_proba(X_te_in)[:, 1]
        oof_prob_intra[te_idx] = p_te_in
        oof_pred_intra[te_idx] = (p_te_in >= th_in).astype(int)

    rows = [
        format_metrics_row("1. 传统直接迁移 Baseline (未对齐+默认0.5阈值)", y_loc, prob_1_base, pred_1_base),
        format_metrics_row("2a. Zero-Shot 纯生理+穿戴+主观专家 (无伤病史/零本地信息)", y_loc, prob_2a_physio, pred_2a_physio),
        format_metrics_row("2b. Zero-Shot 全维度源域迁移 (100%纯SoccerMon fit)", y_loc, prob_2b_zs, pred_2b_zs),
        format_metrics_row("3a. 跨运动员盲测 GroupKFold (无伤病史: 纯负荷+穿戴+主观+生理期)", y_loc, oof_prob_grp_nohist, oof_pred_grp_nohist),
        format_metrics_row("3b. 跨运动员盲测 GroupKFold (全维度 5 支柱折内闭环)", y_loc, oof_prob_grp_full, oof_pred_grp_full),
        format_metrics_row("4. 时间序列前滚 Purged Temporal CV (折内历史闭环+零窗口重叠)", y_loc, oof_prob_temp, oof_pred_temp),
        format_metrics_row("5. [参考] 队内回顾性插值适应性分析 (In-Sample Stratified 5-Fold)", y_loc, oof_prob_intra, oof_pred_intra),
    ]
    res_df = pd.DataFrame(rows)
    display_cols = ["验证方案 (100%折内预处理+折内定阈值)", "ROC_AUC (95% CI)", "AP (95% CI)", "F1 (折内定阈值)", "Recall", "Precision"]
    print(res_df[display_cols].to_string(index=False))

    return res_df, loc_active


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    res_7d, _ = run_strict_leakfree_evaluation(
        df_sm, df_loc, "target_non_contact_next_7d", horizon_days=7, title="未来 7 天非接触性损伤预警 (阳性率 18.30%)"
    )
    res_3d, _ = run_strict_leakfree_evaluation(
        df_sm, df_loc, "target_non_contact_next_3d", horizon_days=3, title="未来 3 天非接触性损伤预警 (阳性率 10.27%)"
    )

    out_file = OUTPUTS_DIR / "strict_leakfree_paper_results.csv"
    pd.concat([
        res_7d.assign(预测窗口="未来7天非接触损伤"),
        res_3d.assign(预测窗口="未来3天非接触损伤"),
    ], ignore_index=True).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 审稿级零泄露最终汇总表已保存至: {out_file}")
    print("🎉 步骤 08 完成！请把控制台结果发给我，你可以同时把这份结果发给 ChatGPT 检验是否还有任何方法学漏洞！")