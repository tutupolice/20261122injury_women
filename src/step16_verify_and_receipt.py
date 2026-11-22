# src/step16_verify_and_receipt.py
"""
Step 16 (全链路可溯源审计版):
生成 outputs/run_receipt.json，包含：
1. Git commit ID 与工作区干净状态；
2. 全部正式管线源码 (src/step*.py) 的 SHA256；
3. 全部处理后输入面板 (data/processed/*.csv) 的 SHA256 与风险集精确计数 (验证 is_rehab / is_injured 已真实存在并执行)；
4. 全部 8 份核心输出表与 5 张论文高清图 (fig1~fig5) 的 SHA256。
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures_final"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit_short": commit, "working_tree_clean": len(status) == 0}
    except Exception:
        return {"commit_short": "not_a_git_repo_or_no_commit", "working_tree_clean": False}


def hash_file_list(base_dir: Path, filenames: list[str]):
    out, missing = {}, []
    for fn in filenames:
        p = base_dir / fn
        if p.exists():
            out[fn] = {"exists": True, "size_bytes": p.stat().st_size, "sha256": file_sha256(p)}
        else:
            out[fn] = {"exists": False}
            missing.append(fn)
    return out, missing


def main():
    sm_panel = pd.read_csv(PROCESSED_DIR / "soccermon_daily_panel.csv")
    loc_panel = pd.read_csv(PROCESSED_DIR / "local_daily_panel.csv")

    # 强制断言：显式康复与伤停列必须真实存在
    for name, df in [("soccermon", sm_panel), ("local_24", loc_panel)]:
        assert "is_rehab" in df.columns and "is_injured" in df.columns, f"{name} 缺少显式 is_rehab/is_injured 列！"

    sm_act = sm_panel[sm_panel["active_day"] == 1]
    sm_risk = sm_act[(sm_act["is_rehab"] == 0) & (sm_act["is_injured"] == 0)]

    loc_act = loc_panel[loc_panel["active_day"] == 1]
    loc_risk = loc_act[(loc_act["is_rehab"] == 0) & (loc_act["is_injured"] == 0)]

    code_files = [
        "step02_build_local_dataset.py",
        "step02b_build_soccermon_dataset.py",
        "step02c_fix_calendar_and_targets.py",
        "step09_final_clean_pipeline.py",
        "step03_feature_engineering.py",
        "step14_soccermon_8models_benchmark.py",
        "step13_shap_analysis.py",
        "step05_freeze_external_validation.py",
        "step11_locked_paper_suite.py",
        "step12_final_figures.py",
        "step16_verify_and_receipt.py",
    ]
    input_files = [
        "soccermon_daily_panel.csv",
        "local_daily_panel.csv",
        "soccermon_features.csv",
        "local_features.csv",
    ]
    output_tables = [
        "step14_soccermon_8models_cv.csv",
        "step14_best_source_model_params.json",
        "step13_shap_soccermon_best_lr_ranking.csv",
        "step05_frozen_lr_external_validation.csv",
        "step05_frozen_lr_budget_sensitivity.csv",
        "step11_table1_main_results.csv",
        "step11_table2_budget_sensitivity.csv",
        "step11_table3_anonymized_folds.csv",
    ]
    output_figs = [
        "fig1_roc_pr_validation.png",
        "fig2_calibration_and_quintiles.png",
        "fig3_alert_budget_and_folds.png",
        "fig4_anonymized_trajectories.png",
        "fig5_shap_soccermon_best_lr.png",
    ]

    code_hashes, miss_code = hash_file_list(SRC_DIR, code_files)
    input_hashes, miss_in = hash_file_list(PROCESSED_DIR, input_files)
    table_hashes, miss_tab = hash_file_list(OUTPUTS_DIR, output_tables)
    fig_hashes, miss_fig = hash_file_list(FIG_DIR, output_figs)

    with open(OUTPUTS_DIR / "step14_best_source_model_params.json", "r", encoding="utf-8") as f:
        best_src = json.load(f)

    receipt = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_state": get_git_info(),
        "pipeline_specification": {
            "prediction_timing": "End of day t (features <= t) -> strict future window (t, t+7] and (t, t+3]",
            "risk_set_rule": "active_day == 1 AND is_rehab == 0 AND is_injured == 0 (explicitly persisted)",
            "stage_1_source": "SoccerMon (all 50 players, all-injury target_all_inj_next_7d)",
            "stage_2_frozen_transfer": "Local 24 players (Zero-Shot frozen source model, same-definition self-reported injury external validation)",
            "stage_3_target_updating": "Local 24 players (Monotonic ensemble + StratifiedGroupKFold + Platt calibration)",
        },
        "cohort_summary": {
            "soccermon": {
                "n_players": int(sm_panel["player_id"].nunique()),
                "calendar_rows": int(len(sm_panel)),
                "active_monitored_days": int(len(sm_act)),
                "at_risk_days_after_rehab_exclusion": int(len(sm_risk)),
                "excluded_injured_or_rehab_days": int(len(sm_act) - len(sm_risk)),
                "all_injury_onsets_total": int(sm_panel["inj_all_onset"].sum()),
                "all_injury_onsets_in_active_days": int(sm_act["inj_all_onset"].sum()),
                "right_censored_excluded_7d_days": int(sm_risk["target_all_inj_next_7d"].isna().sum()),
                "strict_future_7d_positives_in_risk_set": int(sm_risk["target_all_inj_next_7d"].dropna().sum()),
            },
            "local_24": {
                "n_players": int(loc_panel["player_id"].nunique()),
                "calendar_rows": int(len(loc_panel)),
                "active_monitored_days": int(len(loc_act)),
                "at_risk_days_after_rehab_exclusion": int(len(loc_risk)),
                "excluded_injured_or_rehab_days": int(len(loc_act) - len(loc_risk)),
                "strict_future_7d_all_inj_positives": int(loc_risk["target_all_inj_next_7d"].sum()),
                "strict_future_3d_all_inj_positives": int(loc_risk["target_all_inj_next_3d"].sum()),
            },
        },
        "stage_1_best_source_model": {
            "model": best_src["best_model"],
            "pooled_ap": best_src["pooled_ap"],
            "pooled_roc_auc": best_src["pooled_roc_auc"],
            "tuned_hyperparameters": best_src["tuned_hyperparameters"],
            "n_selected_features": len(best_src["feature_audit"]["lasso_selected_features"]),
        },
        "provenance_hashes": {
            "code_files": code_hashes,
            "study_protocol_sha256": file_sha256(PROJECT_ROOT / "config" / "study_protocol.json"),
            "raw_local_injury_log_sha256": next((file_sha256(f) for f in (PROJECT_ROOT / "data" / "raw" / "local_24").rglob("女足损伤记录表.xlsx") if "备份" not in f.name), "not_found"),
            "input_processed_files": input_hashes,
            "output_tables": table_hashes,
            "output_figures": fig_hashes,
        },
        "all_artifacts_verified": len(miss_code + miss_in + miss_tab + miss_fig) == 0,
    }

    receipt_path = OUTPUTS_DIR / "run_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("🧾 全链路可溯源运行回执已生成:", receipt_path)
    print(f"   - Git Commit: {receipt['git_state']['commit_short']} | 工作区干净: {receipt['git_state']['working_tree_clean']}")
    print(f"   - SoccerMon 风险集: {len(sm_risk)} 天 (已显式排除 {len(sm_act)-len(sm_risk)} 伤停/康复日)")
    print(f"   - 本地 24 人风险集: {len(loc_risk)} 天 (已显式排除 {len(loc_act)-len(loc_risk)} 伤停/康复日)")
    print(f"   - 全部代码/输入/表格/图件齐全: {receipt['all_artifacts_verified']}")
    print("=" * 100)


if __name__ == "__main__":
    main()