# src/step02c_fix_calendar_and_targets.py
"""
Step 02c (全线统一口径 + 显式随访协议终稿版):
1. 双队列结局统一为【全部队员自报告运动损伤 (inj_all_onset)】；
2. 显式读取 config/study_protocol.json 中的随访覆盖截止日 (local: 2026-06-25; SoccerMon: 无面板外随访)，
   严格按 t + h <= followup_end_date 判定末端右删失（未满随访且无事件标 NaN 排除）；
3. 直接基于当前上游面板构建，彻底移除旧版 *_pre_audit.csv 备份回退；
4. 精确读取新版《女足损伤记录表.xlsx》，按次日起算 k 天休息日 (is_rehab=1)；
5. 活跃监控日 (active_day) 全面覆盖所有主客观监测列，全线统一风险集：active_day==1 & is_rehab==0 & is_injured==0。
"""
import json
from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROTOCOL_PATH = PROJECT_ROOT / "config" / "study_protocol.json"


def strict_future_window_max(series: pd.Series, horizon_days: int, followup_end_date=None) -> pd.Series:
    """严格未来窗口 [t+1, t+horizon_days]；若 t + horizon_days > followup_end_date 且期间无阳性事件则标 NaN"""
    s_next = series.fillna(0).astype(int).shift(-1).fillna(0).astype(int)
    rev_max = s_next.iloc[::-1].rolling(window=horizon_days, min_periods=1).max()
    out = rev_max.iloc[::-1].astype(float)
    if len(out) > 0:
        eff_end = followup_end_date if followup_end_date is not None else out.index.max()
        uncovered_mask = (out.index + pd.Timedelta(days=horizon_days)) > eff_end
        out.loc[uncovered_mask] = out.loc[uncovered_mask].where(out.loc[uncovered_mask] == 1, float("nan"))
    return out


def load_local_rehab_intervals() -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """精确读取新版整理《女足损伤记录表.xlsx》（自动忽略旧版备份）"""
    inj_files = [f for f in LOCAL_RAW_DIR.rglob("女足损伤记录表.xlsx") if "备份" not in f.name]
    if not inj_files:
        raise FileNotFoundError(f"未在 {LOCAL_RAW_DIR} 下找到新版《女足损伤记录表.xlsx》！")

    inj_df = pd.read_excel(inj_files[0])
    name_col = next(c for c in inj_df.columns if "姓名" in str(c))
    date_col = next(c for c in inj_df.columns if "日期" in str(c))
    sev_col = next(c for c in inj_df.columns if "严重程度" in str(c))

    records: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for _, row in inj_df.iterrows():
        name = str(row[name_col]).strip()
        dt = pd.to_datetime(row[date_col], errors="coerce")
        if pd.isna(dt):
            continue
        m = re.search(r"(\d+)", str(row[sev_col]))
        days = int(m.group(1)) if m else 0
        records.setdefault(name, []).append((dt.normalize(), days))

    print(f"📋 已精确加载新版《{inj_files[0].name}》: 共 {len(inj_df)} 条登记，涉及 {len(records)} 名队员（统一按次日起算 k 天休息日）")
    return records


def detect_all_monitoring_cols(df: pd.DataFrame) -> list[str]:
    keywords = (
        "fatigue", "soreness", "sleep", "stress", "mood", "readiness", "wellness",
        "rpe", "srpe", "duration", "daily_load", "uload",
        "distance", "hsr", "sprint", "speed", "accel", "decel", "imu", "impact", "player_load",
    )
    exclude = ("target_", "future_", "next_", "prev_inj", "inj_", "is_", "active_", "days_since", "ctx_")
    return [
        c for c in df.columns
        if any(k in c.lower() for k in keywords)
        and not c.lower().startswith(exclude)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def fix_panel(csv_path: Path, is_soccermon: bool, local_rehab_map=None, local_followup_end=None):
    # 直接读取当前上游面板（彻底脱离旧版 _pre_audit 备份）
    df = pd.read_csv(csv_path)

    group_col = "player_id" if "player_id" in df.columns else "athlete_id"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([group_col, "date"]).drop_duplicates(subset=[group_col, "date"]).reset_index(drop=True)

    onset_col = next((c for c in ["inj_all_onset", "injury_onset", "inj_non_contact_onset"] if c in df.columns), None)
    df["inj_all_onset"] = df[onset_col].fillna(0).astype(int)

    mon_cols = detect_all_monitoring_cols(df)
    eff_followup_end = None if is_soccermon else local_followup_end

    reindexed_parts = []
    for pid, grp in df.groupby(group_col, sort=False):
        full_dates = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        g = grp.set_index("date").reindex(full_dates)
        g.index.name = "date"
        g[group_col] = pid

        g["active_day"] = g[mon_cols].notna().any(axis=1).astype(int) if mon_cols else 1
        g["inj_all_onset"] = g["inj_all_onset"].fillna(0).astype(int)

        if is_soccermon:
            if "inj_prevalence" in g.columns:
                g["inj_prevalence"] = g["inj_prevalence"].fillna(0).astype(int)
                g["is_rehab"] = ((g["inj_prevalence"] == 1) & (g["inj_all_onset"] == 0)).astype(int)
                g["is_injured"] = ((g["inj_prevalence"] == 1) | (g["inj_all_onset"] == 1)).astype(int)
            else:
                g["is_rehab"] = 0
                g["is_injured"] = (g["inj_all_onset"] == 1).astype(int)
        else:
            onset_dates = set()
            rehab_dates = set()
            for inj_dt, sev_days in (local_rehab_map or {}).get(str(pid).strip(), []):
                onset_dates.add(inj_dt)
                if sev_days >= 1:
                    for offset in range(1, sev_days + 1):
                        rehab_dates.add(inj_dt + pd.Timedelta(days=offset))

            g["inj_all_onset"] = g.index.isin(onset_dates).astype(int)
            g["is_rehab"] = (g.index.isin(rehab_dates) & (g["inj_all_onset"] == 0)).astype(int)
            g["is_injured"] = ((g["inj_all_onset"] == 1) | (g["is_rehab"] == 1)).astype(int)

        g["target_all_inj_next_7d"] = strict_future_window_max(g["inj_all_onset"], 7, followup_end_date=eff_followup_end)
        g["target_all_inj_next_3d"] = strict_future_window_max(g["inj_all_onset"], 3, followup_end_date=eff_followup_end)

        drop_cols = [c for c in ["non_contact_onset", "inj_non_contact_onset", "target_non_contact_next_7d", "target_non_contact_next_3d"] if c in g.columns]
        if drop_cols:
            g = g.drop(columns=drop_cols)

        reindexed_parts.append(g.reset_index())

    fixed_df = pd.concat(reindexed_parts, ignore_index=True)
    fixed_df["date"] = fixed_df["date"].dt.strftime("%Y-%m-%d")
    fixed_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    active_df = fixed_df[fixed_df["active_day"] == 1]
    at_risk_df = active_df[(active_df["is_rehab"] == 0) & (active_df["is_injured"] == 0)]
    n_players = fixed_df[group_col].nunique()
    label = "SoccerMon 50人" if is_soccermon else f"本地 {n_players}人"

    all_onsets = int(fixed_df["inj_all_onset"].sum())
    act_onsets = int(active_df["inj_all_onset"].sum())
    rehab_days_total = int(fixed_df["is_rehab"].sum())
    valid7 = at_risk_df["target_all_inj_next_7d"].dropna()
    valid3 = at_risk_df["target_all_inj_next_3d"].dropna()
    n_pos7, n_pos3 = int(valid7.sum()), int(valid3.sum())
    print(f"✅ [{label}] 监测列={len(mon_cols)} | 活跃日={len(active_df)} | 休息日={rehab_days_total} | 统一风险集={len(at_risk_df)} (7d有效={len(valid7)}, 3d有效={len(valid3)})")
    print(f"   -> 全部伤病日={all_onsets} (活跃日内={act_onsets}) | (t,t+7]阳性={n_pos7} ({n_pos7/max(len(valid7),1)*100:.2f}%) | (t,t+3]阳性={n_pos3} ({n_pos3/max(len(valid3),1)*100:.2f}%)")


def main():
    proto = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    local_end = pd.Timestamp(proto["local_cohort_protocol"]["injury_surveillance_end"])
    print("=" * 112)
    print(f"🔧 Step 02c (显式协议终稿版): 已读取 {PROTOCOL_PATH.name} (本地独立随访截止={local_end.strftime('%Y-%m-%d')})")
    print("=" * 112)
    local_rehab_map = load_local_rehab_intervals()
    fix_panel(PROCESSED_DIR / "soccermon_daily_panel.csv", is_soccermon=True)
    fix_panel(PROCESSED_DIR / "local_daily_panel.csv", is_soccermon=False, local_rehab_map=local_rehab_map, local_followup_end=local_end)
    print("=" * 112)


if __name__ == "__main__":
    main()