# src/step02c_fix_calendar_and_targets.py
"""
Step 02c (双队列同源 + 精确次日康复休息日最终定稿版):
1. 双队列结局 100% 统一为【全部队员自报告运动损伤 (inj_all_onset)】；
2. 严格未来预测窗口 (t, t+h]：使用 <= t 日特征预测 [t+1, t+7] 与 [t+1, t+3]，彻底零同日泄露；
3. 本地 24 人康复休息日 (is_rehab) 严格读取整理后的《女足损伤记录表.xlsx》中的【严重程度（天）】字段：
   若第 d 日受伤且严重程度为 k 天 (k >= 1)，则第 d+1 日至第 d+k 日精确标为损伤休息/康复日 (is_rehab=1, is_injured=1)。
"""
from pathlib import Path
import re
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def strict_future_window_max(series: pd.Series, horizon_days: int) -> pd.Series:
    """严格未来窗口 [t+1, t+horizon_days]（绝不含第 t 日当日）"""
    s_next = series.fillna(0).astype(int).shift(-1).fillna(0).astype(int)
    rev_max = s_next.iloc[::-1].rolling(window=horizon_days, min_periods=1).max()
    return rev_max.iloc[::-1].astype(int)


def load_local_rehab_intervals() -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """读取整理后的《女足损伤记录表.xlsx》，提取每位队员 (受伤日期, 严重程度天数k)"""
    inj_files = list(LOCAL_RAW_DIR.rglob("*损伤记录表*.xlsx"))
    if not inj_files:
        raise FileNotFoundError(f"未在 {LOCAL_RAW_DIR} 下找到《女足损伤记录表.xlsx》，请确认已放入！")

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

    print(f"📋 已加载整理版《{inj_files[0].name}》: 共 {len(inj_df)} 条登记，涉及 {len(records)} 名队员（按次日起计入严重程度休息日）")
    return records


def fix_panel(csv_path: Path, is_soccermon: bool, local_rehab_map=None):
    backup_path = csv_path.with_name(csv_path.stem + "_pre_audit.csv")
    if backup_path.exists():
        df = pd.read_csv(backup_path)
    else:
        shutil.copy2(csv_path, backup_path)
        df = pd.read_csv(csv_path)

    group_col = "player_id" if "player_id" in df.columns else "athlete_id"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([group_col, "date"]).drop_duplicates(subset=[group_col, "date"]).reset_index(drop=True)

    onset_col = next((c for c in ["inj_all_onset", "injury_onset", "inj_non_contact_onset"] if c in df.columns), None)
    df["inj_all_onset"] = df[onset_col].fillna(0).astype(int)

    mon_candidates = [
        "fatigue", "fatigue_01", "soreness", "soreness_01", "sleep_quality",
        "daily_load", "srpe_load", "unified_daily_load", "total_distance", "dist",
    ]
    mon_cols = [c for c in mon_candidates if c in df.columns]

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
            # 严格按整理后的《女足损伤记录表.xlsx》标记新发日与次日起的 k 天损伤休息日
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

        g["target_all_inj_next_7d"] = strict_future_window_max(g["inj_all_onset"], 7)
        g["target_all_inj_next_3d"] = strict_future_window_max(g["inj_all_onset"], 3)

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
    n_pos7 = int(at_risk_df["target_all_inj_next_7d"].sum())
    n_pos3 = int(at_risk_df["target_all_inj_next_3d"].sum())
    print(f"✅ [{label}] 日历={len(fixed_df)}天 | 活跃监控日={len(active_df)}天 | 损伤休息日(is_rehab)={rehab_days_total}天 | 风险集={len(at_risk_df)}天")
    print(f"   -> 全部自报告伤病日={all_onsets} (活跃日内={act_onsets}) | 严格未来(t,t+7]阳性={n_pos7} ({n_pos7/len(at_risk_df)*100:.2f}%) | (t,t+3]阳性={n_pos3} ({n_pos3/len(at_risk_df)*100:.2f}%)")


def main():
    print("=" * 110)
    print("🔧 Step 02c (最终定稿版): 双队列自报告全因损伤 + 严格未来窗 (t, t+h] + 精确次日康复休息日排除")
    print("=" * 110)
    local_rehab_map = load_local_rehab_intervals()
    fix_panel(PROCESSED_DIR / "soccermon_daily_panel.csv", is_soccermon=True)
    fix_panel(PROCESSED_DIR / "local_daily_panel.csv", is_soccermon=False, local_rehab_map=local_rehab_map)
    print("=" * 110)


if __name__ == "__main__":
    main()