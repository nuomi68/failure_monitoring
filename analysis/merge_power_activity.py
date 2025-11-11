from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _locate(keyword: str, suffix: str) -> Path:
    for path in DATA_DIR.glob(f"*{keyword}*{suffix}"):
        return path
    raise FileNotFoundError(f"Could not find file containing '{keyword}' with suffix '{suffix}' in {DATA_DIR}")


def _normalize_datetime_column(series: pd.Series) -> pd.Series:
    """
    Convert messy TIME strings like '2020年08月03日1200' into timezone-naive datetime.
    """
    cleaned = series.astype(str).str.replace(r"\s+", "", regex=True)
    digits = cleaned.str.replace(r"\D", "", regex=True)
    digits = digits.where(digits.str.len() >= 8)
    dt = pd.to_datetime(digits, format="%Y%m%d%H%M", errors="coerce")
    dt = dt.fillna(pd.to_datetime(digits.str[:8], format="%Y%m%d", errors="coerce"))
    return dt


def main() -> None:
    power_path = _locate("功率数据", ".xlsx")
    nuc_path = _locate("特征核素活度数据", ".xls")
    output_path = DATA_DIR / "特征核素活度_功率合并2.xlsx"

    power_df = pd.read_excel(power_path)
    power_df.columns = [str(col).strip() for col in power_df.columns]
    time_col, power_col = power_df.columns[:2]
    power_df[time_col] = pd.to_datetime(power_df[time_col], errors="coerce")
    power_df = power_df.dropna(subset=[time_col])
    power_midnight = power_df[
        power_df[time_col].dt.hour.eq(0) & power_df[time_col].dt.minute.eq(0)
    ].copy()
    power_midnight["DATE"] = power_midnight[time_col].dt.normalize()
    power_midnight.rename(columns={power_col: "功率_00时"}, inplace=True)
    power_midnight = power_midnight[["DATE", "功率_00时"]]

    workbook = pd.ExcelFile(nuc_path)
    sheet_frames: dict[str, pd.DataFrame] = {}
    for sheet in workbook.sheet_names:
        df = workbook.parse(sheet)
        if "TIME" not in [str(col).strip().upper() for col in df.columns]:
            sheet_frames[sheet] = df
            continue
        df = df.copy()
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        if "TIME" not in df.columns:
            sheet_frames[sheet] = df
            continue

        df["解析时间"] = _normalize_datetime_column(df["TIME"])
        df["DATE"] = df["解析时间"].dt.normalize()
        merged = df.merge(power_midnight, on="DATE", how="left")
        merged.drop(columns=["解析时间", "DATE"], inplace=True)
        sheet_frames[sheet] = merged

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, frame in sheet_frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Combined workbook saved to: {output_path}")


if __name__ == "__main__":
    main()
