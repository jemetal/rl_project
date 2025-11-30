# src/preprocess.py
# ---------------------------------------------
# 선택된 (구, 아파트, 평형)에 대해
# 1) 월별 평균 실거래가 집계
# 2) 전월 대비 가격 변화율 + 방향(label) 계산
# ---------------------------------------------

import pandas as pd
import numpy as np


def filter_one_apt(trans_df: pd.DataFrame,
                   gu: str,
                   apt_name: str,
                   area) -> pd.DataFrame:
    """
    전체 실거래가 데이터(trans_df)에서
    선택된 (구, 아파트명, 평형)에 해당하는 행만 필터링.

    return: 필터링된 DataFrame (날짜/가격 포함)
    """
    cond = (
        (trans_df["gu"] == gu) &
        (trans_df["apt_name"] == apt_name) &
        (trans_df["area"] == area)
    )
    df_apt = trans_df[cond].copy()

    if df_apt.empty:
        print(f"[WARN] {gu} {apt_name} {area} 에 해당하는 데이터가 없습니다.")
        return df_apt

    # ym 컬럼이 없다면 year, month로 생성
    if "ym" not in df_apt.columns and {"year", "month"}.issubset(df_apt.columns):
        df_apt["year"] = df_apt["year"].astype(int)
        df_apt["month"] = df_apt["month"].astype(int)
        df_apt["ym"] = df_apt["year"].astype(str) + "-" + df_apt["month"].astype(str).str.zfill(2)

    # 날짜 기준 정렬 (가능하면 date, 아니면 ym 기준)
    if "date" in df_apt.columns:
        df_apt = df_apt.sort_values("date")
    elif "ym" in df_apt.columns:
        # ym 이 "YYYY-MM" 문자열이라고 가정하고 정렬
        df_apt = df_apt.sort_values("ym")

    return df_apt


def make_monthly_panel(df_apt: pd.DataFrame) -> pd.DataFrame:
    """
    단일 아파트/평형 데이터(df_apt)를 받아서
    'ym' 기준으로 월별 평균 가격과 거래 건수를 계산.

    return: 월별 집계 DataFrame
            columns 예시: ['ym', 'mean_price', 'deal_count']
    """
    if "ym" not in df_apt.columns or "price_10k" not in df_apt.columns:
        print("[ERROR] 'ym' 또는 'price_10k' 컬럼이 없습니다. 전처리를 확인해 주세요.")
        return pd.DataFrame()

    # 월별 평균가격 + 거래건수 집계
    monthly = (
        df_apt
        .groupby("ym")
        .agg(
            mean_price=("price_10k", "mean"),
            deal_count=("price_10k", "size")
        )
        .reset_index()
    )

    # 정렬 (문자열 'YYYY-MM' 기준)
    monthly = monthly.sort_values("ym").reset_index(drop=True)

    return monthly


def add_price_direction(monthly_df: pd.DataFrame,
                        threshold: float = 0.01) -> pd.DataFrame:
    """
    월별 평균가격 테이블에
    전월 대비 변화율(pct_change)과 방향(direction)을 추가.

    direction 값:
        -1 : 하락 (변화율 <= -threshold)
         0 : 보합 (-threshold < 변화율 < threshold)
        +1 : 상승 (변화율 >= threshold)

    threshold 기본값 0.01 = 1% 기준.
    """
    if "mean_price" not in monthly_df.columns:
        print("[ERROR] 'mean_price' 컬럼이 없습니다.")
        return monthly_df

    df = monthly_df.copy()

    # 전월 대비 변화율: (이번달 - 저번달) / 저번달
    df["pct_change"] = df["mean_price"].pct_change()

    def _dir_from_pct(p):
        if pd.isna(p):
            return 0  # 첫 달은 보합(0)으로 두거나 NaN으로 둬도 됨
        if p <= -threshold:
            return -1
        elif p >= threshold:
            return 1
        else:
            return 0

    df["direction"] = df["pct_change"].apply(_dir_from_pct)

    return df
