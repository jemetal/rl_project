# src/macro_features.py
# ---------------------------------------------
# 금리/인구 데이터를 월별(ym)로 정리하고
# 아파트 월별 가격 데이터(monthly_df)에 결합하는 유틸 모음
# + 강화학습용 macro 상태(rate_level, pop_trend) 생성
# ---------------------------------------------

from typing import Tuple

import pandas as pd


# ===== 1. 기준금리: 일별 → 월별 평균 =====

def load_monthly_rate(filepath: str) -> pd.DataFrame:
    """
    한국은행 기준금리 일별 데이터를 읽어서
    연-월(ym) 기준 월별 평균금리로 집계.

    입력 파일 컬럼 예시: ['연', '월', '일', '기준금리']
    반환 컬럼: ['ym', 'base_rate']
    """
    df = pd.read_excel(filepath, sheet_name="데이터")

    # 컬럼 이름 통일
    rename_map = {
        "연": "year",
        "월": "month",
        "일": "day",
        "기준금리": "base_rate",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 연월 문자열 생성
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    # 월별 평균금리 집계
    monthly_rate = (
        df.groupby("ym")
        .agg(base_rate=("base_rate", "mean"))
        .reset_index()
        .sort_values("ym")
        .reset_index(drop=True)
    )

    return monthly_rate


# ===== 2. 인구: 분기별 → 월별 =====

def load_monthly_population(filepath: str) -> pd.DataFrame:
    """
    자치구별 분기(1/4,2/4,3/4,4/4) 인구 데이터를 읽어서
    연-월(ym) 기준 월별 인구로 펼침.

    입력 시트 구조:
      - 자치구별(1): 구 이름
      - 성별(1): '합계'만 사용
      - 나머지 컬럼: '2023 1/4', '2023 2/4', ...

    반환 컬럼: ['gu', 'ym', 'population']
    """
    df = pd.read_excel(filepath, sheet_name="데이터")

    # 0행은 설명 행이 섞여 있으므로 제거
    df = df.iloc[1:, :].reset_index(drop=True)

    # 자치구 / 성별 컬럼 이름 정리
    df = df.rename(columns={"자치구별(1)": "gu", "성별(1)": "sex"})

    # '합계' (전체 인구)만 사용
    df = df[df["sex"] == "합계"].copy()

    # 분기 컬럼들만 선택 (gu, sex 제외)
    quarter_cols = [c for c in df.columns if c not in ["gu", "sex"]]

    records = []

    for _, row in df.iterrows():
        gu = row["gu"]

        for col in quarter_cols:
            # 예: col = "2023 1/4"
            try:
                year_str, q_str = str(col).split()
                year = int(year_str)
                quarter = int(q_str.split("/")[0])
            except Exception:
                continue

            pop = row[col]

            if pd.isna(pop):
                continue

            # 각 분기를 해당하는 3개월로 펼치기
            if quarter == 1:
                months = [1, 2, 3]
            elif quarter == 2:
                months = [4, 5, 6]
            elif quarter == 3:
                months = [7, 8, 9]
            else:  # 4분기
                months = [10, 11, 12]

            for m in months:
                ym = f"{year}-{m:02d}"
                records.append({"gu": gu, "ym": ym, "population": pop})

    monthly_pop = pd.DataFrame(records)

    # 혹시 중복이 있을 수 있으니 평균으로 정리
    monthly_pop = (
        monthly_pop.groupby(["gu", "ym"])
        .agg(population=("population", "mean"))
        .reset_index()
        .sort_values(["gu", "ym"])
        .reset_index(drop=True)
    )

    return monthly_pop


# ===== 3. 월별 가격 데이터와 금리/인구를 merge =====

def merge_macro_to_monthly(
    monthly_df: pd.DataFrame,
    selected_gu: str,
    monthly_rate: pd.DataFrame,
    monthly_pop: pd.DataFrame,
) -> pd.DataFrame:
    """
    선택된 구의 월별 가격 데이터(monthly_df)에
    - 전체 기준금리 월별 평균 (monthly_rate)
    - 해당 구의 월별 인구 (monthly_pop, selected_gu 필터)
    를 left-join으로 결합.

    monthly_df: ['ym', 'mean_price', 'direction', ...]
    monthly_rate: ['ym', 'base_rate']
    monthly_pop : ['gu', 'ym', 'population']

    반환: monthly_df에 base_rate, population 컬럼이 추가된 DataFrame
    """
    df = monthly_df.copy()

    # 기준금리 merge (전체 공통, ym 기준)
    df = df.merge(monthly_rate, on="ym", how="left")

    # 인구 merge (구 + ym 기준) → 선택된 구만 사용
    gu_pop = monthly_pop[monthly_pop["gu"] == selected_gu].copy()
    df = df.merge(gu_pop[["ym", "population"]], on="ym", how="left")

    return df


# ===== 4. 강화학습용 macro 상태 만들기 =====

def add_macro_levels(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    월별 가격 + 금리 + 인구가 포함된 DataFrame(df_macro)에
    강화학습용 상태 컬럼을 추가.

    - rate_level: 금리 수준 (0=낮음, 1=중간, 2=높음)
        * 낮음(low)  : base_rate < 3.0
        * 중간(mid)  : 3.0 <= base_rate < 3.5
        * 높음(high) : base_rate >= 3.5

    - pop_trend: 인구 추세 (-1=감소, 0=보합, 1=증가)
        * 전월 대비 population 차이의 부호를 사용
    """
    df = df_macro.copy().reset_index(drop=True)

    # ---- 금리 수준(rate_level) ----
    def _rate_level(r):
        if pd.isna(r):
            return 1  # 정보 없으면 중간으로 처리
        if r < 3.0:
            return 0  # 낮음
        elif r < 3.5:
            return 1  # 중간
        else:
            return 2  # 높음

    df["rate_level"] = df["base_rate"].apply(_rate_level).astype(int)

    # ---- 인구 추세(pop_trend) ----
    # 선택된 구의 월별 데이터만 들어 있으므로, 단순 diff 사용
    df["pop_diff"] = df["population"].diff()

    def _pop_trend(d):
        if pd.isna(d) or d == 0:
            return 0  # 첫 달 or 변화 없음 → 보합
        elif d > 0:
            return 1  # 증가
        else:
            return -1  # 감소

    df["pop_trend"] = df["pop_diff"].apply(_pop_trend).astype(int)

    return df
