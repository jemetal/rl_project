# src/api.py
# ---------------------------------------------
# Streamlit에서 쓰기 좋은 고수준 API 함수 모음
#  - 데이터 로딩
#  - 구/아파트/평형 리스트
#  - 한 아파트/평형의 state_df 생성
#  - Q-learning 학습 + 평가
#  - 1개월 ahead 예측 + 12개월 시나리오 예측
# ---------------------------------------------

from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from src.data_loader import load_transaction_data
from src.preprocess import (
    filter_one_apt,
    make_monthly_panel,
    add_price_direction,
)
from src.macro_features import (
    load_monthly_rate,
    load_monthly_population,
    merge_macro_to_monthly,
    add_macro_levels,
)
from src.environment import HousingEnv
from src.qlearning import train_q_learning, run_greedy_policy
from src.state_encoder import encode_state  # 상태 인코딩용


# ---------- 1. 데이터 로딩 ----------

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    실거래가, 기준금리, 인구 데이터를 한 번에 로딩해서 반환.
    """
    trans_path = "data/3_(전체)아파트(매매)_실거래가_20251129130725.xlsx"
    rate_path = "data/2_기준금리_한국은행 기준금리 및 여수신금리_29140618.xlsx"
    pop_path = "data/1_주민등록인구_20251129142612.xlsx"

    trans_df = load_transaction_data(trans_path)
    monthly_rate = load_monthly_rate(rate_path)
    monthly_pop = load_monthly_population(pop_path)

    return trans_df, monthly_rate, monthly_pop


# ---------- 2. 메뉴용 리스트 함수 ----------

def get_gu_list(trans_df: pd.DataFrame) -> List[str]:
    return sorted(trans_df["gu"].dropna().unique().tolist())


def get_apt_list(trans_df: pd.DataFrame, gu: str) -> List[str]:
    df = trans_df[trans_df["gu"] == gu]
    return sorted(df["apt_name"].dropna().unique().tolist())


def get_area_list(trans_df: pd.DataFrame, gu: str, apt: str) -> List[float]:
    df = trans_df[
        (trans_df["gu"] == gu) &
        (trans_df["apt_name"] == apt)
    ]
    # 숫자형으로 변환 후 정렬
    areas = pd.to_numeric(df["area"], errors="coerce").dropna().unique().tolist()
    areas = sorted(areas)
    return areas


# ---------- 3. 한 아파트/평형의 state_df 생성 ----------

def build_state_df_for_apt(
    trans_df: pd.DataFrame,
    selected_gu: str,
    selected_apt: str,
    selected_area: float,
    monthly_rate: pd.DataFrame,
    monthly_pop: pd.DataFrame,
) -> pd.DataFrame:
    """
    (구, 아파트, 평형)을 지정하면
    월별 가격 + 방향 + 금리 + 인구 + macro 상태(rate_level, pop_trend)를
    모두 포함한 DataFrame(state_df)을 생성해서 반환.
    """
    # 1) 해당 아파트/평형 필터링
    apt_df = filter_one_apt(trans_df, selected_gu, selected_apt, selected_area)
    if apt_df.empty:
        return pd.DataFrame()

    # 2) 월별 평균 실거래가
    monthly_df = make_monthly_panel(apt_df)
    if monthly_df.empty or len(monthly_df) < 2:
        return pd.DataFrame()

    # 3) 전월 대비 변화율 + 방향(label)
    monthly_df = add_price_direction(monthly_df, threshold=0.01)

    # 4) 금리/인구 merge
    monthly_with_macro = merge_macro_to_monthly(
        monthly_df,
        selected_gu,
        monthly_rate,
        monthly_pop,
    )

    # 5) macro 상태(rate_level, pop_trend) 생성
    state_df = add_macro_levels(monthly_with_macro)

    if len(state_df) < 2:
        return pd.DataFrame()

    return state_df


# ---------- 4-A. 1개월 ahead 예측용 함수 ----------

def _predict_next_month_direction(state_df: pd.DataFrame, Q: np.ndarray) -> Dict[str, Any]:
    """
    state_df의 마지막 행(가장 최근 월)을 기준으로,
    Q-table에서 최적 행동을 골라 '다음 달 방향'을 예측한다.
    """
    last_row = state_df.iloc[-1]

    # 마지막 월의 상태를 state index로 인코딩
    s = encode_state(
        int(last_row["direction"]),
        int(last_row["rate_level"]),
        int(last_row["pop_trend"]),
    )

    # Q-table에서 가장 값이 큰 action 선택
    best_action = int(np.argmax(Q[s]))
    direction_label = ["하락", "보합", "상승"][best_action]

    last_ym = str(last_row["ym"])

    # 'YYYY-MM' → 다음 달 'YYYY-MM' 계산
    try:
        period = pd.Period(last_ym, freq="M")
        next_ym = (period + 1).strftime("%Y-%m")
    except Exception:
        # 혹시 형식이 다르면 그냥 "다음 달"로 처리
        next_ym = "다음 달"

    return {
        "last_ym": last_ym,
        "next_ym": next_ym,
        "action_id": best_action,
        "direction_label": direction_label,
    }


# ---------- 4-B. 12개월 시나리오 예측 함수 ----------

def simulate_future_12months(state_df: pd.DataFrame, Q: np.ndarray) -> pd.DataFrame:
    """
    state_df의 마지막 월을 기준으로 강화학습 정책(Q-table)을 이용해
    향후 12개월 동안의 방향(상승/보합/하락)과 시나리오 가격을 생성.

    return:
        scenario_df: 각 월별로
            - step: 1~12
            - ym: 'YYYY-MM'
            - predicted_direction: -1/0/1
            - predicted_direction_label: '하락'/'보합'/'상승'
            - predicted_action_label: '하락 예측'/'보합 예측'/'상승 예측'
            - applied_return: 적용된 월간 수익률
            - scenario_price: 시나리오 가격
    """
    if state_df.empty:
        return pd.DataFrame()

    df = state_df.copy()

    # 과거 데이터에서 방향별 평균 변화율 계산 (없으면 fallback)
    if "pct_change" in df.columns:
        up_mean = df.loc[df["direction"] == 1, "pct_change"].mean()
        flat_mean = df.loc[df["direction"] == 0, "pct_change"].mean()
        down_mean = df.loc[df["direction"] == -1, "pct_change"].mean()
    else:
        up_mean = flat_mean = down_mean = np.nan

    # 안전한 기본값 세팅 (NaN인 경우)
    if not np.isfinite(up_mean):
        up_mean = 0.01  # +1% 기본
    if not np.isfinite(flat_mean):
        flat_mean = 0.0  # 0%
    if not np.isfinite(down_mean):
        down_mean = -0.01  # -1%

    # 마지막 관측 월 기준으로 시작
    last_row = df.iloc[-1]
    current_direction = int(last_row["direction"])
    rate_level = int(last_row["rate_level"])
    pop_trend = int(last_row["pop_trend"])
    current_price = float(last_row["mean_price"])
    last_ym = str(last_row["ym"])

    try:
        period = pd.Period(last_ym, freq="M")
    except Exception:
        # ym 형식이 이상하면 그냥 인덱스로만 증가
        period = None

    action_to_direction = {0: -1, 1: 0, 2: 1}
    direction_label_map = {-1: "하락", 0: "보합", 1: "상승"}
    action_label_map = {0: "하락 예측", 1: "보합 예측", 2: "상승 예측"}

    records = []

    for step in range(1, 13):
        # 현재 상태를 인코딩
        s = encode_state(current_direction, rate_level, pop_trend)
        best_action = int(np.argmax(Q[s]))
        predicted_direction = action_to_direction[best_action]

        # 방향에 따른 평균 수익률 적용
        if predicted_direction == 1:
            applied_return = up_mean
        elif predicted_direction == 0:
            applied_return = flat_mean
        else:
            applied_return = down_mean

        # 가격 업데이트
        current_price = current_price * (1.0 + applied_return)

        # 월 증가
        if period is not None:
            period = period + 1
            ym_str = period.strftime("%Y-%m")
        else:
            ym_str = f"+{step}개월"

        records.append({
            "step": step,
            "ym": ym_str,
            "predicted_direction": predicted_direction,
            "predicted_direction_label": direction_label_map[predicted_direction],
            "predicted_action_label": action_label_map[best_action],
            "applied_return": applied_return,
            "scenario_price": current_price,
        })

        # 다음 step에서 사용할 상태 업데이트
        current_direction = predicted_direction
        # rate_level, pop_trend는 여기서는 고정 가정 (단순화)

    scenario_df = pd.DataFrame(records)
    return scenario_df


# ---------- 4-C. Q-learning 학습 + 평가 ----------

def train_rl_for_state_df(
    state_df: pd.DataFrame,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.98,
) -> Tuple[np.ndarray, List[float], Dict[str, Any], pd.DataFrame]:
    """
    주어진 state_df에 대해 Q-learning을 수행하고,
    탐욕 정책으로 평가한 결과(1 에피소드)까지 반환.
    (1개월 ahead 예측 정보는 metrics에 포함)
    """
    # 환경 생성
    env = HousingEnv(state_df)

    num_states = 27   # 3(방향) × 3(금리) × 3(인구추세)
    num_actions = 3   # 하락 예측, 보합 예측, 상승 예측

    # Q-learning 학습
    Q, episode_rewards = train_q_learning(
        env,
        num_states=num_states,
        num_actions=num_actions,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    # 탐욕 정책으로 평가
    total_reward, steps, history = run_greedy_policy(env, Q, max_steps=100)

    # 정답률 계산
    correct = sum(1 for h in history if h["reward"] > 0)
    wrong = sum(1 for h in history if h["reward"] < 0)
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0.0

    # 1개월 ahead 예측
    future_pred = _predict_next_month_direction(state_df, Q)

    metrics = {
        "total_reward": total_reward,
        "steps": steps,
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy,
        "future_last_ym": future_pred["last_ym"],
        "future_next_ym": future_pred["next_ym"],
        "future_action_id": future_pred["action_id"],
        "future_direction_label": future_pred["direction_label"],
    }

    history_df = pd.DataFrame(history)

    return Q, episode_rewards, metrics, history_df
