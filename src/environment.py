# src/environment.py
# ---------------------------------------------
# 월별 아파트 가격 방향 예측을 위한 강화학습 환경
# (금리수준, 인구추세를 포함한 상태 사용)
# ---------------------------------------------

from typing import Tuple, Dict, Any

import pandas as pd

from src.state_encoder import encode_state


class HousingEnv:
    """
    월별 아파트 가격 방향(direction)을 맞추는 게임 환경.

    - 상태(state): 현재 월의
        * 가격 방향(direction: -1/0/1)
        * 금리 수준(rate_level: 0/1/2)
        * 인구 추세(pop_trend: -1/0/1)
      을 encode_state()로 합쳐 만든 state_id (0~26)

    - 행동(action): 에이전트가 예측하는 다음 달의 방향
        * 0: 하락 예측
        * 1: 보합 예측
        * 2: 상승 예측

    - 보상(reward):
        * 예측 == 실제 방향 → +1
        * 예측 != 실제 방향 → -1
    """

    def __init__(self, monthly_df: pd.DataFrame):
        """
        monthly_df: make_monthly_panel + add_price_direction +
                    merge_macro_to_monthly + add_macro_levels 이후의 DataFrame.

        필수 컬럼:
          - 'ym'
          - 'direction'  (-1/0/1)
          - 'rate_level' (0/1/2)
          - 'pop_trend'  (-1/0/1)
        """
        self.df = monthly_df.reset_index(drop=True).copy()

        self.num_rows = len(self.df)
        # t에서 행동을 하면 t+1의 방향으로 보상을 주므로
        # 마지막 인덱스 직전까지만 스텝을 수행 가능
        self.max_step_index = self.num_rows - 2

        if self.num_rows < 2:
            print("[ERROR] 월별 데이터가 2개 미만입니다. 환경을 만들 수 없습니다.")

        self.current_idx = None

    def _get_state(self, idx: int) -> int:
        """
        현재 인덱스 idx에서의 상태(state_id)를 계산.
        direction, rate_level, pop_trend 모두 사용.
        """
        row = self.df.loc[idx]
        direction = int(row["direction"])
        rate_level = int(row["rate_level"])
        pop_trend = int(row["pop_trend"])

        state_id = encode_state(direction, rate_level, pop_trend)
        return state_id

    def reset(self) -> int:
        """
        에피소드를 초기화하고 초기 상태를 반환.
        항상 처음 월(idx=0)의 state를 사용.
        """
        if self.num_rows < 2:
            raise ValueError("데이터가 너무 적어서 에피소드를 시작할 수 없습니다.")

        self.current_idx = 0
        state = self._get_state(self.current_idx)
        return state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        한 스텝 진행.

        action: 0(하락 예측) / 1(보합 예측) / 2(상승 예측)

        반환:
            next_state: 다음 상태(state_id)
            reward: 보상 (float)
            done: 에피소드 종료 여부 (bool)
            info: 디버깅/로깅용 정보 (dict)
        """
        if self.current_idx is None:
            raise RuntimeError("reset()을 먼저 호출해야 합니다.")

        if self.current_idx > self.max_step_index:
            # 이미 에피소드가 끝난 상태
            return self._get_state(self.current_idx), 0.0, True, {"msg": "에피소드 종료 상태에서 step 호출"}

        # 현재 인덱스 t, 다음 인덱스 t+1
        t = self.current_idx
        t_next = t + 1

        # 실제 다음 달 방향 (direction 컬럼은 -1/0/1 값)
        true_direction_next = int(self.df.loc[t_next, "direction"])

        # 실제 방향을 ID로 바꾼 값 (0/1/2)
        if true_direction_next < 0:
            true_action_id = 0
        elif true_direction_next > 0:
            true_action_id = 2
        else:
            true_action_id = 1

        # 보상 계산: 예측(action) vs 실제(true_action_id)
        reward = 1.0 if int(action) == true_action_id else -1.0

        # 인덱스 한 칸 이동
        self.current_idx = t_next

        # 다음 상태 계산
        next_state = self._get_state(self.current_idx)

        # 에피소드 종료 여부
        done = self.current_idx >= self.max_step_index + 1

        # 디버깅용 정보
        info = {
            "current_ym": self.df.loc[t, "ym"],
            "next_ym": self.df.loc[t_next, "ym"],
            "true_direction_next": true_direction_next,
            "true_direction_label": {
                -1: "하락",
                0: "보합",
                1: "상승"
            }.get(true_direction_next, "알수없음"),
        }

        return next_state, reward, done, info
