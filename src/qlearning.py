# src/qlearning.py
# ---------------------------------------------
# 탭형 Q-learning 알고리즘 구현
# ---------------------------------------------

from typing import Tuple, List, Dict, Any

import numpy as np


def train_q_learning(
    env,
    num_states: int,
    num_actions: int,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.99,
) -> Tuple[np.ndarray, List[float]]:
    """
    탭형 Q-learning으로 에이전트를 학습시키는 함수.

    env        : reset(), step(action) 을 지원하는 환경 (HousingEnv)
    num_states : 상태 개수
    num_actions: 행동 개수
    episodes   : 에피소드 반복 횟수
    alpha      : 학습률
    gamma      : 할인율
    epsilon_*  : 탐험 비율 관련 파라미터

    return:
        Q              : (num_states, num_actions) Q-table
        episode_rewards: 각 에피소드별 총 보상 리스트
    """
    Q = np.zeros((num_states, num_actions), dtype=float)
    episode_rewards: List[float] = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        # 에피소드마다 epsilon 서서히 감소
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** ep))

        while not done:
            # ε-greedy 정책
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)  # 무작위 행동
            else:
                action = int(np.argmax(Q[state]))        # Q값이 가장 큰 행동 선택

            next_state, reward, done, info = env.step(action)

            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]

            # Q 업데이트
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"[학습] 에피소드 {ep+1:3d} / {episodes}, 총 보상 = {total_reward:.1f}, epsilon={epsilon:.3f}")

    return Q, episode_rewards


def run_greedy_policy(
    env,
    Q: np.ndarray,
    max_steps: int = 100
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """
    학습된 Q-table을 이용해 탐욕 정책(ε=0)으로 에피소드 1번 실행.

    env     : 환경 (HousingEnv)
    Q       : 학습된 Q-table
    max_steps: 안전장치(최대 스텝 수)

    return:
        total_reward: 총 보상
        steps       : 실제 수행된 스텝 수
        history     : 각 스텝별 정보 딕셔너리 리스트
    """
    num_actions = Q.shape[1]

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    history: List[Dict[str, Any]] = []

    while not done and steps < max_steps:
        # 탐욕 정책: 항상 Q값이 가장 큰 행동 선택
        action = int(np.argmax(Q[state]))

        next_state, reward, done, info = env.step(action)

        step_info = {
            "step": steps,
            "current_ym": info.get("current_ym"),
            "next_ym": info.get("next_ym"),
            "action_id": action,
            "true_direction_label": info.get("true_direction_label"),
            "reward": reward,
        }
        history.append(step_info)

        total_reward += reward
        state = next_state
        steps += 1

    return total_reward, steps, history
