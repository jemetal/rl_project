# main.py
# ---------------------------------------------
# 1) 실거래가 데이터 로딩
# 2) 구/아파트/평형 선택
# 3) 월별 평균 실거래가 + 방향 계산
# 4) 금리/인구 데이터 merge (월별 단위)
# 5) macro 상태(rate_level, pop_trend) 생성
# 6) 강화학습 환경(HousingEnv) 생성
# 7) Q-learning으로 학습
# 8) 학습된 정책으로 에피소드 실행 및 결과 출력
# ---------------------------------------------

from src.data_loader import load_transaction_data
from src.menu_select import (
    select_gu_menu,
    select_apt_menu,
    select_area_menu
)
from src.preprocess import (
    filter_one_apt,
    make_monthly_panel,
    add_price_direction
)
from src.environment import HousingEnv
from src.qlearning import train_q_learning, run_greedy_policy

from src.macro_features import (
    load_monthly_rate,
    load_monthly_population,
    merge_macro_to_monthly,
    add_macro_levels,
)

import matplotlib.pyplot as plt
import numpy as np


def main():
    # ---------------------------------------------
    # 1. 실거래가 데이터 파일 경로 설정
    # ---------------------------------------------
    filepath = "data/3_(전체)아파트(매매)_실거래가_20251129130725.xlsx"

    print("\n[INFO] 실거래가 데이터 로딩 중...")
    trans_df = load_transaction_data(filepath)

    if trans_df.empty:
        print("[ERROR] 실거래가 데이터가 비어 있습니다. 프로그램을 종료합니다.")
        return

    # ---------------------------------------------
    # 2. 1단계: 구 선택
    # ---------------------------------------------
    selected_gu = select_gu_menu(trans_df)
    if not selected_gu:
        print("[ERROR] 구 선택 실패")
        return
    print(f"\n[선택된 구] {selected_gu}")

    # ---------------------------------------------
    # 3. 2단계: 아파트 단지 선택
    # ---------------------------------------------
    selected_apt = select_apt_menu(trans_df, selected_gu)
    if not selected_apt:
        print("[ERROR] 아파트 선택 실패")
        return
    print(f"\n[선택된 아파트] {selected_gu} {selected_apt}")

    # ---------------------------------------------
    # 4. 3단계: 평형 선택
    # ---------------------------------------------
    selected_area = select_area_menu(trans_df, selected_gu, selected_apt)
    if not selected_area:
        print("[ERROR] 평형 선택 실패")
        return
    print(f"\n[최종 선택] {selected_gu} {selected_apt} {selected_area}")

    # ---------------------------------------------
    # 5. 선택된 조건으로 원본 실거래가 샘플 10건 출력
    # ---------------------------------------------
    sample = trans_df[
        (trans_df["gu"] == selected_gu) &
        (trans_df["apt_name"] == selected_apt) &
        (trans_df["area"] == selected_area)
    ].head(10)

    print("\n[선택한 아파트 평형의 실거래가 샘플 10건]")
    cols_to_show = [
        c for c in ["gu", "dong", "apt_name", "area", "year", "month", "day", "price_10k"]
        if c in sample.columns
    ]
    print(sample[cols_to_show])

    # ---------------------------------------------
    # 6. 전처리: 단일 아파트/평형 데이터 필터링
    # ---------------------------------------------
    apt_df = filter_one_apt(trans_df, selected_gu, selected_apt, selected_area)
    if apt_df.empty:
        print("[ERROR] 선택한 아파트 평형에 대한 데이터가 없습니다.")
        return

    # ---------------------------------------------
    # 7. 월별 평균 실거래가 집계
    # ---------------------------------------------
    monthly_df = make_monthly_panel(apt_df)
    if monthly_df.empty:
        print("[ERROR] 월별 집계 데이터가 비어 있습니다.")
        return

    # ---------------------------------------------
    # 8. 전월 대비 변화율 + 방향(label) 계산
    # ---------------------------------------------
    monthly_df = add_price_direction(monthly_df, threshold=0.01)

    print("\n[월별 가격 + 방향만 있는 DataFrame 미리보기]")
    print(monthly_df.head(5))
    print("컬럼:", monthly_df.columns.tolist())

    # ---------------------------------------------
    # 9. 금리/인구 데이터 로딩 및 merge
    # ---------------------------------------------
    rate_path = "data/2_기준금리_한국은행 기준금리 및 여수신금리_29140618.xlsx"
    pop_path = "data/1_주민등록인구_20251129142612.xlsx"

    monthly_rate = load_monthly_rate(rate_path)
    monthly_pop = load_monthly_population(pop_path)

    print("\n[기준금리 월별 데이터 미리보기]")
    print(monthly_rate.head(5))

    print("\n[구별 월별 인구 데이터 미리보기]")
    print(monthly_pop.head(5))

    monthly_with_macro = merge_macro_to_monthly(
        monthly_df,
        selected_gu,
        monthly_rate,
        monthly_pop,
    )

    print("\n[월별 가격 + 금리 + 인구 데이터 미리보기]")
    print(monthly_with_macro.head(12))
    print("컬럼:", monthly_with_macro.columns.tolist())

    # ---------------------------------------------
    # 10. macro 상태(rate_level, pop_trend) 생성
    # ---------------------------------------------
    state_df = add_macro_levels(monthly_with_macro)

    print("\n[강화학습에 사용할 상태(state) 컬럼 미리보기]")
    print(state_df[["ym", "mean_price", "direction", "rate_level", "pop_trend"]].head(12))

    # ---------------------------------------------
    # 11. 가격 그래프 그리기 (검증용)
    # ---------------------------------------------
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(state_df["ym"], state_df["mean_price"], marker="o")
        plt.xticks(rotation=45)
        plt.xlabel("연-월")
        plt.ylabel("평균 실거래가(만원)")
        plt.title(f"{selected_gu} {selected_apt} {selected_area} 월별 평균 실거래가")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[WARN] 그래프 출력 중 오류 발생:", e)

    # ---------------------------------------------
    # 12. 강화학습 환경 생성 (금리/인구 포함 상태 사용)
    # ---------------------------------------------
    env = HousingEnv(state_df)

    if len(state_df) < 2:
        print("[ERROR] 월별 데이터가 2개 미만이라 환경을 사용할 수 없습니다.")
        return

    # ---------------------------------------------
    # 13. Q-learning 학습
    # ---------------------------------------------
    num_states = 27   # 3(방향) × 3(금리) × 3(인구추세)
    num_actions = 3   # 하락 예측, 보합 예측, 상승 예측

    print("\n[Q-learning] 학습 시작...")
    Q, episode_rewards = train_q_learning(
        env,
        num_states=num_states,
        num_actions=num_actions,
        episodes=300,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.98,
    )
    print("\n[Q-learning] 학습 완료!")
    print("\n[Q-table]")
    print(Q)

    print("\n[에피소드별 총 보상 예시 (앞 10개)]")
    print(episode_rewards[:10])

    # ---------------------------------------------
    # 14. 학습된 정책으로 에피소드 1회 실행 (평가)
    # ---------------------------------------------
    print("\n[평가] 학습된 정책(탐욕 정책)으로 에피소드 1회 실행")
    total_reward, steps, history = run_greedy_policy(env, Q, max_steps=100)

    for h in history:
        action_label = ["하락 예측", "보합 예측", "상승 예측"][h["action_id"]]
        print(
            f"step={h['step']:2d} | "
            f"현재월={h['current_ym']} → 다음월={h['next_ym']} | "
            f"에이전트={action_label} | "
            f"실제={h['true_direction_label']} | "
            f"보상={h['reward']}"
        )

    print(f"\n[평가 종료] 총 스텝 수: {steps}, 총 보상: {total_reward}")


if __name__ == "__main__":
    main()
