# app_streamlit.py
# ---------------------------------------------
# 아파트 선택 + Q-learning 학습/평가 결과를
# 웹 대시보드 형태로 보여주는 Streamlit 앱
# ---------------------------------------------

import streamlit as st
import pandas as pd

from src.api import (
    load_all_data,
    get_gu_list,
    get_apt_list,
    get_area_list,
    build_state_df_for_apt,
    train_rl_for_state_df,
    simulate_future_12months,  # 🔥 12개월 시나리오 예측 함수
)


@st.cache_data
def load_data_cached():
    trans_df, monthly_rate, monthly_pop = load_all_data()
    return trans_df, monthly_rate, monthly_pop


def main():
    # 페이지 설정
    st.set_page_config(page_title="아파트 가격 예측", layout="wide", initial_sidebar_state="expanded")
    
    # 메인 타이틀
    st.title("🏙 강화학습 기반 서울시 아파트 가격 예측")
    st.caption("Q-learning 기반 강화학습 모델을 활용한 가격 방향 예측 시스템")
    st.markdown("---")

    # 데이터 로딩
    trans_df, monthly_rate, monthly_pop = load_data_cached()

    # ---------- 사이드바: 구 / 아파트 / 평형 선택 ----------
    with st.sidebar:
        st.header("아파트 선택")
        
        with st.expander("지역 및 아파트 정보", expanded=True):
            gu_list = get_gu_list(trans_df)
            # 초기값을 빈 값으로 설정하기 위해 placeholder 추가
            gu_options = ["선택하세요"] + gu_list
            selected_gu = st.selectbox("구 선택", gu_options, key="gu_select", index=0)
            # placeholder가 선택된 경우 None으로 처리
            if selected_gu == "선택하세요":
                selected_gu = None

            apt_list = []
            if selected_gu:
                apt_list = get_apt_list(trans_df, selected_gu)
            
            # 아파트 선택: 선택된 구가 있을 때만 활성화
            apt_options = ["선택하세요"] + apt_list if apt_list else ["선택하세요"]
            selected_apt = st.selectbox("아파트 선택", apt_options, key="apt_select", index=0, disabled=(selected_gu is None))
            if selected_apt == "선택하세요":
                selected_apt = None

            area_list = []
            if selected_gu and selected_apt:
                area_list = get_area_list(trans_df, selected_gu, selected_apt)
            
            # 평형 선택: 구와 아파트가 모두 선택되었을 때만 활성화
            area_options = ["선택하세요"] + area_list if area_list else ["선택하세요"]
            selected_area = st.selectbox("평형 선택", area_options, key="area_select", index=0, disabled=(selected_apt is None))
            if selected_area == "선택하세요":
                selected_area = None

        st.markdown("---")
        
        with st.expander("⚙️ 학습 설정", expanded=True):
            episodes = st.slider(
                "학습 에피소드 수", 
                100, 1000, 300, step=50,
                help="에피소드 수가 많을수록 학습이 더 정확해지지만 시간이 오래 걸립니다."
            )

    # ---------- 본문 ----------
    if selected_gu and selected_apt and selected_area:
        # 선택한 아파트 정보 카드
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("구", selected_gu)
        with col2:
            st.metric("아파트", selected_apt)
        with col3:
            st.metric("평형", f"{selected_area} ㎡")
        
        st.markdown("---")

        # state_df 생성
        state_df = build_state_df_for_apt(
            trans_df,
            selected_gu,
            selected_apt,
            selected_area,
            monthly_rate,
            monthly_pop,
        )

        if state_df.empty:
            st.error("⚠️ 해당 아파트/평형에 대한 월별 데이터가 부족합니다.")
            return

        # 데이터 시각화 섹션
        tab1, tab2 = st.tabs(["📊 가격 추이", "📋 데이터 미리보기"])
        
        with tab1:
            st.subheader("월별 평균 실거래가")
            price_df = state_df[["ym", "mean_price"]].copy()
            price_df = price_df.set_index("ym")
            st.line_chart(price_df, use_container_width=True)
            
            # 최근 가격 정보
            if not price_df.empty:
                recent_price = price_df["mean_price"].iloc[-1]
                prev_price = price_df["mean_price"].iloc[-2] if len(price_df) > 1 else recent_price
                price_change = recent_price - prev_price
                price_change_pct = ((price_change / prev_price) * 100) if prev_price > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("최근 평균가", f"{recent_price:,.0f}만원", f"{price_change:+,.0f}만원")
                with col2:
                    st.metric("변동률", f"{price_change_pct:+.2f}%")

        with tab2:
            st.subheader("상태(state) 데이터 미리보기")
            preview_df = state_df[["ym", "mean_price", "direction", "rate_level", "pop_trend"]].head(12)
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"전체 {len(state_df)}개월 데이터 중 최근 12개월 표시")

        st.markdown("---")

        # ---------- 학습 버튼 ----------
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button(
                "🚀 Q-learning 학습/평가 실행", 
                type="primary",
                use_container_width=True
            )

        if train_button:
            with st.spinner("강화학습 에이전트 학습 중... 잠시만 기다려주세요."):
                Q, episode_rewards, metrics, history_df = train_rl_for_state_df(
                    state_df,
                    episodes=episodes,
                )

            st.success("✅ 학습이 완료되었습니다!")
            st.markdown("---")

            # ---------- 학습 결과를 탭으로 구성 ----------
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                "📊 학습 결과 요약", 
                "🔮 1개월 예측", 
                "📉 12개월 시나리오", 
                "📜 상세 로그"
            ])

            # 탭 1: 학습 결과 요약
            with result_tab1:
                st.subheader("핵심 지표")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "총 보상", 
                        f"{metrics['total_reward']:.1f}",
                        help="에피소드 동안 누적된 총 보상"
                    )
                with col2:
                    st.metric(
                        "정답률", 
                        f"{metrics['accuracy']*100:.1f}%",
                        help="예측 정확도"
                    )
                with col3:
                    st.metric(
                        "스텝 수", 
                        metrics["steps"],
                        help="학습에 사용된 스텝 수"
                    )
                with col4:
                    st.metric(
                        "에피소드", 
                        episodes,
                        help="실행된 학습 에피소드 수"
                    )

                st.markdown("---")
                
                st.subheader("에피소드별 총 보상 변화")
                rewards_df = pd.DataFrame(
                    {"episode": range(1, len(episode_rewards) + 1),
                     "total_reward": episode_rewards}
                ).set_index("episode")
                st.line_chart(rewards_df, use_container_width=True)
                st.caption("에피소드가 진행될수록 보상이 증가하는 것을 확인할 수 있습니다.")

            # 탭 2: 1개월 예측
            with result_tab2:
                st.subheader("미래 1개월 가격 방향 예측")
                
                # 예측 결과를 카드 형태로 표시
                direction_emoji = {
                    "상승": "📈",
                    "하락": "📉",
                    "보합": "➡️"
                }
                direction_emoji_str = direction_emoji.get(metrics['future_direction_label'], "❓")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f"""
                        <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px;'>
                            <h3 style='margin-bottom: 1rem; color: #000000;'>{direction_emoji_str} {metrics['future_direction_label']}</h3>
                            <p style='font-size: 1.1rem; color: #000000;'>
                                기준 월: <strong>{metrics['future_last_ym']}</strong><br>
                                예측 대상 월: <strong>{metrics['future_next_ym']}</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 탭 3: 12개월 시나리오
            with result_tab3:
                st.subheader("12개월 시나리오 가격 예측")
                st.caption("⚠️ 참고용: 과거 평균 상승/하락률을 이용한 단순 시나리오로, 실제 시장과 차이가 있을 수 있습니다.")

                scenario_df = simulate_future_12months(state_df, Q)

                if not scenario_df.empty:
                    # 실제 마지막 월 + 시나리오 12개월을 하나의 곡선으로 표시
                    last_row = state_df.iloc[-1]
                    base_price = float(last_row["mean_price"])
                    base_ym = str(last_row["ym"])

                    base_record = pd.DataFrame(
                        [{"ym": base_ym, "scenario_price": base_price, "label": "실제 마지막 월"}]
                    )
                    scen_plot_df = pd.concat(
                        [base_record,
                         scenario_df[["ym", "scenario_price"]].assign(label="시나리오")],
                        ignore_index=True,
                    )

                    scen_plot_df = scen_plot_df.set_index("ym")
                    st.line_chart(scen_plot_df[["scenario_price"]], use_container_width=True)

                    # 시나리오 상세 테이블
                    with st.expander("📋 12개월 시나리오 상세 데이터", expanded=False):
                        st.dataframe(
                            scenario_df[["step", "ym", "predicted_direction_label",
                                         "predicted_action_label", "applied_return", "scenario_price"]],
                            use_container_width=True
                        )
                else:
                    st.info("시나리오 예측을 생성할 수 없습니다. 데이터가 충분한지 확인해주세요.")

            # 탭 4: 상세 로그
            with result_tab4:
                st.subheader("평가 에피소드 상세 로그")
                st.caption("탐욕 정책(greedy policy)으로 1회 실행한 결과입니다.")

                action_map = {0: "하락 예측", 1: "보합 예측", 2: "상승 예측"}
                history_df = history_df.copy()
                history_df["action_label"] = history_df["action_id"].map(action_map)

                show_cols = ["step", "current_ym", "next_ym",
                             "action_label", "true_direction_label", "reward"]
                st.dataframe(history_df[show_cols], use_container_width=True)

    else:
        # 초기 안내 메시지
        st.info("👈 왼쪽 사이드바에서 구, 아파트, 평형을 선택해주세요.")
        
        # 사용 가이드
        with st.expander("📖 사용 가이드", expanded=False):
            st.markdown("""
            **사용 방법:**
            1. 사이드바에서 구, 아파트, 평형을 순서대로 선택합니다.
            2. 학습 에피소드 수를 조정합니다 (기본값: 300).
            3. 'Q-learning 학습/평가 실행' 버튼을 클릭합니다.
            4. 학습 결과를 탭에서 확인합니다.
            
            **주요 기능:**
            - 📊 가격 추이: 월별 평균 실거래가 그래프
            - 📋 데이터 미리보기: 상태(state) 데이터 확인
            - 📊 학습 결과 요약: 핵심 지표 및 학습 곡선
            - 🔮 1개월 예측: 다음 달 가격 방향 예측
            - 📉 12개월 시나리오: 장기 가격 시나리오 (참고용)
            - 📜 상세 로그: 평가 에피소드 상세 내역
            """)


if __name__ == "__main__":
    main()
