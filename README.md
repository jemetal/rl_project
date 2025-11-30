# 🏙 RL Housing Project  
서울 아파트 가격 방향 예측 (Q-learning + Streamlit)

이 프로젝트는 **서울시 아파트 실거래가 / 한국은행 기준금리 / 각 구별 인구 데이터** 를 이용하여  
특정 아파트·평형의 **다음 달 가격 방향(상승/보합/하락)** 을 강화학습(Q-learning)으로 예측하고,  
Streamlit 대시보드로 결과를 시각화하는 데모입니다.

---

## 1. 프로젝트 개요

- **목표**
    - 실거래가 시계열 + 기준금리 + 인구 데이터를 이용해
    - 다음 달 가격이 **상승 / 보합 / 하락** 중 무엇일지 예측
    - 학습된 정책을 바탕으로 **12개월 시나리오 가격 곡선** 생성
- **핵심 아이디어**
  - 상태(state):  
    가격 방향(direction) + 기준금리 수준(rate_level) + 인구 추세(pop_trend)
  - 행동(action):  
    다음 달 가격을 하락/보합/상승 중 하나로 예측
  - 보상(reward):  
    예측이 맞으면 +1, 틀리면 -1  
  - 알고리즘:  
    Tabular Q-learning + ε-greedy 탐색

---

## 2. 폴더 구조

프로젝트 기본 구조는 다음과 같습니다.

rl_project/
├─ app_streamlit.py     # Streamlit 메인 앱
├─ src/
│  ├─ data_loader.py    # 원본 데이터 로딩
│  ├─ preprocess.py     # 아파트 필터링, 월별 패널 생성
│  ├─ macro_features.py # 기준금리/인구 처리 및 macro 상태 생성
│  ├─ state_encoder.py  # (direction, rate_level, pop_trend) → state index
│  ├─ environment.py    # HousingEnv (강화학습 환경)
│  ├─ qlearning.py      # Q-learning 알고리즘 & 평가
│  └─ api.py            # Streamlit에서 사용하는 고수준 API 함수
└─ data/
   ├─ 1_주민등록인구_20251129142612.xlsx
   ├─ 2_기준금리_한국은행 기준금리 및 여수신금리_29140618.xlsx
   └─ 3_(전체)아파트(매매)_실거래가_20251129130725.xlsx
````

> 'data/' 폴더의 엑셀 파일은 개인 환경에서 준비한 파일입니다.
> 실행 시에는 동일한 파일명을 사용하거나, 소스 코드에서 경로를 수정해 주세요.

---

## 3. 실행 환경

* Python 3.13.2 (로컬 venv 권장)
* 주요 패키지

  * pandas
  * numpy
  * streamlit
  * (그 외: matplotlib등 일부 분석용 패키지)


---

## 4. 설치 방법

### 4-1. 저장소 클론

```bash
git clone https://github.com/사용자ID/rl_project.git
cd rl_project
```

### 4-2. 가상환경 생성 (선택)

```bash
# Windows (PowerShell 기준)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4-3. 패키지 설치

```bash
pip install streamlit pandas numpy
# 필요 시 다른 패키지도 추가 설치
```

### 4-4. 데이터 파일 위치

프로젝트 루트에 `data/` 폴더를 만들고, 아래 파일들을 넣습니다.

* `data/1_주민등록인구_20251129142612.xlsx`
* `data/2_기준금리_한국은행 기준금리 및 여수신금리_29140618.xlsx`
* `data/3_(전체)아파트(매매)_실거래가_20251129130725.xlsx`

파일명이 다르다면 `src/data_loader.py` / `src/api.py` 내 경로를 수정해주세요.

---

## 5. 실행 방법

프로젝트 루트( `app_streamlit.py` 가 있는 위치 )에서 아래 명령을 실행합니다.

```bash
streamlit run app_streamlit.py
```

실행하면 기본 웹 브라우저가 열리거나,
터미널에 다음과 비슷한 주소가 표시됩니다.

```text
Local URL: http://localhost:8501
```

브라우저에서 해당 주소로 접속하면 대시보드 화면을 볼 수 있습니다.

---

## 6. 사용 방법 (대시보드)

1. **사이드바에서 아파트 선택**

   * `구 선택`: 예) 강동구, 마포구 …
   * `아파트 선택`: 선택한 구에 존재하는 단지 목록이 자동으로 뜹니다.
   * `평형 선택`: 해당 단지의 전용면적(㎡) 목록에서 선택
   * `학습 에피소드 수`: 슬라이더로 Q-learning 반복 횟수 조절 (예: 300)

2. **메인 화면 확인**

   * 선택한 아파트 정보(구 / 단지명 / 평형)
   * 월별 평균 실거래가 라인 그래프
   * 강화학습에 사용되는 state 데이터 미리보기

3. **강화학습 실행**

   * `🚀 Q-learning 학습/평가 실행` 버튼 클릭
   * 학습 후 다음 내용을 확인할 수 있습니다.

     * 총 보상, 정답률, 스텝 수
     *** 미래 1개월 가격 방향 예측 결과**
         예) 기준 월 2025-08 → 예측 대상 월 2025-09 은(는) 상승으로 예상됩니다.
     *** 12개월 시나리오 가격 곡선**
       * 과거 평균 상승/하락률을 이용한 가상의 가격 경로 (보너스_참고용)
     * 에피소드별 총 보상 변화 그래프
     * 평가 에피소드(탐욕 정책 1회) step-by-step 상세 로그

> (참고)12개월 시나리오 곡선은 **실제 가격 예측이 아니라**,
> 학습된 정책과 과거 평균 수익률을 바탕으로 그려본 “가상 시나리오”입니다.

---

## 7. 한계 및 TODO

* 현재는 **단일 아파트·단일 평형**에 대해서만 학습/예측을 수행합니다.
* Tabular Q-learning 한 가지 알고리즘만 사용하며,
  여러 seed / 여러 hyperparameter 설정에 대한 통계적 비교는 아직 없습니다.

* 향후 계획
  * 여러 단지를 동시에 학습하는 일반화 실험
  * Q-learning vs 다른 RL 알고리즘 비교(SARSA, DQN 등)
  * state에 거래량, 변동성 등 추가 특성 반영
  * 더 다양한 인터랙티브 기능이 있는 Streamlit 대시보드로 확장

---

## 8. 소속및 작성자

소속 : 서강대 AI.SW대학원 
과목명 : 강화학습의 기초(소정민 교수님)
작성자 : 데이터사이언스 인공지능학과 백지은

```