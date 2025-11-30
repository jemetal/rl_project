# src/state_encoder.py
# ---------------------------------------------
# 가격 방향(direction), 금리수준, 인구추세를
# 상태 ID로 변환하는 유틸 모음
# ---------------------------------------------

# 방향(direction) → 상태 ID 매핑
# -1: 하락, 0: 보합, 1: 상승
DIRECTION_TO_ID = {
    -1: 0,
    0: 1,
    1: 2,
}

# 상태 ID → 방향(direction) 역매핑
ID_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_ID.items()}

# 인구추세(pop_trend: -1,0,1) → ID 매핑
POP_TREND_TO_ID = {
    -1: 0,  # 감소
    0: 1,   # 보합
    1: 2,   # 증가
}
ID_TO_POP_TREND = {v: k for k, v in POP_TREND_TO_ID.items()}


def encode_direction(direction: int) -> int:
    """
    가격 방향(-1, 0, 1)을 상태 ID(0, 1, 2)로 변환.
    혹시 이상한 값이 들어오면 보합(0)으로 처리.
    """
    try:
        d = int(direction)
    except Exception:
        d = 0
    return DIRECTION_TO_ID.get(d, 1)  # 기본값: 보합(0) → state_id=1


def decode_direction(state_id: int) -> int:
    """
    상태 ID(0, 1, 2)를 가격 방향(-1, 0, 1)으로 변환.
    """
    try:
        s = int(state_id)
    except Exception:
        s = 1
    return ID_TO_DIRECTION.get(s, 0)


def describe_direction(direction: int) -> str:
    """
    방향(-1, 0, 1)을 사람이 읽기 쉬운 문자열로 변환.
    """
    d = int(direction)
    if d == -1:
        return "하락"
    elif d == 1:
        return "상승"
    else:
        return "보합"


# ===== 금리 수준 / 인구 추세 인코딩 =====

def encode_rate_level(level: int) -> int:
    """
    금리 수준(level)을 0~2 ID로 변환.
    level 자체를 0/1/2로 계산해 두었기 때문에
    0~2 범위로 클리핑만 해준다.
    """
    try:
        x = int(level)
    except Exception:
        x = 1
    if x < 0:
        x = 0
    if x > 2:
        x = 2
    return x


def encode_pop_trend(trend: int) -> int:
    """
    인구 추세(-1,0,1)를 0~2 ID로 변환.
    """
    try:
        t = int(trend)
    except Exception:
        t = 0
    return POP_TREND_TO_ID.get(t, 1)  # 기본값: 보합(0) → ID=1


# ===== 전체 state 인코딩 =====

def encode_state(direction: int, rate_level: int, pop_trend: int) -> int:
    """
    3차원 상태 (direction, rate_level, pop_trend)를
    단일 상태 ID(0~26)로 변환.

    각 차원의 ID 범위:
      - dir_id  : 0~2  (하락/보합/상승)
      - rate_id : 0~2  (낮음/중간/높음)
      - pop_id  : 0~2  (감소/보합/증가)

    state_id = dir_id * 9 + rate_id * 3 + pop_id
    """
    dir_id = encode_direction(direction)      # 0~2
    rate_id = encode_rate_level(rate_level)   # 0~2
    pop_id = encode_pop_trend(pop_trend)      # 0~2

    state_id = dir_id * 9 + rate_id * 3 + pop_id
    return int(state_id)
