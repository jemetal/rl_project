# src/menu_select.py

def select_from_list(options, title):
    """
    공통 선택 메뉴 함수.
    options: 선택지 리스트
    title: 화면 상단에 표시할 제목 문자열
    """
    if not options:
        print("[ERROR] 선택지가 없습니다.")
        return ""

    while True:
        print(f"\n[{title}]")
        for i, item in enumerate(options, 1):
            print(f"{i}. {item}")

        choice = input("\n번호 선택: ").strip()

        # 숫자인지 확인
        if not choice.isdigit():
            print("숫자를 입력해 주세요.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]

        print("잘못된 입력입니다. 다시 선택해 주세요.")


def select_gu_menu(df):
    """
    1단계: 구 선택 메뉴
    """
    if "gu" not in df.columns:
        print("[ERROR] 'gu' 컬럼이 없습니다. 데이터 컬럼명을 확인해 주세요.")
        return ""

    gu_list = sorted(df["gu"].dropna().unique().tolist())
    return select_from_list(gu_list, "1단계: 구 선택")


def select_apt_menu(df, selected_gu):
    """
    2단계: 선택한 구 내의 아파트 단지 선택 메뉴
    """
    if {"gu", "apt_name"}.issubset(df.columns) is False:
        print("[ERROR] 'gu' 또는 'apt_name' 컬럼이 없습니다.")
        return ""

    sub = df[df["gu"] == selected_gu]
    apt_list = sorted(sub["apt_name"].dropna().unique().tolist())

    if not apt_list:
        print(f"[ERROR] {selected_gu} 내에 아파트 데이터가 없습니다.")
        return ""

    return select_from_list(apt_list, f"2단계: {selected_gu} 아파트 선택")


def select_area_menu(df, selected_gu, selected_apt):
    """
    3단계: 선택한 (구, 아파트)에 대해 평형(전용면적) 선택 메뉴

    - 데이터에서 해당 구 + 아파트에 해당하는 행만 필터링한 뒤
      'area' 컬럼의 고유값 목록을 보여줌.
    - area 값이 숫자(전용면적 m²)일 수도 있고, '32평형' 같은 문자열일 수도 있음.
    """
    if {"gu", "apt_name", "area"}.issubset(df.columns) is False:
        print("[ERROR] 'gu', 'apt_name', 'area' 컬럼이 없습니다.")
        return ""

    sub = df[(df["gu"] == selected_gu) & (df["apt_name"] == selected_apt)]

    if sub.empty:
        print(f"[ERROR] {selected_gu} {selected_apt} 데이터가 없습니다.")
        return ""

    area_list = sub["area"].dropna().unique().tolist()

    # 정렬: 숫자/문자 섞여 있을 수 있으므로 문자열 기준 정렬
    area_list = sorted(area_list, key=lambda x: str(x))

    return select_from_list(area_list, f"3단계: {selected_gu} {selected_apt} 평형 선택")
