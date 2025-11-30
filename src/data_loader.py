import pandas as pd

def load_transaction_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print("[ERROR] 파일 로딩 실패:", e)
        return pd.DataFrame()

    print("[INFO] 실거래가 데이터 미리보기:")
    print(df.head())
    print(df.columns)

    rename_map = {
        "구": "gu",
        "동": "dong",
        "아파트명": "apt_name",
        "평형": "area",
        "계약년": "year",
        "계약월": "month",
        "계약일": "day",
        "거래금액(만원)": "price_10k",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "price_10k" in df.columns:
        df["price_10k"] = (
            df["price_10k"].astype(str).str.replace(",", "").astype(int)
        )

    if {"year", "month", "day"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
        df["ym"] = df["date"].dt.strftime("%Y-%m")

    return df
