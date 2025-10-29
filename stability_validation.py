# stability_validation.py
from pathlib import Path
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
from scipy.stats import ttest_ind

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

PATH_PARQUET = "business_stability.parquet"  # <-- 여기서 Parquet 읽음

def load_parquet(path=PATH_PARQUET) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' 파일이 없습니다. 안정 지수 먼저 계산/저장하세요.")
    print(f"📦 {path} 불러오는 중...")
    return pd.read_parquet(path)

def main():
    df = load_parquet()

    # 필수 컬럼 체크
    needed = ["StableIndex", "is_open", "stars_mean", "review_count", "stars_std", "low_star_ratio"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 없음: {missing}")

    # 기본 통계
    print("\n=== [1] 기본 통계 ===")
    print(df["StableIndex"].describe())

    print("\n=== [2] 상관 행렬 (StableIndex vs 기본 지표) ===")
    print(df[["StableIndex","stars_mean","review_count","stars_std","low_star_ratio"]].corr(numeric_only=True))

    # 집단 간 평균 차이: 생존(1) vs 폐업(0)
    print("\n=== [3] 집단 평균 차이 검정 (t-test) : is_open 1 vs 0 ===")
    alive = df.loc[df["is_open"]==1, "StableIndex"].dropna()
    closed = df.loc[df["is_open"]==0, "StableIndex"].dropna()
    if len(alive) > 5 and len(closed) > 5:
        stat, p = ttest_ind(alive, closed, equal_var=False)
        print(f"t={stat:.3f}, p={p:.6f}  (p<0.05면 유의차)")
        print(f"mean(alive)={alive.mean():.4f}, mean(closed)={closed.mean():.4f}")
    else:
        print("샘플이 부족해 t-test를 건너뜁니다.")

    # 로지스틱 회귀: 폐업(=1) 예측력
    print("\n=== [4] 로지스틱 회귀: 폐업(=1) ~ StableIndex ===")
    y = (df["is_open"]==0).astype(int)  # 폐업 라벨
    X = sm.add_constant(df[["StableIndex"]])
    try:
        model = sm.Logit(y, X, missing="drop").fit(disp=False)
        print(model.summary())
    except Exception as e:
        print(f"로지스틱 회귀 실패: {e}")

    # 다중 공변량 포함: 독립효과 확인
    print("\n=== [5] 다중 로지스틱: 폐업(=1) ~ StableIndex + stars_mean + review_count + stars_std + low_star_ratio ===")
    X2 = df[["StableIndex","stars_mean","review_count","stars_std","low_star_ratio"]].copy()
    X2 = sm.add_constant(X2)
    try:
        model2 = sm.Logit(y, X2, missing="drop").fit(disp=False)
        print(model2.summary())
    except Exception as e:
        print(f"다중 로지스틱 회귀 실패: {e}")

    # 분위수별 폐업률 비교 (디실)
    print("\n=== [6] StableIndex 분위수별(Decile) 폐업률 ===")
    tmp = df[["StableIndex","is_open"]].dropna().copy()
    tmp["decile"] = pd.qcut(tmp["StableIndex"], 10, labels=False, duplicates="drop")  # 0=하위 → 9=상위
    dec = tmp.groupby("decile").apply(lambda g: pd.Series({
        "count": len(g),
        "mean_stable": g["StableIndex"].mean(),
        "close_rate": (g["is_open"]==0).mean()
    })).reset_index()
    print(dec.sort_values("decile"))

    # 상/하위 10% 요약
    if "decile" in tmp.columns:
        bottom = dec.loc[dec["decile"]==dec["decile"].min(), "close_rate"].values
        top = dec.loc[dec["decile"]==dec["decile"].max(), "close_rate"].values
        if len(bottom) and len(top):
            print(f"\n하위 10% 폐업률: {bottom[0]:.3%}  |  상위 10% 폐업률: {top[0]:.3%}  |  리프트: {(bottom[0]-top[0]):.3%}")

if __name__ == "__main__":
    main()
