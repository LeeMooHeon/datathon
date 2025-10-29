# stability_index.py
from pathlib import Path
import pandas as pd
import numpy as np
import os

# =======================
# 0) 공통 로더 (무헌님 버전)
# =======================
# 🔥 파일이 있는 폴더를 기준으로 경로 고정
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # 현재 작업 폴더를 이 파일이 있는 위치로 변경

def load_dataset(name):
    """
    Yelp Parquet 로더
    name: 'business', 'user', 'tip', 'check_in', 'review'
    """
    pq_path = f"{name}.parquet"

    if os.path.exists(pq_path):
        print(f"📦 {pq_path} 불러오는 중...")
        return pd.read_parquet(pq_path)

    json_path = f"{name}.json"
    if os.path.exists(json_path):
        print(f"🧩 {json_path} → Parquet 변환 중...")
        df = pd.read_json(json_path, lines=True)
        df.to_parquet(pq_path, compression="zstd", index=False)
        print(f"✅ 변환 완료: {pq_path}")
        return df

    raise FileNotFoundError(f"{pq_path} 또는 {json_path} 파일을 찾을 수 없습니다.")

# =======================
# 1) 설정값(초기 파라미터 v1)
# =======================
CONFIG = {
    # Shock(저평점 이벤트) 탐지 파라미터
    "K_PRE": 20,        # 이벤트 직전 참조 리뷰 개수(k)
    "M_EVENT": 3,       # 이벤트 직후 평균에 쓸 리뷰 개수(m)
    "TAU_DROP": 0.8,    # 드롭 폭 임계값(Δ >= tau면 쇼크로 인정)
    # Recovery(회복) 판정 파라미터
    "ROLL_POST": 5,     # 이벤트 후 롤링 평균 계산용 윈도우(개수 기준)
    "DELTA_RECOV": 0.2, # 회복 목표선 허용 오차(|μ_post - μ_pre| <= δ)
    "SUSTAIN_N": 5,     # 목표선 도달 후 유지해야 하는 리뷰 수(S)
    "MAX_DAYS": 180,    # 회복 측정을 위한 최대 추적 일수(검열 한도)
    # 점수 가중치
    "W_TIME": 0.6,      # 회복속도 가중치
    "W_EXTENT": 0.4,    # 회복폭 가중치
    "ALPHA": 0.5,       # 안정지수: (기본신뢰도) vs (회복력) 비중
    # 저신뢰 코호트 최소 리뷰 수
    "MIN_REVIEWS": 25,
}

# =======================
# 2) 유틸
# =======================
def robust_minmax(s: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    """백분위 기반 (5~95%) 강건 정규화: 0~1로 스케일"""
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    s_clip = s.clip(lo, hi)
    den = (hi - lo) if (hi - lo) != 0 else 1.0
    return (s_clip - lo) / den

# =======================
# 3) 리뷰 기반 기본 집계
# =======================
def aggregate_reviews(review: pd.DataFrame) -> pd.DataFrame:
    """
    business_id 단위 기본 통계:
    - stars_mean / stars_std / review_count
    - low_star_count (1~2점) / low_star_ratio
    """
    df = review[['business_id','stars','date']].copy()

    # 날짜 파싱
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['business_id','stars','date'])
    df['is_low'] = (df['stars'] <= 2).astype(int)

    agg = df.groupby('business_id', as_index=False).agg(
        stars_mean=('stars','mean'),
        stars_std=('stars','std'),
        review_count=('stars','count'),
        low_star_count=('is_low','sum')
    )
    agg['stars_std'] = agg['stars_std'].fillna(0.0)
    agg['low_star_ratio'] = agg['low_star_count'] / agg['review_count'].replace(0, np.nan)
    agg['low_star_ratio'] = agg['low_star_ratio'].fillna(0.0)
    return agg

# =======================
# 4) Shock & RecoveryScore 계산 (안전가드 포함)
# =======================
def compute_recovery_score_per_business(df_biz: pd.DataFrame, cfg=CONFIG) -> float:
    """
    단일 business의 리뷰 시계열로부터 RecoveryScore(0~1) 계산.
    (안전가드 포함: 인덱스 경계, 전부 NaN인 롤링 처리)
    """
    if df_biz is None or len(df_biz) == 0:
        return 0.0

    s = df_biz[['stars','date']].dropna().copy()
    if not np.issubdtype(s['date'].dtype, np.datetime64):
        s['date'] = pd.to_datetime(s['date'], errors='coerce')
    s = s.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    if len(s) < max(cfg["MIN_REVIEWS"], cfg["K_PRE"] + cfg["M_EVENT"] + cfg["SUSTAIN_N"]):
        return 0.0  # 데이터 부족

    stars = s['stars'].astype(float).to_numpy()
    dates = s['date'].to_numpy()

    K, M = cfg["K_PRE"], cfg["M_EVENT"]
    tau   = cfg["TAU_DROP"]
    roll  = cfg["ROLL_POST"]
    delta = cfg["DELTA_RECOV"]
    sustain = cfg["SUSTAIN_N"]
    max_days = np.timedelta64(cfg["MAX_DAYS"], 'D')

    # -----------------
    # 1) Shock 탐지
    # -----------------
    shock_idx = None
    mu_pre = None
    mu_event = None

    for i in range(K, len(stars) - M + 1):
        pre   = stars[i-K:i].mean()
        event = stars[i:i+M].mean()
        drop  = pre - event
        if drop >= tau:
            shock_idx = i
            mu_pre = pre
            mu_event = event
            break

    if shock_idx is None:
        return 0.0  # 쇼크 없음

    # -----------------
    # 2) 회복 탐색
    # -----------------
    # 이벤트 이후 구간의 롤링 평균
    post_series = pd.Series(stars[shock_idx:])
    post_vals = post_series.rolling(roll, min_periods=roll).mean().to_numpy()
    post_idx_offset = shock_idx + (roll - 1)  # dates 상의 첫 유효 인덱스

    t0_date = dates[shock_idx]
    t_rec_date = None

    # j는 post_vals에서의 인덱스 (post_idx_offset + j) 가 dates 인덱스를 넘지 않게 체크
    for j in range(len(post_vals)):
        if np.isnan(post_vals[j]):
            continue

        idx_global = post_idx_offset + j
        if idx_global >= len(dates):
            break  # 경계 보호

        mu_post = post_vals[j]

        # 회복 목표선 도달?
        if abs(mu_post - mu_pre) <= delta:
            # 지속성 체크: 이후 sustain-1개도 조건 만족?
            ok = True
            cnt = 1
            k = j + 1
            while cnt < sustain:
                if k >= len(post_vals):
                    ok = False
                    break
                if np.isnan(post_vals[k]):
                    ok = False
                    break

                idx_global_k = post_idx_offset + k
                if idx_global_k >= len(dates):
                    ok = False
                    break

                if abs(post_vals[k] - mu_pre) > delta:
                    ok = False
                    break

                cnt += 1
                k += 1

            if ok:
                t_rec_date = dates[idx_global]
                break

        # 검열(시간 한도)
        idx_for_time = min(shock_idx + j, len(dates)-1)
        if (dates[idx_for_time] - t0_date) > max_days:
            break

    # -----------------
    # 3) 속도 성분 S_time
    # -----------------
    if t_rec_date is not None:
        days = (t_rec_date - t0_date) / np.timedelta64(1, 'D')
    else:
        days = cfg["MAX_DAYS"]  # 미회복 → 최대일수로 취급
    S_time = 1.0 / (1.0 + days / 60.0)  # 60일 스케일링

    # -----------------
    # 4) 탄성 성분 S_elas (전부 NaN 방지)
    # -----------------
    delta_drop = mu_pre - mu_event
    horizon_end = t0_date + max_days
    horizon_mask = (dates >= t0_date) & (dates <= horizon_end)

    post_h = stars[horizon_mask]
    if post_h.size >= roll:
        post_h_roll = pd.Series(post_h).rolling(roll, min_periods=roll).mean().to_numpy()
        valid = post_h_roll[~np.isnan(post_h_roll)]
        if valid.size > 0:
            mu_max = float(valid.max())
            S_elas = max(0.0, min(1.0, (mu_max - mu_event) / max(delta_drop, 1e-6)))
        else:
            S_elas = 0.0
    else:
        S_elas = 0.0

    # -----------------
    # 5) 결합
    # -----------------
    rec = CONFIG["W_TIME"] * S_time + CONFIG["W_EXTENT"] * S_elas
    return float(np.clip(rec, 0.0, 1.0))

def compute_recovery_scores(review: pd.DataFrame) -> pd.DataFrame:
    """business_id별 RecoveryScore(0~1) 계산"""
    df = review[['business_id','stars','date']].copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['business_id','stars','date'])

    scores = []
    for bid, grp in df.groupby('business_id', sort=False):
        rec = compute_recovery_score_per_business(grp, CONFIG)
        scores.append((bid, rec))
    return pd.DataFrame(scores, columns=['business_id','RecoveryScore'])

# =======================
# 5) StableIndex 산출
# =======================
def build_stable_index(business: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """
    StableIndex 생성 파이프라인
    1) 리뷰 집계
    2) 회복 점수
    3) 기본신뢰도(Base) & 패널티 & 정규화
    4) 안정지수 결합
    """
    agg = aggregate_reviews(review)
    rec = compute_recovery_scores(review)

    # 병합
    out = agg.merge(rec, on='business_id', how='left')
    out['RecoveryScore'] = out['RecoveryScore'].fillna(0.0)

    # 기본신뢰도 구성
    out['Base'] = out['stars_mean'] * np.log1p(out['review_count'])
    # 변동성 패널티
    out['PenaltyVar'] = 1.0 / (1.0 + out['stars_std'])
    # 저평점 패널티
    out['PenaltyLow'] = 1.0 - out['low_star_ratio']

    # 종합 베이스
    out['BaseScore'] = out['Base'] * out['PenaltyVar'] * out['PenaltyLow']

    # 강건 정규화(0~1)
    out['BaseScore_norm'] = robust_minmax(out['BaseScore'])
    out['RecoveryScore_norm'] = out['RecoveryScore']  # 이미 0~1

    # 안정 지수
    a = CONFIG["ALPHA"]
    out['StableIndex'] = a * out['BaseScore_norm'] + (1 - a) * out['RecoveryScore_norm']

    # 신뢰도 태그(리뷰수 부족)
    out['low_data_flag'] = (out['review_count'] < CONFIG["MIN_REVIEWS"]).astype(int)

    # business 기본 메타와 합치기(존재하는 컬럼만)
    cols_meta = ['business_id','name','state','is_open','categories']
    meta_cols = [c for c in cols_meta if c in business.columns]
    out = out.merge(business[meta_cols], on='business_id', how='left')

    # 컬럼 정리
    keep = [
        'business_id','name','state','is_open','categories',
        'review_count','stars_mean','stars_std','low_star_count','low_star_ratio',
        'Base','PenaltyVar','PenaltyLow','BaseScore','BaseScore_norm',
        'RecoveryScore','StableIndex','low_data_flag'
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values('StableIndex', ascending=False).reset_index(drop=True)

# =======================
# 6) 실행 스크립트
# =======================
if __name__ == "__main__":
    # 이미 계산된 파일이 있으면 불러오기 (옵션)
    if os.path.exists("business_stability.parquet"):
        print("📂 기존 안정 지수 파일 발견 → 불러옵니다.")
        result = pd.read_parquet("business_stability.parquet")
    else:
        business = load_dataset("business")
        review = load_dataset("review")
        print("⚙️ 안정 지수 계산 중...")
        result = build_stable_index(business, review)

        # 저장 (전체 Parquet + 전체 CSV + 상위 50 CSV)
        result.to_parquet("business_stability.parquet", compression="zstd", index=False)
        result.to_csv("business_stability.csv", index=False, encoding="utf-8-sig")  # ✅ 전체 CSV
        result.head(50).to_csv("business_stability_head50.csv", index=False, encoding="utf-8-sig")

    print("✅ 안정 지수 준비 완료 → business_stability.parquet / business_stability.csv / business_stability_head50.csv")
    print(result.head())
