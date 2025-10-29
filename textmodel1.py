# -*- coding: utf-8 -*-
"""
CA vs LA — 폐업 식당 부정 리뷰 n-gram 비교 (bi/tri-gram 동시)
- 데이터: business_restaurants.parquet, review_restaurants.parquet
- 절차:
  1) 폐업(is_open=0) + 주(state in ['CA','LA']) 리뷰만 추출
  2) VADER로 감정 라벨링 후 '부정(neg)' 리뷰만 사용
  3) 텍스트 정제 + 불용어/공통 도메인어 제거
  4) CountVectorizer(ngram_range=(2,3))로 bi/tri-gram 카운트
  5) CA/LA 상대 빈도(log-odds with add-1 smoothing)로 비교
  6) 각 주에서 상대적으로 특이한 n-gram Top 30 출력 + CSV 저장
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# ==============================
# 0) 경로/데이터 로드
# ==============================
BASE_DIR = Path(__file__).resolve().parent
business = pd.read_parquet(BASE_DIR / "business_restaurants.parquet")
review   = pd.read_parquet(BASE_DIR / "review_restaurants.parquet")

# ==============================
# 1) 폐업 + 대상 주 필터 (CA, LA)
# ==============================
target_states = ["CA", "LA"]
closed_biz = business[(business["is_open"] == 0) & (business["state"].isin(target_states))][
    ["business_id", "state"]
]
df = review.merge(closed_biz, on="business_id", how="inner")[["business_id","state","text","date"]].copy()
df["text"] = df["text"].astype(str)

print(f"📦 폐업 식당 리뷰 수 — CA: {(df['state']=='CA').sum():,} / LA: {(df['state']=='LA').sum():,}")

# ==============================
# 2) VADER 감정 라벨링 (부정 리뷰만 사용)
# ==============================
# 필요한 리소스 다운로드(최초 1회)
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
POS_TH, NEG_TH = 0.05, -0.05

df["compound"] = df["text"].apply(lambda t: sia.polarity_scores(t)["compound"])
df["sentiment_label"] = np.where(df["compound"] > POS_TH, "pos",
                          np.where(df["compound"] < NEG_TH, "neg", "neu"))

neg = df[df["sentiment_label"]=="neg"].copy()
print(f"☠️  부정 리뷰 수 — CA: {(neg['state']=='CA').sum():,} / LA: {(neg['state']=='LA').sum():,}")

# ==============================
# 3) 전처리 + 커스텀 불용어
# ==============================
stop_en = set(stopwords.words("english"))
# 도메인 공통어(구분력 낮은 일반 토큰) 강제 제거
domain_stop = {
    "food","place","service","time","order","ordered","back","one","us","get","go","like","even","good","would",
    "got","really","just","also","still","bit","much","well","first","second","third","make","made","came","went",
    "im","ive","dont","didnt","wasnt","isnt","cant","couldnt","wont","theyre","youre","thats","its"
}
stop_all = stop_en.union(domain_stop)

def clean_for_ngram(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [w for w in t.split() if (w not in stop_all and len(w) >= 2)]
    return " ".join(tokens)

neg["clean"] = neg["text"].apply(clean_for_ngram)

# ==============================
# 4) bi/tri-gram 벡터화
# ==============================
# min_df: 너무 희귀한 n-gram 제거 (데이터량에 맞춰 조정 가능)
vec = CountVectorizer(ngram_range=(2,3), min_df=20, max_df=0.95)
X = vec.fit_transform(neg["clean"])
terms = np.array(vec.get_feature_names_out())

# ==============================
# 5) 주별 카운트 분해
# ==============================
mask_ca = (neg["state"]=="CA").values
X_ca = X[mask_ca]
X_la = X[~mask_ca]

ca_counts = np.asarray(X_ca.sum(axis=0)).ravel()
la_counts = np.asarray(X_la.sum(axis=0)).ravel()

# ==============================
# 6) log-odds with add-1 smoothing (상대적 선호도)
# ==============================
ca_total = ca_counts.sum()
la_total = la_counts.sum()

# add-1 스무딩(라플라스) + 전체 용어수 보정
V = len(terms)
ca_share = (ca_counts + 1) / (ca_total + V)
la_share = (la_counts + 1) / (la_total + V)

log_odds = np.log(ca_share / la_share)  # +면 CA 쏠림, -면 LA 쏠림

df_ng = pd.DataFrame({
    "ngram": terms,
    "ca_count": ca_counts,
    "la_count": la_counts,
    "ca_share": ca_share,
    "la_share": la_share,
    "log_odds": log_odds
})

# 너무 희귀한 것 제거(각 주 최소 등장 횟수 기준; 필요 시 조정)
df_ng["min_count"] = df_ng[["ca_count","la_count"]].min(axis=1)
df_ng_f = df_ng[df_ng["min_count"] >= 30].copy()  # 예: 각 주 최소 30회 이상 등장한 구절

# ==============================
# 7) 각 주 특이 구절 Top 30
# ==============================
top_CA = (df_ng_f.sort_values("log_odds", ascending=False)
                    .head(30))[["ngram","ca_count","la_count","log_odds"]]
top_LA = (df_ng_f.sort_values("log_odds", ascending=True)
                    .head(30))[["ngram","ca_count","la_count","log_odds"]]

print("\n🔥 CA에서 상대적으로 더 많이 쓰인 부정 구절 (Top 30)")
print(top_CA.to_string(index=False))

print("\n🔥 LA에서 상대적으로 더 많이 쓰인 부정 구절 (Top 30)")
print(top_LA.to_string(index=False))

# ==============================
# 8) CSV 저장(옵션)
# ==============================
out_dir = BASE_DIR / "outputs_ngram"
out_dir.mkdir(exist_ok=True)
top_CA.to_csv(out_dir / "neg_ngrams_CA_top30.csv", index=False)
top_LA.to_csv(out_dir / "neg_ngrams_LA_top30.csv", index=False)
df_ng_f.sort_values("log_odds", ascending=False).to_csv(out_dir / "neg_ngrams_all_logodds.csv", index=False)

print(f"\n📁 저장 완료: {out_dir/'neg_ngrams_CA_top30.csv'}")
print(f"📁 저장 완료: {out_dir/'neg_ngrams_LA_top30.csv'}")
print(f"📁 저장 완료: {out_dir/'neg_ngrams_all_logodds.csv'}")
