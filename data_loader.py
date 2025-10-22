
from pathlib import Path
import pandas as pd
import os

# 🔥 파일이 있는 폴더를 기준으로 경로 고정
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # 현재 작업 폴더를 이 파일이 있는 위치로 변경

def load_dataset(name):
    json_path = BASE_DIR / f"{name}_restaurants.json"
    pq_path   = BASE_DIR / f"{name}_restaurants.parquet"
    ...

def load_dataset(name):
    """
    Yelp JSON → Parquet 캐시 자동 변환 로더
    name: 'business', 'user', 'tip', 'checkin'
    """
    json_path = f"{name}_restaurants.json"
    pq_path = f"{name}_restaurants.parquet"

    if os.path.exists(pq_path):
        print(f"📦 {pq_path} 불러오는 중...")
        return pd.read_parquet(pq_path)

    print(f"🧩 {json_path} → Parquet 변환 중...")
    df = pd.read_json(json_path, lines=True)
    df.to_parquet(pq_path, compression="zstd", index=False)
    print(f"✅ 변환 완료: {pq_path}")
    return df
