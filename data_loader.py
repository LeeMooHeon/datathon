from pathlib import Path
import pandas as pd
import os

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
