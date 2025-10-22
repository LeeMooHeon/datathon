# -*- coding: utf-8 -*-
"""
business 데이터로 가벼운 시각화 (순수 matplotlib / 한글 고정)
리뷰 수(로그) vs 평점 관계
"""
import os, pandas as pd, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 0) 경로 고정 (파일 찾기 에러 방지)
os.chdir(os.path.dirname(__file__))

# 1) 한글 폰트 (테스트에서 쓰던 그대로)
fm.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
print("font.family =", mpl.rcParams['font.family'])  # 확인용

# 2) 데이터 로드
biz = pd.read_json("business_restaurants.json", lines=True)

# 3) 리뷰 수 로그
x = np.log1p(biz['review_count'].astype(float))
y = biz['stars'].astype(float)

# 4) 산점도 + 단순 회귀선(직접 계산)
plt.figure(figsize=(7,5))
plt.scatter(x, y, s=8, alpha=0.25)  # 산점도

# 회귀선
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 200)
plt.plot(xs, m*xs + b, linewidth=2)

# 5) 한글 라벨/제목 (tick까지 폰트 고정)
plt.title("리뷰 수(로그)와 평점의 관계")
plt.xlabel("리뷰 수 (로그 스케일)")
plt.ylabel("평점 (Stars)")
for lab in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    lab.set_fontfamily('Malgun Gothic')

plt.tight_layout()
plt.show()
