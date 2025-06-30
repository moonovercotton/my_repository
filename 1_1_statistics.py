import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import collections
from sklearn.cluster import KMeans

# 中文字体设置
font_path = "/home/liuyunlong/.fonts/NotoSansCJK-Black.ttc"
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

# 文件路径
root_path = "./data"
data_path = "raw_hourly_data.csv"

# 读取数据
df_raw = pd.read_csv(os.path.join(root_path, data_path))
df_raw["date"] = pd.to_datetime(df_raw["date"])

# 设定时间范围
start_date = pd.to_datetime("2023-03-01 00:00:00")
end_date = pd.to_datetime("2024-08-11 23:00:00")
df_all_days = df_raw[(df_raw["date"] >= start_date) & (df_raw["date"] <= end_date)]

price = df_all_days['历史日前出清价格']

# 1. 准备数据
prices_by_day = price.values.reshape(-1, 24)      # (n_days, 24)
n_days = prices_by_day.shape[0]

# 2. 聚类：这里假设分成 10 个簇，可根据需要调整
k = 10
kmeans = KMeans(n_clusters=k, random_state=42).fit(prices_by_day)
labels = kmeans.labels_                            # 每天对应的簇编号 0~k-1

# 3. 统计簇大小
from collections import Counter
cluster_counts = Counter(labels)
max_count = max(cluster_counts.values())

# 4. 绘制
plt.figure(figsize=(12, 6))

# 使用 colormap（从浅到深蓝）
cmap = plt.get_cmap('Blues')

for day_idx, day_curve in enumerate(prices_by_day):
    cnt = cluster_counts[labels[day_idx]]
    weight = cnt / max_count                       # 归一化到 [0,1]
    alpha = 0.6                                     # 固定透明度
    lw    = 0.5 + 2.5 * weight                      # 根据簇大小调整线宽
    color = cmap(0.3 + 0.7 * weight)                # 调整颜色深度（避免太浅）
    plt.plot(range(24), day_curve, color=color, alpha=alpha, linewidth=lw)

plt.title("基于 KMeans 聚类后按簇大小用蓝色深浅绘制的每日价格曲线", fontsize=14)
plt.xlabel("小时", fontsize=12)
plt.ylabel("价格", fontsize=12)
plt.grid(True)
plt.xticks(np.arange(0, 24, 1))
plt.tight_layout()
plt.savefig('历史日前出清价格-每天-聚类-蓝色渐变')
plt.show()

