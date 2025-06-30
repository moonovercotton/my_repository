import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

# 中文字体设置
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"
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

# 初始化最大比例信息
max_ratio = -1
best_x, best_y = None, None

# 保存比例大于90%的所有XY及其比例
high_ratio_results = []

# 遍历 X 和 Y
for x in range(20000, 20001, 100):
    high_load_dates = df_all_days[df_all_days["日前风电光伏总负荷"] > x]["date"].unique()
    if len(high_load_dates) == 0:
        continue

    for y in range(50, 51, 1):
        target_prices = df_all_days[df_all_days["date"].isin(high_load_dates)]["历史日前出清价格"]
        if len(target_prices) == 0:
            continue

        ratio = (target_prices <= y).sum() / len(target_prices)

        print(ratio)
        exit(0)

        # 保存最大比例及对应X、Y
        if ratio > max_ratio:
            max_ratio = ratio
            best_x, best_y = x, y

        # 保存大于90%的 (X, Y, ratio)
        if ratio > 0.6:
            high_ratio_results.append((x, y, ratio))

# 查找比例大于90%中最小的比例对应的 X 和 Y
if high_ratio_results:
    min_high_ratio_entry = min(high_ratio_results, key=lambda item: item[2])
    min_ratio_x, min_ratio_y, min_ratio_val = min_high_ratio_entry
    print(f"比例 > 90% 中最小的比例为: {min_ratio_val:.4f}，对应的 X = {min_ratio_x}, Y = {min_ratio_y}")
    # 比例 > 90% 中最小的比例为: 0.9004，对应的 X = 16700, Y = 185
else:
    print("没有找到比例大于90%的 X 和 Y")

# 输出最大比例
print(f"最大比例为: {max_ratio:.4f}，对应的 X = {best_x}, Y = {best_y}")



# # 统计“日前风电光伏总负荷”大于20000的时刻数
# count_exceed_20000 = (df_45days["日前风电光伏总负荷"] > 20000).sum()
# print(f"‘日前风电光伏总负荷’大于20000的时刻数为：{count_exceed_20000}")
# print(f"‘日前风电光伏总负荷’大于20000的百分比为：{count_exceed_20000 / df_45days.shape[0] * 100}")

# # 统计“历史日前出清价格”小于100的时刻数
# count_price_below_100 = (df_45days["历史日前出清价格"] < 100).sum()
# print(f"‘历史日前出清价格’小于100的时刻数为：{count_price_below_100}")
