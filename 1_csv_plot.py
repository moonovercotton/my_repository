import os
import pandas as pd
import matplotlib.pyplot as plt


# 读取CSV文件，并将日期列转换为datetime类型
csv_file = './data/raw_hourly_data.csv'
df = pd.read_csv(csv_file, parse_dates=['date'])

# 绘图
l = 0
r = l + 7 * 30 * 24
plt.figure(figsize=(12, 6))
# 历史实时出清价格, 日前风电光伏总负荷, 日前联络线计划, 日前统调负荷
target = '历史实时出清价格'
plt.plot(df['date'][l:r], df[target][l:r], marker='o', linestyle='-', markersize=2)
plt.xlabel('date')
# plt.ylabel('历史实时出清价格')
# plt.title('历史实时出清价格走势图')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 确保保存目录存在
output_dir = './plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图像为 PDF 文件
output_path = os.path.join(output_dir, f'{target}.pdf')
plt.savefig(output_path)

# 如果需要显示图像，也可以调用 plt.show()
plt.show()

print(f"图像已保存至: {output_path}")
