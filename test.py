import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取数据
df = pd.read_csv('./data/raw_hourly_data.csv', parse_dates=['date'], index_col='date')

# 过滤时间范围
start_time = '2023-01-01 00:00:00'
end_time = '2024-08-12 23:00:00'
df_filtered = df.loc[start_time:end_time]

# 取出目标列
price_series = df_filtered['历史日前出清价格']

# 固定横轴长度：原始时间范围
x_full = price_series.index

# 设置最大聚合间隔（小时）
max_interval = 64  # 即最多按每64小时聚合
interval = 1

plot_id = 1

while interval <= max_interval:
    # 重设索引为整数便于分组
    df_group = price_series.reset_index()
    
    # 分组：每 interval 行作为一组，计算平均值
    df_group['group'] = df_group.index // interval
    grouped = df_group.groupby('group').agg({
        'date': 'first',  # 取每组第一个时间作为代表
        '历史日前出清价格': 'mean'
    })

    # 提取时间和平均值
    x_sampled = grouped['date']
    y_sampled = grouped['历史日前出清价格']

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.plot(x_sampled, y_sampled, label=f'每 {interval} 点平均')
    plt.title(f'出清价格（每 {interval} 点平均）')
    plt.xlabel('时间')
    plt.ylabel('平均价格')
    plt.grid(True)
    
    # 保持横轴一致
    plt.xlim(x_full[0], x_full[-1])
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./test/clearing_price_avg_interval_{interval}.png')
    plt.show()

    interval *= 2
    plot_id += 1
