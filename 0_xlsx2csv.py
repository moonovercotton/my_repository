import pandas as pd
from datetime import timedelta
import numpy as np
import sys

# 读取频率为15min的变量
# 历史日前出清价格、日前联络线计划
# 历史实时出清价格
def read_15min_variate(input_path, col_name, interpolate=False):
    # 读取 Excel 文件
    df = pd.read_excel(input_path)

    # 假设第一列为日期，将其转换为 datetime 类型
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # 按日期排序
    df = df.sort_values(by=date_col)

    # 用于存储转换后的数据
    rows = []

    # 对每一天的数据进行转换
    for _, row in df.iterrows():
        base_date = row[date_col]
        # 针对每个数据点（假设后面 96 列为数据点）
        for i, col in enumerate(df.columns[1:]):
            # 计算对应的时间：基础日期加上 i * 15 分钟
            timestamp = base_date + timedelta(minutes=15 * i)
            value = row[col]
            rows.append({'date': timestamp, col_name: value})

    # 构建新的 DataFrame
    new_df = pd.DataFrame(rows)
    # 按时间排序
    new_df = new_df.sort_values(by='date')

    if interpolate:
        # 使用线性插值对第二列中的0值进行替换
        new_df[col_name] = new_df[col_name].replace(0, float('nan'))  # 将0值替换为NaN
        new_df[col_name] = new_df[col_name].interpolate(method='linear')  # 线性插值


    return new_df


# 读取日前新能源负荷
def read_day_ahead_renewable_energy_load(input_path):
    # 读取 Excel 文件
    df = pd.read_excel(input_path)

    # 假设第一列为日期，将其转换为 datetime 类型
    date_col = df.columns[0]
    type_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])

    # 按日期排序
    df = df.sort_values(by=date_col)

    # 用于存储转换后的数据
    rows = []

    # 对每一天的数据进行转换
    for _, row in df.iterrows():
        base_date = row[date_col]
        data_type = row[type_col]
        # 针对每个数据点（假设后面 96 列为数据点）
        for i, col in enumerate(df.columns[2:]):
            # 计算对应的时间：基础日期加上 i * 15 分钟
            timestamp = base_date + timedelta(minutes=15 * i)
            value = row[col]
            if data_type == 'ALL':
                col_name = '日前风电光伏总负荷'
            elif data_type == 'WIND':
                col_name = '日前风电负荷'
            elif data_type == 'PHOTOVOLTAIC':
                col_name = '日前光伏负荷'
            rows.append({'date': timestamp, col_name: value})

    # 构建新的 DataFrame
    new_df = pd.DataFrame(rows)
    # 按时间排序
    new_df = new_df.sort_values(by='date')
    # 将同一时间点的三行合并
    df_merged = new_df.groupby("date", as_index=False).first()

    return df_merged

# 读取日前检修总容量
def read_total_capacity_of_maintenance_recently(input_path1, input_path2):
    # 读取 Excel 文件
    df1 = pd.read_excel(input_path1)
    df2 = pd.read_excel(input_path2)

    # 假设第一列为日期，将其转换为 datetime 类型
    date_col = df1.columns[0]
    df1[date_col] = pd.to_datetime(df1[date_col])
    df2[date_col] = pd.to_datetime(df2[date_col])

    # 按日期排序
    df1 = df1.sort_values(by=date_col)
    df2 = df2.sort_values(by=date_col)

    # 用于存储转换后的数据
    rows1 = []
    rows2 = []

    # 对每一天的数据进行转换
    for _, row in df1.iterrows():
        base_date = row[date_col]
        # 针对每个数据点（假设后面 96 列为数据点）
        for i, col in enumerate(df1.columns[1:]):
            # 计算对应的时间：基础日期加上 i * 15 分钟
            timestamp = base_date + timedelta(minutes=15 * i)
            value = row[col]
            rows1.append({'date': timestamp, '日前检修总容量': value})

    for _, row in df2.iterrows():
        base_date = row[date_col]
        # 针对每个数据点（假设后面 96 列为数据点）
        for i, col in enumerate(df2.columns[1:]):
            # 计算对应的时间：基础日期加上 i * 15 分钟
            timestamp = base_date + timedelta(minutes=15 * i)
            value = row[col]
            rows2.append({'date': timestamp, '日前检修总容量': value})

    # 构建新的 DataFrame
    new_df1 = pd.DataFrame(rows1)
    new_df2 = pd.DataFrame(rows2)
    # 按时间排序
    new_df1 = new_df1.sort_values(by='date')
    new_df2 = new_df2.sort_values(by='date')

    # 设置 date 为索引
    new_df1.set_index('date', inplace=True)
    new_df2.set_index('date', inplace=True)

    # 数据纠正
    new_df1.update(new_df2)

    return new_df1

# 读取日前正负备用
def read_day_ahead_positive_negative_reserve(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 重新排列数据格式
    df_pivot = df.pivot(index='日期', columns='备用类型', values='备用负荷').fillna(0)
    df_pivot.reset_index(inplace=True)
    df_pivot.columns.name = None  # 去掉列索引的名称
    
    # 生成时间序列（96个15分钟间隔）
    time_intervals = pd.date_range("00:00", "23:45", freq="15min").time
    
    # 扩展数据
    expanded_data = []
    for _, row in df_pivot.iterrows():
        date = row['日期']
        for time in time_intervals:
            timestamp = pd.Timestamp(f"{date} {time}")
            expanded_data.append([timestamp] + list(row[1:]))
    
    # 创建新的 DataFrame
    new_columns = ['date'] + list(df_pivot.columns[1:])
    result_df = pd.DataFrame(expanded_data, columns=new_columns)
    
    return result_df

def read_real_time_reserve_amount(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 生成时间序列（96个15分钟间隔）
    time_intervals = pd.date_range("00:00", "23:45", freq="15min").time
    
    # 扩展数据
    expanded_data = []
    for _, row in df.iterrows():
        date = row['日期']
        for time in time_intervals:
            timestamp = pd.Timestamp(f"{date} {time}")
            expanded_data.append([timestamp, row['上旋最小值'], row['下旋最小值']])
    
    # 创建新的 DataFrame
    result_df = pd.DataFrame(expanded_data, columns=['date', '上旋最小值', '下旋最小值'])
    
    return result_df

# 读取实时电网运行数据
def read_real_time_power_grid_operation_data(input_path):

    df = pd.read_excel(input_path, header=None)
    data_part = df.iloc[1:, 1:]

    rows = data_part.shape[0]
    order = ['LOAD', 'EXPORT', 'WIND', 'PHOTOVOLTAIC', 'HYDRO', 
                'FREQUENCY', 'UP_RESERVE', 'DOWN_RESERVE', 'NON_MARKET_POWER']

    reshaped_data_list = []
    for i in range(0, rows, 9):
        daily_data = data_part.iloc[i:i+9, :]
        # print(daily_data)
        # sys.exit(0)

        # 将第一列设置为分类类型，使用自定义顺序
        daily_data[1] = pd.Categorical(daily_data[1], categories=order, ordered=True)
        # 按照自定义顺序排序
        daily_data_sorted = daily_data.sort_values(by=1)
        block = daily_data_sorted.iloc[:, 1:].T.values.tolist()
        reshaped_data_list = reshaped_data_list + block

    reshaped_data_df = pd.DataFrame(reshaped_data_list)
    reshaped_data_df.columns = ['LOAD', 'EXPORT', 'WIND', 'PHOTOVOLTAIC', 'HYDRO', 'FREQUENCY', 'UP_RESERVE', 'DOWN_RESERVE', 'NON_MARKET_POWER']

    date_series = pd.date_range(start='2023-06-01 00:00:00', periods=len(reshaped_data_df), freq='15min')
    reshaped_data_df.insert(0, 'date', date_series)
    # print(reshaped_data_df.iloc[:5, :])
    # print(reshaped_data_df.iloc[-96:-91, :])

    return reshaped_data_df


if __name__ == '__main__':
    # df_historical_day_ahead_clearing_price = read_15min_variate(
    #     './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/历史日前出清价格.xlsx',
    #     '历史日前出清价格',
    #     interpolate=True)
    # l = 235
    # r = l + 20
    # print(df_historical_day_ahead_clearing_price[l:r])
    # sys.exit(0)

    # 日前历史数据（6时序数据）（剩余4非时序数据）
    df_historical_day_ahead_clearing_price = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/历史日前出清价格.xlsx',
        '历史日前出清价格',
        interpolate=False)
    df_day_ahead_scheduled_load = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前统调负荷.xlsx',
        '日前统调负荷')
    df_day_ahead_contact_line_plan = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前联络线计划.xlsx',
        '日前联络线计划')
    df_day_ahead_renewable_energy_load = read_day_ahead_renewable_energy_load(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前新能源负荷.xlsx'
    )
    df_day_ahead_positive_negative_reserve = read_day_ahead_positive_negative_reserve(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前正负备用.xlsx'
    )
    df_day_ahead_total_capacity_of_maintenance = read_total_capacity_of_maintenance_recently(
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前检修总容量.xlsx',
        './data/山西电力市场售电侧相关模型构建/日前历史数据(1)/日前检修总容量-数据纠正.xlsx'
        )

    # 实时历史数据
    df_historical_real_time_clearing_price = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/实时历史数据/历史实时出清价格.xlsx',
        '历史实时出清价格',
        interpolate=False)
    df_real_time_power_grid_operation_data = read_real_time_power_grid_operation_data(
        './data/山西电力市场售电侧相关模型构建/实时历史数据/实时电网运行数据.xlsx'
    )
    df_real_time_contact_line_plan = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/实时历史数据/实时联络线计划.xlsx',
        '实时联络线计划')
    df_real_time_spot_clearing_power = read_15min_variate(
        './data/山西电力市场售电侧相关模型构建/实时历史数据/实时现货出清电量.xlsx',
        '实时现货出清电量')
    df_real_time_reserve_amount = read_real_time_reserve_amount(
        './data/山西电力市场售电侧相关模型构建/实时历史数据/实时备用总量.xlsx'
    )


    # 将所有 DataFrame 放入字典，便于管理列名
    df_dict = {
        "日前历史出清价格": df_historical_day_ahead_clearing_price,
        "实时历史出清价格": df_historical_real_time_clearing_price,

        "日前统调负荷": df_day_ahead_scheduled_load,
        "日前联络线计划": df_day_ahead_contact_line_plan,
        "日前新能源负荷": df_day_ahead_renewable_energy_load,
        "日前正负备用": df_day_ahead_positive_negative_reserve,
        "日前检修总容量": df_day_ahead_total_capacity_of_maintenance,
        "实时电网运行数据": df_real_time_power_grid_operation_data,
        "实时联络线计划": df_real_time_contact_line_plan,
        "实时现货出清电量": df_real_time_spot_clearing_power,
        "实时备用总量": df_real_time_reserve_amount
    }

    # 统一列名
    # for name, df in df_dict.items():
    #     df.rename(columns={"date": "date", df.columns[1]: name}, inplace=True)

    # 以第一个 DataFrame 为基准，依次合并
    merged_df = None
    for df in df_dict.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="date", how="outer")

    # 按日期排序
    merged_df.sort_values(by="date", inplace=True)

    # 保存为 CSV 文件
    merged_df.to_csv("./data/raw_data.csv", index=False, encoding="utf-8-sig")

    print("数据合并完成，并已保存为 raw_data.csv")


    
    # df_merged = pd.merge(df_historical_day_ahead_clearing_price, df_historical_real_time_clearing_price, 
    #                      on="date", how="outer")
    # df_merged = pd.merge(df_merged, df_day_ahead_total_capacity_of_maintenance,
    #                      on="date", how="outer")
    # # 按时间排序（防止时间点混乱）
    # df_merged.sort_values(by="date", inplace=True)
    # # 选择填充方法（ffill 前向填充 或 bfill 后向填充）
    # df_merged.fillna(0, inplace=True)  # 缺失值默认填0
    # l = 25000
    # r = l + 10
    # print(df_merged[l:r])
