import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features,convert_timestamp_to_int
import warnings
from bisect import bisect
import configparser
import pickle
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train',
                 features='S', data_path='weather.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.patch_len = args.patch_len
        self.endogenous_list = args.endogenous_list
        self.exogenous_list = args.exogenous_list
        self.scaler_endogenous = StandardScaler()
        self.scaler_exogenous = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # for name, value in self.__dict__.items():
        #     print(f"{name}: {value}")
        # sys.exit(0)
        if self.data_path == 'weather/weather.csv':
            # 修改 df_raw 列名，去掉单位
            new_columns = ['date'] + self.exogenous_list + ['OT']
            df_raw.columns = new_columns


        # 设定时间范围
        start_date = "2023-06-01 00:00:00"
        end_date = self.args.input_end_time
        df_raw = df_raw[(df_raw["date"] >= start_date) & (df_raw["date"] <= end_date)].reset_index(drop=True)

        total_size = len(df_raw) // 24 * 24  # 向下取整为24的倍数
        # 按 7:2:1 比例划分
        # num_train = total_size * 7 // 10 // 24 * 24
        # num_test = total_size * 2 // 10 // 24 * 24
        # num_vali = total_size - num_train - num_test  # 保证总和等于total_size
        num_vali = 90 * 24
        num_test = 90 * 24
        num_train = total_size - num_test - num_vali

        print(f"训练集: {num_train / 24.0}天，验证集: {num_vali / 24.0}天，测试集: {num_test / 24.0}天")

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]  # 从第一个值是num_train可以看到没有考虑pred的部分，这部分落在了vali里面，感觉验证集合训练集还是重合了
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        endogenous_data = df_raw[self.endogenous_list]
        exogenous_data = df_raw[self.exogenous_list]

        if self.scale:
            if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                self.scaler_exogenous.fit(exogenous_train_data.values)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
                exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
            else:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
        else:
            endogenous_data = endogenous_data.values
            if not self.args.cross2self_attention:
                exogenous_data = exogenous_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.endogenous_data_x = endogenous_data[border1:border2]
        self.exogenous_data_x = exogenous_data[border1:border2]
        self.data_stamp = data_stamp
        self.time_points = df_raw['date'][border1:border2].values.tolist()

        # 加入 GLAFF 的时间处理
        time = self._get_time_feature(df_raw[['date']])
        self.time = time[border1:border2]

    @staticmethod
    def _get_time_feature(df):
        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].apply(lambda row: row.month / 12 - 0.5)
        df['day'] = df['date'].apply(lambda row: row.day / 31 - 0.5)
        df['weekday'] = df['date'].apply(lambda row: row.weekday() / 6 - 0.5)
        df['hour'] = df['date'].apply(lambda row: row.hour / 23 - 0.5)
        df['minute'] = df['date'].apply(lambda row: row.minute / 59 - 0.5)
        df['second'] = df['date'].apply(lambda row: row.second / 59 - 0.5)

        return df[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values


    def __getitem__(self, index):
        # exogenous part, only iTransformer, no label_len
        if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
            s_exogenous_begin = index
            s_exogenous_end = s_exogenous_begin + self.seq_len
            r_exogenous_begin = s_exogenous_end
            r_exogenous_end = r_exogenous_begin + self.pred_len
            seq_exogenous_x = self.exogenous_data_x[s_exogenous_begin:s_exogenous_end]
        else:
            seq_exogenous_x = torch.tensor([])

        # endogenous part, itransformer and PatchTST, label_len no use，因为最后还是全连接
        s_endogenous_begin = index
        s_endogenous_end = s_endogenous_begin + self.seq_len
        r_endogenous_begin = s_endogenous_end - self.label_len
        r_endogenous_end = r_endogenous_begin + self.label_len + self.pred_len

        seq_endogenous_x = self.endogenous_data_x[s_endogenous_begin:s_endogenous_end]
        seq_endogenous_y = self.endogenous_data_x[r_endogenous_begin:r_endogenous_end]
        seq_x_mark = self.data_stamp[s_endogenous_begin:s_endogenous_end]
        seq_y_mark = self.data_stamp[r_endogenous_begin:r_endogenous_end]
        # 文本形式保存的时间点
        x_time_points = self.time_points[s_exogenous_begin:s_exogenous_end]
        y_time_points = self.time_points[r_exogenous_begin:r_exogenous_end]

        # 提取月、日、星期、时、分、秒的时间
        x_time = self.time[s_endogenous_begin:s_endogenous_end]
        y_time = self.time[r_endogenous_begin:r_endogenous_end]

        # print(len(seq_exogenous_x))
        # print(len(seq_endogenous_x))
        # print(len(seq_endogenous_y))
        # sys.exit(0)

        return seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark, x_time_points, y_time_points, x_time, y_time

    def __len__(self):
        return len(self.endogenous_data_x) - self.seq_len - self.pred_len + 1

    def inverse_endogenous_transform(self, data):
        return self.scaler_endogenous.inverse_transform(data)

    def inverse_exogenous_transform(self, data):
        return self.scaler_exogenous.inverse_transform(data)
    
# class Dataset_Custom_SD(Dataset):
#     def __init__(self, args, root_path, flag='train',
#                  features='S', data_path='weather.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         self.args = args
#         # info
#         self.flag = flag
#         self.seq_len = args.seq_len
#         self.label_len = args.label_len
#         self.pred_len = args.pred_len
#         # init

#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.similar_day_path = args.similar_day_path
#         self.patch_len = args.patch_len
#         self.endogenous_list = args.endogenous_list
#         self.exogenous_list = args.exogenous_list
#         self.scaler_endogenous = StandardScaler()
#         self.scaler_exogenous = StandardScaler()
#         self.__read_data__()
#         self.__build_similar_day_index__(flag)

#     def __read_data__(self):
#         # === 读取相似日索引 ===
#         with open(self.similar_day_path, "rb") as f:
#             self.similar_days = pickle.load(f)

#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
#         # 设定时间范围
#         start_date = "2023-06-01 00:00:00"
#         end_date = "2024-08-07 23:45:00"
#         df_raw = df_raw[(df_raw["date"] >= start_date) & (df_raw["date"] <= end_date)].reset_index(drop=True)

#         df_raw['date'] = pd.to_datetime(df_raw['date'])  # 确保是datetime格式
#         df_raw['date_only'] = df_raw['date'].dt.date

#         # 获取所有唯一日期（排好序）
#         unique_days = sorted(df_raw['date_only'].unique())
#         total_days = len(unique_days)

#         num_train_days = int(total_days * 0.7)
#         num_test_days = int(total_days * 0.2)
#         num_vali_days = total_days - num_train_days - num_test_days

#         # print(num_train_days)
#         # print(num_vali_days)
#         # print(num_test_days)
#         # sys.exit(0)
#         # 303
#         # 45
#         # 86

#         # 划分日期
#         train_days = unique_days[:num_train_days]
#         val_days = unique_days[num_train_days:num_train_days + num_vali_days]
#         test_days = unique_days[-num_test_days:]

#         # 记录每段在 df_raw 中的边界索引（以行为单位）
#         train_idx = df_raw[df_raw['date_only'].isin(train_days)].index
#         val_idx = df_raw[df_raw['date_only'].isin(val_days)].index
#         test_idx = df_raw[df_raw['date_only'].isin(test_days)].index

#         border1s = [train_idx[0], val_idx[0], test_idx[0]]
#         border2s = [train_idx[-1] + 1, val_idx[-1] + 1, test_idx[-1] + 1]  # 加1表示结束边界是开区间
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         # # 查看训练集、验证集、测试集边界
#         # print(df_raw.iloc[border1s[0]][0])
#         # print(df_raw.iloc[border2s[0]-1][0])
#         # print(df_raw.iloc[border1s[1]][0])
#         # print(df_raw.iloc[border2s[1]-1][0])
#         # print(df_raw.iloc[border1s[2]][0])
#         # print(df_raw.iloc[border2s[2]-1][0])
#         # sys.exit(0)

#         endogenous_data = df_raw[self.endogenous_list]
#         exogenous_data = df_raw[self.exogenous_list]

#         # print(endogenous_data.shape)
#         # print(exogenous_data.shape)
#         # sys.exit(0)
#         # (41664, 1)
#         # (41664, 3)

#         if self.scale:
#             if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
#                 endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
#                 exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
#                 self.scaler_endogenous.fit(endogenous_train_data.values)
#                 self.scaler_exogenous.fit(exogenous_train_data.values)
#                 endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
#                 exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
#             else:
#                 endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
#                 self.scaler_endogenous.fit(endogenous_train_data.values)
#                 endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
#         else:
#             endogenous_data = endogenous_data.values
#             if not self.args.cross2self_attention:
#                 exogenous_data = exogenous_data.values

#         df_stamp = df_raw[['date']]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.endogenous_data_x = endogenous_data
#         self.exogenous_data_x = exogenous_data
#         self.data_stamp = data_stamp

#         # self.endogenous_data_x 分成 seasonal 和 non-seasonal 两部分
#         stl = STL(self.endogenous_data_x, period=24)
#         result = stl.fit()

#         self.endogenous_periodic_data_x = result.seasonal.reshape(-1,1)
#         self.endogenous_non_periodic_data_x = (result.trend + result.resid).reshape(-1,1)

#         # print(self.endogenous_data_x.shape)
#         # print(self.exogenous_data_x.shape)
#         # print(self.data_stamp.shape)
#         # sys.exit(0)
#         # (41664, 1)
#         # (41664, 3)
#         # (41664, 5)

#         self.endogenous_data_x = endogenous_data.reshape(-1, 24, endogenous_data.shape[1])
#         self.endogenous_periodic_data_x = self.endogenous_periodic_data_x.reshape(-1, 24, self.endogenous_periodic_data_x.shape[1])
#         self.endogenous_non_periodic_data_x = self.endogenous_non_periodic_data_x.reshape(-1, 24, self.endogenous_non_periodic_data_x.shape[1])
#         self.exogenous_data_x = exogenous_data.reshape(-1, 24, exogenous_data.shape[1])
#         self.data_stamp = data_stamp.reshape(-1, 24, data_stamp.shape[-1])

#         # print(self.endogenous_data_x.shape)
#         # print(self.exogenous_data_x.shape)
#         # print(self.data_stamp.shape)
#         # sys.exit(0)
#         # (434, 24, 1)
#         # (434, 24, 3)
#         # (434, 24, 5)

#         self.train_days = train_days
#         self.val_days = val_days
#         self.test_days = test_days

#     def __build_similar_day_index__(self, flag):
#         assert flag in ['train', 'val', 'test']
#         type_map = {'train': self.train_days, 'val': self.val_days, 'test': self.test_days}
#         selected_days = type_map[flag]

#         self.day_to_index = {}  # 日期 -> 在daily_tensor中的索引
#         all_days = self.train_days + self.val_days + self.test_days
#         for idx, d in enumerate(all_days):
#             self.day_to_index[d] = idx

#         self.samples = []
#         for day in selected_days:
#             if day not in self.similar_days:
#                 continue
#             sim_days = self.similar_days[day]
#             # 仅限相似日都在我们处理的范围内（train/val/test所有日期集合）
#             if all([d in self.day_to_index for d in sim_days]):
#                 self.samples.append((sim_days, day))

#         # print(flag, len(self.samples))
#         # sys.exit(0)
#         # train 303
#         # val 45
#         # test 86


#     def __getitem__(self, index):
#         sim_days, target_day = self.samples[index]
#         sim_idxs = [self.day_to_index[d] for d in sim_days]
#         target_idx = self.day_to_index[target_day]

#         # === endogenous & target ===
#         # seq_endogenous_x = self.endogenous_data_x[sim_idxs]    # (5, 24, dim)
#         seq_endogenous_periodic_data_x = self.endogenous_periodic_data_x[sim_idxs]
#         seq_endogenous_non_periodic_data_x = self.endogenous_non_periodic_data_x[sim_idxs]
#         seq_endogenous_y = self.endogenous_data_x[target_idx]  # (24, dim)


#         # === time encodings ===
#         data_stamp_days = self.data_stamp  # (num_days, 24, time_dim)
#         seq_x_mark = data_stamp_days[sim_idxs]       # (5, 24, time_dim)
#         seq_y_mark = data_stamp_days[target_idx]  # (24, time_dim)

#         return seq_endogenous_periodic_data_x, seq_endogenous_non_periodic_data_x, seq_endogenous_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         # return len(self.endogenous_data_x) - self.seq_len - self.pred_len + 1
#         return len(self.samples)

#     def inverse_endogenous_transform(self, data):
#         return self.scaler_endogenous.inverse_transform(data)

#     def inverse_exogenous_transform(self, data):
#         return self.scaler_exogenous.inverse_transform(data)
    

class Dataset_Custom_SD(Dataset):
    def __init__(self, args, root_path, flag='train',
                 features='S', data_path='weather.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.all_samples = []
        self.samples = []

        self.root_path = root_path
        self.data_path = data_path
        self.similar_day_path = args.similar_day_path
        self.patch_len = args.patch_len
        self.endogenous_list = args.endogenous_list
        self.exogenous_list = args.exogenous_list
        self.sd_index_cache_path = './data/sd_index_cache_path'
        self.scaler_endogenous = StandardScaler()
        self.scaler_exogenous = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        with open(self.similar_day_path, "rb") as f:
            self.similar_days = pickle.load(f)

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        start_date = pd.to_datetime("2023-06-01 00:00:00")
        end_date = pd.to_datetime("2024-08-07 23:00:00")
        df_raw = df_raw[(df_raw["date"] >= start_date) & (df_raw["date"] <= end_date)].reset_index(drop=True)

        total_size = len(df_raw) // 24 * 24

        num_vali = 90 * 24
        num_test = 90 * 24
        num_train = total_size - num_vali - num_test

        print(f"训练集: {num_train / 24.0}天，验证集: {num_vali / 24.0}天，测试集: {num_test / 24.0}天")

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]  # 从第一个值是num_train可以看到没有考虑pred的部分，这部分落在了vali里面，感觉验证集合训练集还是重合了
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        endogenous_data = df_raw[self.endogenous_list]
        exogenous_data = df_raw[self.exogenous_list]
        datetime_index = df_raw['date'].values

        if self.scale:
            if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                self.scaler_exogenous.fit(exogenous_train_data.values)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
                exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
            else:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
        else:
            endogenous_data = endogenous_data.values
            if not self.args.cross2self_attention:
                exogenous_data = exogenous_data.values

        df_stamp = df_raw[['date']]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.endogenous_data_x = endogenous_data[border1:border2]

        # STL 分解
        stl = STL(endogenous_data, period=24)
        result = stl.fit()
        self.endogenous_periodic_data_x = result.seasonal.reshape(-1,1)
        self.endogenous_non_periodic_data_x = (result.trend + result.resid).reshape(-1,1)

        self.endogenous_data_x = endogenous_data
        self.exogenous_data_x = exogenous_data
        self.data_stamp = data_stamp
        self.datetime_index = datetime_index

        # 如果缓存文件存在，直接加载缓存
        if os.path.exists(self.sd_index_cache_path):
            with open(self.sd_index_cache_path, "rb") as f:
                print(endogenous_data.shape)
                self.all_samples = pickle.load(f)
            print("已加载缓存的相似日索引。")

        # 如果缓存文件不存在，则构建相似日索引
        else:
            all_times = list(self.similar_days.keys())

            for target_time in all_times:
                sim_times = self.similar_days[target_time]
                try:
                    target_idx = np.where(self.datetime_index == np.datetime64(target_time))[0][0]
                    sim_idxs = [np.where(self.datetime_index == np.datetime64(t))[0][0] for t in sim_times]

                    # 检查是否所有索引都足够完整取出1080点窗口
                    if all(idx + 1080 <= len(self.datetime_index) for idx in [target_idx] + sim_idxs):
                        self.all_samples.append((sim_idxs, target_idx))
                except Exception as e:
                    print(f"跳过时间 {target_time}，错误: {e}")
                    continue  # 某个时间不在数据中，跳过

            # 缓存相似日索引
            # with open(self.sd_index_cache_path, "wb") as f:
                # pickle.dump(self.all_samples, f)
            # print("相似日索引已缓存。")

        self.samples = self.all_samples[border1:border2]


    def __getitem__(self, index):
        sim_idxs, target_idx = self.samples[index]


        sim_periodic = np.stack([self.endogenous_periodic_data_x[i:i+1080] for i in sim_idxs])
        sim_non_periodic = np.stack([self.endogenous_non_periodic_data_x[i:i+1080] for i in sim_idxs])
        target_y = self.endogenous_data_x[target_idx:target_idx+1080]

        sim_stamp = np.stack([self.data_stamp[i:i+1080] for i in sim_idxs])
        target_stamp = self.data_stamp[target_idx:target_idx+1080]

        return sim_periodic, sim_non_periodic, target_y, sim_stamp, target_stamp
    
    
    def __len__(self):
        # return len(self.endogenous_data_x) - self.seq_len - self.pred_len + 1
        return len(self.samples)

    def inverse_endogenous_transform(self, data):
        return self.scaler_endogenous.inverse_transform(data)

    def inverse_exogenous_transform(self, data):
        return self.scaler_exogenous.inverse_transform(data)
    

class Dataset_Custom_periodic_nonperiodic(Dataset):
    def __init__(self, args, root_path, flag='train',
                 features='S', data_path='weather.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init

        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.patch_len = args.patch_len
        self.endogenous_list = args.endogenous_list
        # self.exogenous_list = args.exogenous_list
        self.exogenous_periodic_list = args.exogenous_periodic_list
        self.exogenous_non_periodic_list = args.exogenous_non_periodic_list
        self.scaler_endogenous = StandardScaler()
        # self.scaler_exogenous = StandardScaler()
        self.scaler_exogenous_periodic = StandardScaler()
        self.scaler_exogenous_non_periodic = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # for name, value in self.__dict__.items():
        #     print(f"{name}: {value}")
        # sys.exit(0)
        if self.data_path == 'weather/weather.csv':
            # 修改 df_raw 列名，去掉单位
            new_columns = ['date'] + self.exogenous_list + ['OT']
            df_raw.columns = new_columns

        # 设定时间范围
        start_date = "2023-06-01 00:00:00"
        end_date = self.args.input_end_time
        df_raw = df_raw[(df_raw["date"] >= start_date) & (df_raw["date"] <= end_date)].reset_index(drop=True)

        total_size = len(df_raw) // 24 * 24  # 向下取整为24的倍数
        # 按 7:1:2 比例划分
        # num_train = total_size * 7 // 10 // 24 * 24
        # num_test = total_size * 2 // 10 // 24 * 24
        # num_vali = total_size - num_train - num_test  # 保证总和等于total_size
        num_vali = 90 * 24
        num_test = 90 * 24
        num_train = total_size - num_vali - num_test
        print(f"训练集: {num_train / 24.0}天，验证集: {num_vali / 24.0}天，测试集: {num_test / 24.0}天")

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)] 
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        endogenous_data = df_raw[self.endogenous_list]
        exogenous_periodic_data = df_raw[self.exogenous_periodic_list]
        exogenous_non_periodic_data = df_raw[self.exogenous_non_periodic_list]

        if self.scale:
            if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                # exogenous_train_data = exogenous_data[border1s[0]:border2s[0]]
                exogenous_periodic_train_data = exogenous_periodic_data[border1s[0]:border2s[0]]
                exogenous_non_periodic_train_data = exogenous_non_periodic_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                # self.scaler_exogenous.fit(exogenous_train_data.values)
                self.scaler_exogenous_periodic.fit(exogenous_periodic_train_data)
                self.scaler_exogenous_non_periodic.fit(exogenous_non_periodic_train_data)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
                # exogenous_data = self.scaler_exogenous.transform(exogenous_data.values)
                exogenous_periodic_data = self.scaler_exogenous_periodic.transform(exogenous_periodic_data.values)
                exogenous_non_periodic_data = self.scaler_exogenous_non_periodic.transform(exogenous_non_periodic_data.values)
            else:
                endogenous_train_data = endogenous_data[border1s[0]:border2s[0]]
                self.scaler_endogenous.fit(endogenous_train_data.values)
                endogenous_data = self.scaler_endogenous.transform(endogenous_data.values)
        else:
            endogenous_data = endogenous_data.values
            if not self.args.cross2self_attention:
                # exogenous_data = exogenous_data.values
                exogenous_periodic_data = exogenous_periodic_data.values
                exogenous_non_periodic_data = exogenous_non_periodic_data.values


        # # 构造新的 DataFrame
        # df_new = pd.DataFrame({
        #     'date': df_raw.iloc[:, 0],
        #     'endogenous_data': endogenous_data.reshape(-1)
        # })

        # # 确保保存的目录存在
        # os.makedirs('./data', exist_ok=True)

        # # 保存为 CSV 文件，不包含索引
        # df_new.to_csv('./data/scale_raw_data.csv', index=False)
        # sys.exit(0)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        self.endogenous_data_x = endogenous_data[border1:border2]

        # self.endogenous_data_x 分成 seasonal 和 non-seasonal 两部分
        stl = STL(self.endogenous_data_x, period=24)
        result = stl.fit()

        self.endogenous_periodic_data_x = result.seasonal.reshape(-1,1)
        self.endogenous_non_periodic_data_x = (result.trend + result.resid).reshape(-1,1)


        # print(self.endogenous_periodic_data_x[:5])
        # print(self.endogenous_non_periodic_data_x[:5])

        # fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        # l = 20000
        # r = l + 24 * 7

        # # 绘制原始数据
        # axes[0].plot(self.endogenous_data_x[l:r], color='blue')
        # axes[0].set_title('Original Data')
        # axes[0].set_ylabel('Value')

        # # 绘制周期性成分
        # axes[1].plot(self.endogenous_periodic_data_x[l:r], color='green')
        # axes[1].set_title('Seasonal Component')
        # axes[1].set_ylabel('Value')

        # # 绘制非周期性成分
        # axes[2].plot(self.endogenous_non_periodic_data_x[l:r], color='orange')
        # axes[2].set_title('Trend + Residual')
        # axes[2].set_ylabel('Value')
        # axes[2].set_xlabel('Time')

        # # 调整子图之间的间距
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('stl.pdf')

        # sys.exit(0)

        self.exogenous_periodic_data_x = exogenous_periodic_data[border1:border2]
        self.exogenous_non_periodic_data_x = exogenous_non_periodic_data[border1:border2]
        self.time_points = df_raw['date'][border1:border2].values.tolist()
        self.data_stamp = data_stamp

        # 加入 GLAFF 的时间处理
        time = self._get_time_feature(df_raw[['date']])
        self.time = time[border1:border2]

    @staticmethod
    def _get_time_feature(df):
        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].apply(lambda row: row.month / 12 - 0.5)
        df['day'] = df['date'].apply(lambda row: row.day / 31 - 0.5)
        df['weekday'] = df['date'].apply(lambda row: row.weekday() / 6 - 0.5)
        df['hour'] = df['date'].apply(lambda row: row.hour / 23 - 0.5)
        df['minute'] = df['date'].apply(lambda row: row.minute / 59 - 0.5)
        df['second'] = df['date'].apply(lambda row: row.second / 59 - 0.5)

        return df[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values



    def __getitem__(self, index):
        # exogenous part, only iTransformer, no label_len
        if not self.args.cross2self_attention and not self.args.no_embedding and not self.args.no_variate_embedding:
            s_exogenous_begin = index
            s_exogenous_end = s_exogenous_begin + self.seq_len
            r_exogenous_begin = s_exogenous_end
            r_exogenous_end = r_exogenous_begin + self.pred_len
            # seq_exogenous_x = self.exogenous_data_x[s_exogenous_begin:s_exogenous_end]
            seq_exogenous_periodic_x = self.exogenous_periodic_data_x[s_exogenous_begin:s_exogenous_end]
            seq_exogenous_non_periodic_x = self.exogenous_non_periodic_data_x[s_exogenous_begin:s_exogenous_end]
        else:
            # seq_exogenous_x = torch.tensor([])
            seq_exogenous_periodic_x = torch.tensor([])
            seq_exogenous_non_periodic_x = torch.tensor([])

        # endogenous part, itransformer and PatchTST, label_len no use，因为最后还是全连接
        s_endogenous_begin = index
        s_endogenous_end = s_endogenous_begin + self.seq_len
        r_endogenous_begin = s_endogenous_end - self.label_len
        r_endogenous_end = r_endogenous_begin + self.label_len + self.pred_len

        seq_endogenous_periodic_x = self.endogenous_periodic_data_x[s_endogenous_begin:s_endogenous_end]
        seq_endogenous_non_periodic_x = self.endogenous_non_periodic_data_x[s_endogenous_begin:s_endogenous_end]
        seq_endogenous_y = self.endogenous_data_x[r_endogenous_begin:r_endogenous_end]
        seq_x_mark = self.data_stamp[s_endogenous_begin:s_endogenous_end]
        seq_y_mark = self.data_stamp[r_endogenous_begin:r_endogenous_end]
        # 文本形式保存的时间点
        x_time_points = self.time_points[s_exogenous_begin:s_exogenous_end]
        y_time_points = self.time_points[r_exogenous_begin:r_exogenous_end]

        # 提取月、日、星期、时、分、秒的时间
        x_time = self.time[s_endogenous_begin:s_endogenous_end]
        y_time = self.time[r_endogenous_begin:r_endogenous_end]

        # print(len(seq_endogenous_periodic_x))
        # print(len(seq_endogenous_non_periodic_x))
        # print(len(seq_endogenous_y))
        # sys.exit(0)

        return seq_exogenous_periodic_x, seq_exogenous_non_periodic_x, seq_endogenous_periodic_x, seq_endogenous_non_periodic_x, seq_endogenous_y, seq_x_mark, seq_y_mark, x_time_points, y_time_points, x_time, y_time

    def __len__(self):
        return len(self.endogenous_data_x) - self.seq_len - self.pred_len + 1

    def inverse_endogenous_transform(self, data):
        return self.scaler_endogenous.inverse_transform(data)

    # def inverse_exogenous_transform(self, data):
    #     return self.scaler_exogenous.inverse_transform(data)
    
    def inverse_exogenous_periodic_transform(self, data):
        return self.scaler_exogenous_periodic.inverse_transform(data)
    
    def inverse_exogenous_non_periodic_transform(self, data):
        return self.scaler_exogenous_non_periodic.inverse_transform(data)
    
from torch.utils.data import Subset

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset  # 保存原始数据集

    def __getattr__(self, name):
        """
        代理属性和方法到原始数据集，确保 CustomSubset 可以访问原数据集的所有方法和属性
        """
        return getattr(self.dataset, name)


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    cache_path = args.cache_path
    cache_file = os.path.join(cache_path, flag + '.pkl')
    if os.path.exists(cache_file) and args.load_pickle:
        f = open(cache_file, "rb")
        data_set = pickle.load(f)
        f.close()
    else:
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq
        )
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, "wb") as f:
            pickle.dump(data_set, f)

    if flag == 'test' and args.test_stride:
        indices = list(range(0, len(data_set), args.test_stride))  # 仅取 test_stride 的倍数索引
        data_set = CustomSubset(data_set, indices)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def data_provider_SD(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    cache_path = args.cache_path
    cache_file = os.path.join(cache_path, flag + '.pkl')
    if os.path.exists(cache_file) and args.load_pickle:
        f = open(cache_file, "rb")
        data_set = pickle.load(f)
        f.close()
    else:
        data_set = Dataset_Custom_SD(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq
        )
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, "wb") as f:
            pickle.dump(data_set, f)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def data_provider_periodic_nonperiodic(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    cache_path = args.cache_path
    cache_file = os.path.join(cache_path, flag + '.pkl')
    if os.path.exists(cache_file) and args.load_pickle:
        f = open(cache_file, "rb")
        data_set = pickle.load(f)
        f.close()
    else:
        data_set = Dataset_Custom_periodic_nonperiodic(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq
        )
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, "wb") as f:
            pickle.dump(data_set, f)

    if flag == 'test' and args.test_stride:
        indices = list(range(0, len(data_set), args.test_stride))  # 仅取 test_stride 的倍数索引
        data_set = CustomSubset(data_set, indices)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader