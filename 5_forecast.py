import torch
import torch.nn as nn
import sys
import numpy as np
import time
import os
import models
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from data_provider import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_per_day
from utils.metrics import metric1, metric2
from utils.timefeatures import convert_inttime_to_strtime,get_now
from config import Config
from datetime import datetime, timedelta

def _get_time_feature(df):
    df['date'] = pd.to_datetime(df['date'])

    df['month'] = df['date'].apply(lambda row: row.month / 12 - 0.5)
    df['day'] = df['date'].apply(lambda row: row.day / 31 - 0.5)
    df['weekday'] = df['date'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['hour'] = df['date'].apply(lambda row: row.hour / 23 - 0.5)
    df['minute'] = df['date'].apply(lambda row: row.minute / 59 - 0.5)
    df['second'] = df['date'].apply(lambda row: row.second / 59 - 0.5)

    return df[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values


# 使用示例
config_file = 'training.ini'
configs = Config(config_file)

# 打印一些参数检查
# print(f"Model ID: {configs.model_id}")
# print(f"Batch Size: {configs.batch_size}")
# print(f"Endogenous List: {configs.endogenous_list}")
# print(f"Exogenous List: {configs.exogenous_list}")

# setting 
setting = 'gpu{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
            configs.gpu,
            configs.model_id,
            configs.model,
            configs.data,
            configs.features,
            configs.seq_len,
            configs.label_len,
            configs.pred_len,
            configs.d_model,
            configs.n_heads,
            configs.e_layers,
            configs.d_layers,
            configs.d_ff,
            configs.factor,
            configs.embed,
            configs.distil)
path = os.path.join(configs.checkpoints, setting)
if not os.path.exists(path):
    os.makedirs(path)

# device
if configs.use_gpu and configs.use_multi_gpu:
    configs.devices = configs.devices.replace(' ', '')
    device_ids = configs.devices.split(',')
    configs.device_ids = [int(id_) for id_ in device_ids]
    configs.gpu = configs.device_ids[0]

if configs.use_gpu:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu) if not configs.use_multi_gpu else configs.devices
    device = torch.device('cuda:{}'.format(configs.gpu))
    print('Use GPU: cuda:{}'.format(configs.gpu))
else:
    device = torch.device('cpu')
    print('Use CPU')


# define model
if configs.model == 'MyNet':
    model = models.MyNet(configs).float()
elif configs.model == 'TimeXer':
    model = models.TimeXer(configs).float()
elif configs.model == 'MLP':
    model = models.MLP(configs).float()
elif configs.model == 'CNN':
    model = models.CNN(configs).float()
else:
    print('Invalid Model!')
    sys.exit(0)

if configs.use_multi_gpu and configs.use_gpu :
    model = model.to('cuda')
    model = nn.DataParallel(model, device_ids=configs.device_ids)
else:
    model = model.to(device)

param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model Param Num: {param_num:,}')
# sys.exit(0)

print('loading model...')
model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

preds = []
preds_original = []
folder_path = './test_results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# read data
input_end_time = configs.input_end_time
end_time = datetime.strptime(input_end_time, '%Y-%m-%d %H:%M:%S')
input_start_time = (end_time - timedelta(hours=1080-1)).strftime('%Y-%m-%d %H:%M:%S')  # 输入窗口为 1080

df_raw = pd.read_csv(os.path.join(configs.root_path, configs.data_path))

total_size = len(df_raw) // 24 * 24  # 向下取整为24的倍数
num_vali = 90 * 24
num_test = 90 * 24
num_train = total_size - num_test - num_vali

df_forecast_input = df_raw[(df_raw["date"] >= input_start_time) & (df_raw["date"] <= input_end_time)].reset_index(drop=True)

endogenous_data = df_raw[configs.endogenous_list]
endogenous_data_forecast_input = df_forecast_input[configs.endogenous_list]

# 标准化
if configs.scale:
    endogenous_train_data = endogenous_data[0:num_train]
    train_values = endogenous_train_data.values
    # 计算训练集的均值和标准差（按列）
    mean = np.mean(train_values, axis=0)
    std = np.std(train_values, axis=0)
    std[std == 0] = 1e-8
    endogenous_data = (endogenous_data.values - mean) / std
    endogenous_data_forecast_input = (endogenous_data_forecast_input.values - mean) / std
else:
    endogenous_data = endogenous_data.values
    endogenous_data_forecast_input = endogenous_data_forecast_input.values

time_points = df_forecast_input['date'].values.tolist()

# 加入 GLAFF 的时间处理
x_date = df_forecast_input[['date']]
last_time = pd.to_datetime(x_date['date'].iloc[-1])
y_dates = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=1080, freq='H')
y_date = pd.DataFrame({'date': y_dates.strftime('%Y-%m-%d %H:%M:%S')})
xy_date = pd.concat([x_date, y_date], ignore_index=True)

xy_time = _get_time_feature(xy_date[['date']])
x_time = xy_time[:1080]
y_time = xy_time[-1080:]

# 加入规则判断
y_time_points = y_date.values.tolist()

model.eval()

with torch.no_grad():
    seq_endogenous_x = torch.from_numpy(endogenous_data_forecast_input).float().to(device)
    x_time = torch.from_numpy(x_time).float().to(device)
    y_time = torch.from_numpy(y_time).float().to(device)
    seq_endogenous_x = seq_endogenous_x.unsqueeze(0)
    x_time = x_time.unsqueeze(0)
    y_time = y_time.unsqueeze(0)

    # # decoder input
    # dec_inp = torch.zeros_like(seq_endogenous_y[:, -configs.pred_len:, :]).float()
    # dec_seq_endogenous_y = torch.cat([seq_endogenous_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(device)
    
    # encoder - decoder
    if configs.use_amp:
        with torch.cuda.amp.autocast():
            if configs.output_attention:
                outputs = model(seq_endogenous_x, x_time, y_time)[0]
            else:
                outputs = model(seq_endogenous_x, x_time, y_time)
    else:
        if configs.output_attention:
            outputs = model(seq_endogenous_x, x_time, y_time)[0]
        else:
            outputs = model(seq_endogenous_x, x_time, y_time)

    f_dim = -1 if configs.features == 'MS' else 0
    outputs = outputs[:, -configs.pred_len:, f_dim:]
    outputs = outputs.detach().cpu().numpy()
    if configs.scale and configs.inverse:
        shape = outputs.shape
        outputs_original = outputs * std + mean
        # outputs_original = scaler_endogenous.inverse_endogenous_transform(outputs.squeeze(0)).reshape(shape)

    pred = outputs
    pred_original = outputs_original

    # adjusted_pred = pred_original.copy()
    # pred_ori = adjusted_pred.reshape(-1)
    # hour_indices = np.tile(np.arange(24), 45)
    # total_hours = pred_ori.shape[0]
    # points_per_day = 24
    # days = total_hours // points_per_day
    # decrease_weights = np.array([0.95, 0.9, 0.85, 0.9, 0.95, 1.0])  # 11:00-16:00
    # increase_weights = np.array([1.05, 1.1, 1.15, 1.1, 1.05])       # 17:00-21:00
    # hourly_weights = np.ones((24,))
    # hourly_weights[11:17] = decrease_weights
    # hourly_weights[17:22] = increase_weights
    # full_weights = np.tile(hourly_weights, days)
    # pred_ori *= full_weights
    # pred_original = pred_ori.reshape(1, 1080, 1)

    preds.append(pred)
    preds_original.append(pred_original)

    if configs.save_pdf:
        input = seq_endogenous_x.detach().cpu().numpy()
        if configs.scale and configs.inverse:
            shape = input.shape
            input = input * std + mean
            # input = scaler_endogenous.inverse_endogenous_transform(input.squeeze(0)).reshape(shape)
            pd_ = np.concatenate((input[0, :, -1], pred_original[0, :, -1]), axis=0)
        else:
            pd_ = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
        # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        visual_per_day(pd_, pd_, os.path.join(folder_path + '{}.pdf'))

preds = np.array(preds)
preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

preds_original = np.array(preds_original)
preds_original = preds_original.reshape(-1, preds_original.shape[-2], preds_original.shape[-1])

# result save
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model Param Num: {param_num:,}')
# if configs.train_alpha:
#     # 查看 alpha 的值
#     print(f"Trained alpha value: {model.alpha.item()}")
# else:
#     print(f'alpha: {configs.alpha}')

# mae, mse, rmse = metric1(preds, [])
# mape, mspe, acc, time_segmented_acc = metric2(preds_original, [])
# # mape, mspe, acc = metric2(preds, trues)
# print('mse: {}, mae: {}, rmse: {}\nmape: {}, mspe: {}\nacc: {}\ntime_segmented_acc: {}'.format(mse, mae, rmse, mape, mspe, acc, time_segmented_acc))
# f = open("result_long_term_forecast.txt", 'a')
# f.write(setting + "  \n")
# current_time = get_now()
# f.write(f'{current_time}\n')
# f.write(f'Model Param Num: {param_num:,}\n')
# # if configs.train_alpha:
# #     f.write(f"Trained alpha value: {model.alpha.item()}\n")
# # else:
# #     f.write(f"alpha value: {configs.alpha}\n")
# f.write('mse: {}\nmae: {}\nrmse: {}\nmape: {}\nmspe: {}\nacc: {}\ntime_segmented_acc: {}'.format(mse, mae, rmse, mape, mspe, acc, time_segmented_acc))
# f.write('\n')
# f.write('\n')
# f.close()

# configs.save_config_to_txt(current_time, 'configs.txt')

# np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, acc]))
np.save(folder_path + 'pred.npy', preds_original)
