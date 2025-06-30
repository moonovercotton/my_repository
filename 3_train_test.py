import torch
import torch.nn as nn
import sys
import numpy as np
import time
import os
import models
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from data_provider import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_per_day
from utils.metrics import metric1, metric2
from utils.timefeatures import convert_inttime_to_strtime,get_now
from config import Config

only_test = True

if only_test:
    need_train = False
    need_load = True
else:
    need_train = True
    need_load = False

need_test = True

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

if need_train:
    # get data
    train_data, train_loader = data_provider(configs, flag='train')
    valid_data, valid_loader = data_provider(configs, flag='val')
    
    # choose optim and criterion
    model_optim = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()

    # train_steps and early_stopping
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)

    time_now = time.time()

    if configs.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    valid_loss_list = []

    for epoch in tqdm.tqdm(range(configs.train_epochs)):
    # for epoch in range(configs.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (seq_exogenous_x, 
                seq_endogenous_x,
                seq_endogenous_y, 
                seq_x_mark, 
                seq_y_mark, x_time_points, y_time_points,
                x_time, y_time) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            seq_exogenous_x = seq_exogenous_x.float().to(device)
            seq_endogenous_x = seq_endogenous_x.float().to(device)
            seq_endogenous_y = seq_endogenous_y.float().to(device)
            seq_x_mark = seq_x_mark.float().to(device)
            seq_y_mark = seq_y_mark.float().to(device)
            x_time = x_time.float().to(device)
            y_time = y_time.float().to(device)

            # decoder input
            # dec_inp = torch.zeros_like(seq_endogenous_y[:, -configs.pred_len:, :]).float()  # 先把后面预测的位置空下来
            # dec_seq_endogenous_y = torch.cat([seq_endogenous_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(device)# 这个和iTransformer无关，所以其实是没用代码

            # encoder - decoder
            # seq_endogenous_periodic_x, seq_endogenous_non_periodic_x
            if configs.use_amp:
                with torch.cuda.amp.autocast():
                    if configs.output_attention:
                        outputs = model(seq_endogenous_x, x_time, y_time)[0]
                    else:
                        outputs = model(seq_endogenous_x, x_time, y_time)

                    f_dim = -1 if configs.features == 'MS' else 0
                    outputs = outputs[:, -configs.pred_len:, f_dim:]
                    seq_endogenous_y = seq_endogenous_y[:, -configs.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, seq_endogenous_y)
                    train_loss.append(loss.item())
            else:
                if configs.output_attention:
                    outputs = model(seq_endogenous_x, x_time, y_time)[0]
                else:
                    outputs = model(seq_endogenous_x, x_time, y_time)

                f_dim = -1 if configs.features == 'MS' else 0
                outputs = outputs[:, -configs.pred_len:, f_dim:]
                seq_endogenous_y = seq_endogenous_y[:, -configs.pred_len:, f_dim:].to(device)  # 对于TimeXer目前只有一个变量，所以f_dim是啥都无所谓

                # print(f'outputs = {outputs}')
                # print(f'y = {seq_endogenous_y}')
                # sys.exit(0)

                loss = criterion(outputs, seq_endogenous_y)
                # print(f'loss = {loss.item()}')
                # sys.exit(0)
                train_loss.append(loss.item())

            if configs.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        # print(f'train_loss = {train_loss}')
        # sys.exit(0)

        train_loss = np.average(train_loss)

        # valid
        print('\nbegin valid...')
        valid_loss = []
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark, x_time_points, y_time_points, x_time, y_time) in enumerate(valid_loader):
                
                seq_exogenous_x = seq_exogenous_x.float().to(device)
                seq_endogenous_x = seq_endogenous_x.float().to(device)
                seq_endogenous_y = seq_endogenous_y.float().to(device)
                seq_x_mark = seq_x_mark.float().to(device)
                seq_y_mark = seq_y_mark.float().to(device)
                x_time = x_time.float().to(device)
                y_time = y_time.float().to(device)

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
                outputs = outputs[:, -configs.pred_len:, :]
                seq_endogenous_y = seq_endogenous_y[:, -configs.pred_len:, :].to(device)

                pred = outputs.detach().cpu()#.numpy()
                true = seq_endogenous_y.detach().cpu()#.numpy()
                pred = pred[:, :, f_dim:]
                true = true[:, :, f_dim:]

                preds.append(pred)
                trues.append(true)

                # print(f'pred = {pred}')
                # print(f'true = {true}')
                # sys.exit(0)

                loss = criterion(pred, true)
                valid_loss.append(loss.item())

        valid_loss = np.average(valid_loss)
        valid_loss_list.append(valid_loss)
        model.train()

        # print
        print(f"train_loss: {train_loss}\nvalid_loss: {valid_loss}\n")
        early_stopping(valid_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(model_optim, epoch + 1, configs)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(valid_loss_list) + 1), valid_loss_list, marker='o', label='Validation Loss')
    plt.title('Validation Loss Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图表为 PDF 文件
    output_path = "./validation_loss_plot.pdf"  # 替换为你想保存的路径
    plt.savefig(output_path, format='pdf')


if need_test:
    df = pd.read_csv('./data/raw_hourly_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    test_data, test_loader = data_provider(configs, flag='test')
    if need_load:
        print('loading model...')
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    preds_original = []
    trues_original = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (seq_exogenous_x, seq_endogenous_x, seq_endogenous_y, seq_x_mark, seq_y_mark, x_time_points, y_time_points, x_time, y_time) in tqdm.tqdm(enumerate(test_loader)):
            seq_exogenous_x = seq_exogenous_x.float().to(device)
            seq_endogenous_x = seq_endogenous_x.float().to(device)
            seq_endogenous_y = seq_endogenous_y.float().to(device)
            seq_x_mark = seq_x_mark.float().to(device)
            seq_y_mark = seq_y_mark.float().to(device)  
            x_time = x_time.float().to(device)
            y_time = y_time.float().to(device)

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
            seq_endogenous_y = seq_endogenous_y[:, -configs.pred_len:, f_dim:].to(device)
            outputs = outputs.detach().cpu().numpy()
            seq_endogenous_y = seq_endogenous_y.detach().cpu().numpy()
            if test_data.scale and configs.inverse:
                shape = outputs.shape
                outputs_original = test_data.inverse_endogenous_transform(outputs.squeeze(0)).reshape(shape)
                seq_endogenous_y_original = test_data.inverse_endogenous_transform(seq_endogenous_y.squeeze(0)).reshape(shape)

            pred = outputs
            true = seq_endogenous_y
            pred_original = outputs_original
            true_original = seq_endogenous_y_original

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
            # adjusted_pred = pred_ori.reshape(1, 1080, 1)

            # if configs.mask_high_renewable:
            #     # 展开 tuple 列表
            #     y_time_points_flat = [t[0] for t in y_time_points]  # 提取每个 tuple 中的字符串
            #     # 2. 将 y_time_points 转为时间格式
            #     y_times = pd.to_datetime(y_time_points_flat)
            #     # 3. 创建 DataFrame 作为 lookup 基准
            #     lookup_df = pd.DataFrame({'date': y_times})
            #     # 为了合并，先把预测值也压平
            #     flat_pred = pred_original.squeeze()  # shape: (1080,)
            #     lookup_df['pred'] = flat_pred
            #     # 4. 合并风电光伏总负荷值
            #     merged = pd.merge(lookup_df, df[['date', '日前风电光伏总负荷']], on='date', how='left')
            #     # 5. 使用 NumPy 向量操作进行条件替换
            #     mask = merged['日前风电光伏总负荷'].values > 16700
            #     merged.loc[mask, 'pred'] = 25
            #     # 6. 恢复为 (1, 1080, 1) 的 numpy 数组
            #     pred_original = merged['pred'].values.reshape(1, -1, 1)
                
            preds.append(pred)
            trues.append(true)
            preds_original.append(pred_original)
            trues_original.append(true_original)

            if i % 45 == 0 and configs.save_pdf:
                input = seq_endogenous_x.detach().cpu().numpy()
                if test_data.scale and configs.inverse:
                    shape = input.shape
                    input = test_data.inverse_endogenous_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true_original[0, :, -1]), axis=0)
                    pd_ = np.concatenate((input[0, :, -1], pred_original[0, :, -1]), axis=0)
                else:
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd_ = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                visual_per_day(gt, pd_, os.path.join(folder_path + '{}.pdf'))


            if configs.check_data == True and configs.batch_size == 1:
                if test_data.scale and configs.inverse:
                    shape_endx = seq_endogenous_x.shape
                    shape_exox = seq_exogenous_x.shape
                    seq_endogenous_x = seq_endogenous_x.detach().cpu().numpy()
                    seq_exogenous_x = seq_exogenous_x.detach().cpu().numpy()
                    seq_endogenous_x_inversed = test_data.inverse_endogenous_transform(seq_endogenous_x.squeeze(0)).reshape(shape_endx)
                    seq_endogenous_y_inversed = seq_endogenous_y
                    seq_exogenous_x_inversed = test_data.inverse_exogenous_transform(seq_exogenous_x.squeeze(0)).reshape(shape_exox)
                    print(f"batch_size: {i}, train_others: {seq_exogenous_x_inversed} train_target: {seq_endogenous_x_inversed} true_target {seq_endogenous_y_inversed}")
                    print(f"pred_target:", pred)

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    preds_original = np.array(preds_original)
    trues_original = np.array(trues_original)
    preds_original = preds_original.reshape(-1, preds_original.shape[-2], preds_original.shape[-1])
    trues_original = trues_original.reshape(-1, trues_original.shape[-2], trues_original.shape[-1])

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

    mae, mse, rmse = metric1(preds, trues)
    mape, mspe, acc, time_segmented_acc = metric2(preds_original, trues_original)
    # mape, mspe, acc = metric2(preds, trues)
    print('mse: {}, mae: {}, rmse: {}\nmape: {}, mspe: {}\nacc: {}\ntime_segmented_acc: {}'.format(mse, mae, rmse, mape, mspe, acc, time_segmented_acc))
    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    current_time = get_now()
    f.write(f'{current_time}\n')
    f.write(f'Model Param Num: {param_num:,}\n')
    # if configs.train_alpha:
    #     f.write(f"Trained alpha value: {model.alpha.item()}\n")
    # else:
    #     f.write(f"alpha value: {configs.alpha}\n")
    f.write('mse: {}\nmae: {}\nrmse: {}\nmape: {}\nmspe: {}\nacc: {}\ntime_segmented_acc: {}'.format(mse, mae, rmse, mape, mspe, acc, time_segmented_acc))
    f.write('\n')
    f.write('\n')
    f.close()

    configs.save_config_to_txt(current_time, 'configs.txt')

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, acc]))
    np.save(folder_path + 'pred.npy', preds_original)
    np.save(folder_path + 'true.npy', trues_original)
