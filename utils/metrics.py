import numpy as np
import sys

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, epsilon=1e-5):
    return np.mean(np.abs((pred - true) / (true + epsilon))) * 100


def MSPE(pred, true, epsilon=1e-5):
    return np.mean(np.square((pred - true) / (true + epsilon))) * 100

def ACC(pred, true):
    # print(np.sum(np.abs(true - pred)))
    # print(np.sum(true))
    # print(pred.shape)
    # print(true.shape)
    # sys.exit(0)
    return  (1 - np.sum(np.abs(true - pred)) / np.sum(true)) * 100

# 按天计算准确率后取平均
# def ACC(pred, true):
#     # 计算每天的准确率
#     daily_accuracies = []
#     for i in range(pred.shape[0]):
#         # 取第 i 天的数据
#         pred_day = pred[i, :, 0]  # (96,)
#         true_day = true[i, :, 0]  # (96,)
#         daily_accuracy = (1 - np.sum(np.abs(true_day - pred_day)) / np.sum(true_day)) * 100
#         daily_accuracies.append(daily_accuracy)

#     # 返回86天的平均准确率
#     return np.mean(daily_accuracies)

def Time_segmented_ACC(pred, true):
    # 计算每小时的平均值
    pred_hour = pred.reshape(pred.shape[0] * 45, 24, 1)
    true_hour = true.reshape(pred.shape[0] * 45, 24, 1)

    # 定义时间段索引
    time_ranges = {
        "6:00-10:00": (6, 10),
        "11:00-16:00": (11, 16),
        "17:00-21:00": (17, 21),
    }

    # 计算并打印每个时间段的 ACC
    acc_results = {}
    for label, (start, end) in time_ranges.items():
        acc_results[label] = ACC(pred_hour[:, start:end, :], true_hour[:, start:end, :])

    return acc_results

def Time_segmented_ACC_SD(pred, true):
    # 计算每小时的平均值
    # pred_hour = pred.reshape(pred.shape[0] * 45, 24, 1)
    # true_hour = true.reshape(pred.shape[0] * 45, 24, 1)

    # 定义时间段索引
    time_ranges = {
        "6:00-10:00": (6, 10),
        "11:00-16:00": (11, 16),
        "17:00-21:00": (17, 21),
    }

    # 计算并打印每个时间段的 ACC
    acc_results = {}
    for label, (start, end) in time_ranges.items():
        acc_results[label] = ACC(pred[:, start:end, :], true[:, start:end, :])

    return acc_results


def metric1(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    return mae, mse, rmse

def metric2(pred, true):
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    acc = ACC(pred, true)
    time_segmented_acc = Time_segmented_ACC(pred, true)
    return mape, mspe, acc, time_segmented_acc

def metric2_SD(pred, true):
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    acc = ACC(pred, true)
    time_segmented_acc = Time_segmented_ACC_SD(pred, true)
    return mape, mspe, acc, time_segmented_acc
