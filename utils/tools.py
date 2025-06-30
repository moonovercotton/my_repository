import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 10))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_per_day(true, preds=None, name='./pic/test_day{}.pdf'):
    """
    Results visualization
    """
    true = true[-1080:]
    # true = true[:1080]
    preds = preds[-1080:]

    segment_length = 120  # 每段24个点
    num_segments = len(true) // segment_length  # 总的分段数，假设true和preds长度相同且能整除

    for i in range(num_segments):
        # 获取当前段的数据
        start = i * segment_length
        end = (i + 1) * segment_length
        true_segment = true[start:end]
        preds_segment = preds[start:end] if preds is not None else None
        
        # 绘制当前段的数据
        plt.figure()
        if not np.all(np.abs(np.array(true_segment)) <= 1e-5):
            plt.plot(true_segment, label='GroundTruth', linewidth=2)
        if preds_segment is not None:
            plt.plot(preds_segment, label='Prediction', linewidth=2)
        plt.legend()
        
        # 保存每段为单独的PDF文件
        plt.savefig(name.format(i + 1), bbox_inches='tight')
        # print(name.format(i + 1))
        plt.close()  # 关闭当前图形，避免内存占用过高


def visual_SD(true, pred=None, input=None, name='./pic/test.pdf'):
    input_days = [input[0, i, :, 0] for i in range(5)]  # 得到 list，每个元素 shape 为 (96,)

    # reshape true 和 pred
    true_reshaped = true[0, :, 0]
    pred_reshaped = pred[0, :, 0]

    # 绘图
    plt.figure(figsize=(12, 6))

    # 画5天的输入
    for i, day in enumerate(input_days):
        plt.plot(day, label=f'Input Day {i+1}', linestyle='--')

    # 画 true 和 pred
    plt.plot(true_reshaped, label='GroundTruth', linewidth=2, color='black')
    plt.plot(pred_reshaped, label='Prediction', linewidth=2, color='red')

    # 图形美化
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Inputs (5 days), True and Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(name, bbox_inches='tight')



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def save_csv(data,filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
