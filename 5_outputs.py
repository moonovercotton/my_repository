import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.metrics import metric1, metric2
import sys

# 加载 .npy 文件
pred = np.load('./results/gpu0_electricity_price_1080_1080_MLP_electricity_price_ftMS_sl1080_ll0_pl1080_dm64_nh8_el9_dl1_df64_fc1_ebtimeF_dtTrue/pred.npy')
true = np.load('./results/gpu0_electricity_price_1080_1080_MLP_electricity_price_ftMS_sl1080_ll0_pl1080_dm64_nh8_el9_dl1_df64_fc1_ebtimeF_dtTrue/true.npy')


# pred_np 是 shape (1080,) 或 (1080, 1)
true = true[-1:]
pred = pred[-1:]
print(true.shape)
# pred_np = true.reshape(45, 24)  # (天数, 每天24小时)
pred_np = pred.reshape(45, 24)  # (天数, 每天24小时)
# 每小时的平均值
hourly_mean = np.mean(pred_np, axis=0)
import matplotlib.pyplot as plt
# 中文字体
import matplotlib
import matplotlib.font_manager as fm
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"
font_prop = fm.FontProperties(fname=font_path)
# 更新 matplotlib 配置
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
hours = range(24)
plt.plot(hours, hourly_mean)
plt.xlabel("Hour of Day")
plt.ylabel("Average Prediction")
plt.title("Average Prediction per Hour (over 45 days)")
plt.grid(True)
plt.show()
# plt.savefig('真实结果.pdf')
plt.savefig('预测结果.pdf')
# plt.savefig('预测结果_plugin.pdf')
# exit(0)

# 展平
pred_flat = pred[-1, :, :].reshape(-1)
# true_flat = true[-1, :, :].reshape(-1)


# pred_flat = pred_flat[76:-20]
# true_flat = true_flat[76:-20]

# pred = pred_flat.reshape(85, 96, 1)
# true = true_flat.reshape(85, 96, 1)

# mae, mse, rmse = metric1(pred, true)
# mape, mspe, acc, time_segmented_acc = metric2(pred, true)
# print('mse: {}, mae: {}, rmse: {}\nmape: {}, mspe: {}\nacc: {}\ntime_segmented_acc: {}'.format(mse, mae, rmse, mape, mspe, acc, time_segmented_acc))


# mape, mspe, acc = metric2(preds, trues)
    


# 生成时间列（从最后一个时间点倒推）
end_time = datetime.strptime('2024-09-26 23:00:00', '%Y-%m-%d %H:%M:%S')
time_delta = timedelta(minutes=60)
time_list = [end_time - i * time_delta for i in range(len(pred_flat))]
time_list.reverse()  # 转为正序（从早到晚）

# 构建 DataFrame
df = pd.DataFrame({
    'time': time_list,
    'pred': pred_flat,
    # 'true': true_flat
})

# 保存为 Excel 文件
# df.to_excel('pred_true_with_time.xlsx', index=False)
df.to_excel('pred_true_with_time.xlsx', index=False)
