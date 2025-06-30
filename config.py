import configparser
from datetime import datetime
import sys

class Config:
    def __init__(self, config_file):
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        self.device_ids = None

        # 读取配置文件
        config.read(config_file)

        # 读取 Data 部分
        self.root_path = config.get('Data', 'root_path')
        self.data_path = config.get('Data', 'data_path')
        self.similar_day_path = config.get('Data', 'similar_day_path')
        self.data = config.get('Data', 'data')
        self.check_data = config.getboolean('Data', 'check_data')
        self.cache_path = config.get('Data', 'cache_path')
        self.load_pickle = config.getboolean('Data', 'load_pickle')
        self.freq = config.get('Data', 'freq')
        self.inverse = config.getboolean('Data', 'inverse')
        self.num_of_similar_days = config.getint('Data', 'num_of_similar_days')

        # 读取 Endogenous 部分
        self.endogenous_list = config.get('Endogenous', 'target').split(',')

        # 读取 Exogenous 部分
        self.exogenous_list = config.get('Exogenous', 'target').split(',')

        # 读取 Exogenous_periodic 部分
        self.exogenous_periodic_list = config.get('Exogenous_periodic', 'target').split(',')

        # 读取 Exogenous_non_peridic 部分
        self.exogenous_non_periodic_list = config.get('Exogenous_non_peridic', 'target').split(',')

        # 读取 Model 部分
        self.model_id = config.get('Model', 'model_id')
        self.model = config.get('Model', 'model')
        self.features = config.get('Model', 'features')
        self.embed = config.get('Model', 'embed')
        self.seq_len = config.getint('Model', 'seq_len')
        self.pred_len = config.getint('Model', 'pred_len')
        self.label_len = config.getint('Model', 'label_len')
        self.patch_len = config.getint('Model', 'patch_len')
        self.e_layers = config.getint('Model', 'e_layers')
        self.L1 = config.getint('Model', 'L1')
        self.L2 = config.getint('Model', 'L2')
        self.enc_in = config.getint('Model', 'enc_in')
        self.dec_in = config.getint('Model', 'dec_in')
        self.c_out = config.getint('Model', 'c_out')
        self.d_model = config.getint('Model', 'd_model')
        self.d_ff = config.getint('Model', 'd_ff')
        self.dropout = config.getfloat('Model', 'dropout')
        self.n_heads = config.getint('Model', 'n_heads')
        self.d_layers = config.getint('Model', 'd_layers')
        self.activation = config.get('Model', 'activation')
        self.num_workers = config.getint('Model', 'num_workers')
        self.use_norm = config.getboolean('Model', 'use_norm')
        self.stride = config.getint('Model', 'stride')
        self.factor = config.getint('Model', 'factor')
        self.hw_rate = config.getfloat('Model', 'hw_rate')
        self.alpha = config.getfloat('Model', 'alpha')
        self.train_alpha = config.getboolean('Model', 'train_alpha')
        self.endogenous_separate = config.getboolean('Model', 'endogenous_separate')
        self.cross2self_attention = config.getboolean('Model', 'cross2self_attention')
        self.no_cross_attention = config.getboolean('Model', 'no_cross_attention')
        self.no_variate_embedding = config.getboolean('Model', 'no_variate_embedding')
        self.no_attention = config.getboolean('Model', 'no_attention')
        self.no_embedding = config.getboolean('Model', 'no_embedding')
        self.no_self_attention = config.getboolean('Model', 'no_self_attention')
        self.no_FF = config.getboolean('Model', 'no_FF')

        # 读取 Training 部分
        self.target = config.get('Training', 'target')
        self.use_gpu = config.getboolean('Training', 'use_gpu')
        self.mask_high_renewable = config.getboolean('Training', 'mask_high_renewable')
        self.gpu = config.getint('Training', 'gpu')
        self.use_multi_gpu = config.getboolean('Training', 'use_multi_gpu')
        self.devices = config.get('Training', 'devices')
        self.batch_size = config.getint('Training', 'batch_size')
        self.train_epochs = config.getint('Training', 'train_epochs')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.scale = config.getboolean('Training', 'scale')
        self.output_attention = config.getboolean('Training', 'output_attention')
        self.do_predict = config.getboolean('Training', 'do_predict')
        self.patience = config.getint('Training', 'patience')
        self.distil = config.getboolean('Training', 'distil')
        self.checkpoints = config.get('Training', 'checkpoints')
        self.use_amp = config.getboolean('Training', 'use_amp')
        self.lradj = config.get('Training', 'lradj')
        self.save_pdf = config.getboolean('Training', 'save_pdf')

        # 读取 Plugin 部分
        self.use_glaff = config.getboolean('Plugin', 'use_glaff')
        self.q = config.getfloat('Plugin', 'q')
        self.dim = config.getint('Plugin', 'dim')
        self.dff = config.getint('Plugin', 'dff')
        self.head_num = config.getint('Plugin', 'head_num')
        self.layer_num = config.getint('Plugin', 'layer_num')
        self.glaff_dropout = config.getfloat('Plugin', 'glaff_dropout')

        # 读取 Test 部分
        self.test_stride = config.getint('Test', 'test_stride')
        self.input_end_time = config.get('Test', 'input_end_time')


    def save_config_to_txt(self, current_time, filename):
        # 获取当前时间
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 获取Config对象的所有属性
        config_attributes = vars(self)

        # 打开文件并写入信息
        with open(filename, 'a') as file:
            file.write(f"{current_time}\n")
            for attr, value in config_attributes.items():
                file.write(f"{attr}: {value}\n")
            file.write('\n\n')