import os
#import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset #, DataLoader
# from sklearn.preprocessing import StandardScaler

print("###load data_loader_pred")

import warnings
warnings.filterwarnings('ignore')

class BpSeqsPred(object):
    bp_seq_x = None
    bp_seq_y = None
    bp_seq_x_mark = None
    bp_seq_y_mark = None

    @classmethod
    def set_bypass_seqs(cls, seq_x, seq_y, seq_x_mark, seq_y_mark):
        print("Dataset_Pred:received bypass seqs")
        cls.bp_seq_x = seq_x
        cls.bp_seq_y = seq_y
        cls.bp_seq_x_mark = seq_x_mark
        cls.bp_seq_y_mark = seq_y_mark

    @classmethod
    def set_bypass_seqs_numpy(cls, seq_x, seq_y, seq_x_mark, seq_y_mark, args):
        print("Dataset_Pred:received bypass seqs")
        from exp.exp_basic import select_device
        device = select_device(args)
        cls.bp_seq_x = torch.from_numpy(seq_x).to(device)
        cls.bp_seq_y = torch.from_numpy(seq_y).to(device)
        cls.bp_seq_x_mark = torch.from_numpy(seq_x_mark).to(device)
        cls.bp_seq_y_mark = torch.from_numpy(seq_y_mark).to(device)

    @classmethod
    def reset_bypass_seqs(cls):
        print("Dataset_Pred: reset bypass seqs")
        cls.bp_seq_x = None
        cls.bp_seq_y = None
        cls.bp_seq_x_mark = None
        cls.bp_seq_y_mark = None 

    @classmethod
    def has_bypass_seqs(cls):
        if cls.bp_seq_x is not None:
            return True
        return False

    @classmethod
    def get_bypass_seqs(cls):
        return cls.bp_seq_x, cls.bp_seq_y, cls.bp_seq_x_mark, cls.bp_seq_y_mark

    @classmethod
    def _sio(cls, batch_tensor):
        len_size = len(batch_tensor.shape)
        if len_size == 3:
            return batch_tensor[0]
        elif len_size == 2:
            return batch_tensor
        raise ValueError("noidea")

    @classmethod
    def get_bypass_seqs_item_one(cls):
        return cls._sio(cls.bp_seq_x), cls._sio(cls.bp_seq_y), cls._sio(cls.bp_seq_x_mark), cls._sio(cls.bp_seq_y_mark)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _find_cvs_path(self):
        path = os.path.join(self.root_path, self.data_path)
        if os.path.isfile(path):
            return path
        #path = path.replace("./", "/")
        #path = app_root + path
        #if os.path.isfile(path):
        #    return path
        raise ValueError("no cvs")
        

    def __read_data__(self):
        from utils.tools import StandardScaler
        from utils.timefeatures import time_features

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self._find_cvs_path())
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        if BpSeqsPred.has_bypass_seqs():
            print("Dataset_Pred:return bypass seqs")
            return BpSeqsPred.get_bypass_seqs_item_one()

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


