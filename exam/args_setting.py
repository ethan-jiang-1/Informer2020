import torch
#import collections

#from utils.tools import dotdict

#class DotDictEx(collections.OrderedDict):
class DotDictEx(dict):
    def __init__(self):
        super(DotDictEx, self).__init__()

    def keys(self):
        return super(DotDictEx, self).keys()
    
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

        
class Args(object):
    @classmethod
    def create_args_base(cls, args_alter=None,  print_args=False):
        args = DotDictEx()

        args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

        args.root_path = './ETDataset/ETT-small/' # root path of data file
        args.data = 'ETTh1' # data
        args.data_path = 'ETTh1.csv' # data file
        args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args.target = 'OT' # target feature in S or MS task
        args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        args.checkpoints = './informer_checkpoints' # location of model checkpoints

        args.seq_len = 96 # input sequence length of Informer encoder
        args.label_len = 48 # start token length of Informer decoder
        args.pred_len = 24 # prediction sequence length
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        args.enc_in = 7 # encoder input size
        args.dec_in = 7 # decoder input size
        args.c_out = 7 # output size
        args.factor = 5 # probsparse attn factor
        args.d_model = 512 # dimension of model
        args.n_heads = 8 # num of heads
        args.e_layers = 2 # num of encoder layers
        args.d_layers = 1 # num of decoder layers
        args.d_ff = 2048 # dimension of fcn in model
        args.dropout = 0.05 # dropout
        args.attn = 'prob' # attention used in encoder, options:[prob, full]
        args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
        args.activation = 'gelu' # activation
        args.distil = True # whether to use distilling in encoder
        args.output_attention = False # whether to output attention in ecoder

        args.batch_size = 32 
        args.learning_rate = 0.0001
        args.loss = 'mse'
        args.lradj = 'type1'
        args.use_amp = False # whether to use automatic mixed precision training

        args.itr = 1
        args.train_epochs = 6 #20 #6
        args.patience = 3
        args.des = 'exp'

        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        # alter the change here
        if args_alter is not None:
            for key in args_alter.keys():
                print("{} altered from {} to {}".format(key, args[key], args_alter[key]))
                args[key] = args_alter[key]

        cls.update_args_for_running_platform(args)
        cls.update_args_for_datasrc(args) 

        if print_args:
            print('Args in experiment:')
            print(args)

        return args

    @classmethod
    def update_args_for_running_platform(cls, args):
        args.num_workers = 0
        args.use_gpu = True if torch.cuda.is_available() else False
        args.gpu = 0

        args.use_multi_gpu = False
        args.devices = '0,1,2,3'

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ','')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

    @classmethod
    def update_args_for_datasrc(cls, args):
        # Set augments by using data name
        data_parser = {
            'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        }
        if args.data in data_parser.keys():
            data_info = data_parser[args.data]
            args.data_path = data_info['data']
            args.target = data_info['T']
            args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    @classmethod
    def create_args_ETTh1_M_96_48_24(cls):
        args_alter = DotDictEx()
        args_alter.data = 'ETTh1' # data
        args_alter.data_path = 'ETTh1.csv' # data file
        args_alter.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args_alter.seq_len = 96 # input sequence length of Informer encoder
        args_alter.label_len = 48 # start token length of Informer decoder
        args_alter.pred_len = 24 # prediction sequence length
        args_alter.learning_rate = 0.00005 #0.0001
        args_alter.train_epochs = 20 #20 #6

        return cls.create_args_base(args_alter = args_alter)

    @classmethod
    def create_args_ETTh1_MS_96_48_24(cls):
        args_alter = DotDictEx()
        args_alter.data = 'ETTh1' # data
        args_alter.data_path = 'ETTh1.csv' # data file
        args_alter.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args_alter.seq_len = 96 # input sequence length of Informer encoder
        args_alter.label_len = 48 # start token length of Informer decoder
        args_alter.pred_len = 24 # prediction sequence length
        args_alter.learning_rate = 0.00005 #0.0001
        args_alter.train_epochs = 20 #20 #6
        return cls.create_args_base(args_alter = args_alter)

    @classmethod
    def create_args_ETTh1_S_96_48_24(cls):
        args_alter = DotDictEx()
        args_alter.data = 'ETTh1' # data
        args_alter.data_path = 'ETTh1.csv' # data file
        args_alter.features = 'S' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args_alter.seq_len = 96 # input sequence length of Informer encoder
        args_alter.label_len = 48 # start token length of Informer decoder
        args_alter.pred_len = 24 # prediction sequence length
        args_alter.learning_rate = 0.00005 #0.0001
        args_alter.train_epochs = 20 #20 #6
        return cls.create_args_base(args_alter = args_alter)

    @classmethod
    def create_args_ETTh1_M_96_60_12(cls):
        args_alter = DotDictEx()
        args_alter.data = 'ETTh1' # data
        args_alter.data_path = 'ETTh1.csv' # data file
        args_alter.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args_alter.seq_len = 96 # input sequence length of Informer encoder
        args_alter.label_len = 60 # start token length of Informer decoder
        args_alter.pred_len = 12 # prediction sequence length
        return cls.create_args_base(args_alter = args_alter)

    @classmethod
    def create_args_96_48_24_s(cls, args_alter=None, print_args=False):
        return cls.create_args_base()


def create_setting(args):
    ii = 0
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.des, ii)
    return setting


if __name__ == "__main__":
    ex1 = DotDictEx()
    ex1.aa = "aa"
    ex1.bb = "bb"

    print(ex1)
    print(ex1.keys())

    args1 = Args.create_args_ETTh1_M_96_48_24()
    args2 = Args.create_args_ETTh1_MS_96_48_24()
    args3 = Args.create_args_ETTh1_S_96_48_24()

    keys1 = args1.keys()
    keys2 = args2.keys()
    keys3 = args3.keys()
    print(len(keys1), len(keys2), len(keys3))

    for key in args1.keys():
        val1 = args1[key]
        val2 = args2[key]
        val3 = args3[key]
        if val1 == val2 and val2 == val3:
            continue

        print("key", key)
        print("  val1", val1)
        print("  val2", val2)
        print("  val3", val3)

