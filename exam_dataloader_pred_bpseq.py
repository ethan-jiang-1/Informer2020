
#python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

#from exam.args_setting import create_args_base
import numpy as np
from exp.exp_informer import Exp_Informer
#from exam.inspector import Inspectorormer
from exam.args_setting import Args, create_setting
from data.data_loader_pred import BpSeqsPred


def alter_sys_path():
    import os
    import sys
    module_root = os.path.dirname(__file__)
    if module_root not in sys.path:
        sys.path.append(module_root)
    app_root = os.path.dirname(module_root)
    if app_root not in sys.path:
        sys.path.append(app_root)


if __name__ == "__main__":

    args = Args.create_args_base()
    setting = create_setting(args)

    exp = Exp_Informer(args)
    print(exp)

    data_set, data_loader = exp._get_data(flag="pred")
    print(data_set)
    print(data_loader)

    print("")
    print("init")
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        #print(i)
        print(batch_x.shape)
    
    train_data_set, train_data_loader = exp._get_data(flag="train")
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_data_loader):
        if i == 0:
            print(type(batch_x), batch_x.shape)
            print(type(batch_y), batch_y.shape)
            print(type(batch_x_mark), batch_x_mark.shape)
            print(type(batch_y_mark), batch_y_mark.shape)
            print("")
            BpSeqsPred.set_bypass_seqs(batch_x, batch_y, batch_x_mark, batch_y_mark)
            break 

    print("")
    print("has BpSeq")
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        #print(i)
        print(type(batch_x), batch_x.shape)
        print(type(batch_y), batch_y.shape)
        print(type(batch_x_mark), batch_x_mark.shape)
        print(type(batch_y_mark), batch_y_mark.shape)

    print("")
    BpSeqsPred.reset_bypass_seqs()
    print("has no BpSeq")
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        #print(i)
        print(type(batch_x), batch_x.shape)

    print("")
    print("setup on by numpy")
    BpSeqsPred.set_bypass_seqs(np.zeros((196,7)), np.zeros((172,7)), np.zeros((196,4)), np.zeros((172,4)))

    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        #print(i)
        print(type(batch_x), batch_x.shape)
        print(type(batch_y), batch_y.shape)
        print(type(batch_x_mark), batch_x_mark.shape)
        print(type(batch_y_mark), batch_y_mark.shape)

    print("")
    BpSeqsPred.reset_bypass_seqs()
    print("has no BpSeq")
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
        #print(i)
        print(type(batch_x), batch_x.shape)
