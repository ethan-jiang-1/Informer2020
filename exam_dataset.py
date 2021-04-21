
#python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

from exam.args_setting import create_args_base
from exp.exp_informer import Exp_Informer
from exam.inspector import Inspector


args = create_args_base()
args.root_path = "./data/ETT"
#args.use_gpu = False

exp = Exp_Informer(args)

dataset, dataloader = exp._get_data("train")

print(dataset)
print(dataloader)

odict = Inspector.inspect_obj(dataset)
#print(odict)

ret = Inspector.inspect_model(exp.model, args, dataloader)
print(ret)