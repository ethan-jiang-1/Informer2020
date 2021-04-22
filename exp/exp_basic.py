import os
import torch

def select_device(args, prompt=False):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        if prompt:
            print('Use GPU: cuda:{}'.format(args.gpu))
        return device
    else:
        device = torch.device('cpu')
        if prompt:
            print('Use CPU')
        return device

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        device = select_device(self.args, prompt=True)
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    