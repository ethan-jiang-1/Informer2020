import inspect
import collections


class Inspector(object):
    @classmethod
    def dump_dict(cls, dict, max_len=64):
        for key in dict.keys():
            val_str = str(dict[key])
            if len(val_str) > max_len: 
                val_str = val_str[0:max_len] + "..."
            val_str = val_str.replace("\n", "")
            val_str = val_str.replace("\r", " ")
            print(key, '\t', val_str)

    @classmethod
    def inspect_obj(cls, obj):
        odict = collections.OrderedDict()
        for (key, value)in inspect.getmembers(obj):
            if key.startswith("_"):
                continue
            if inspect.ismethod(value):
                continue
            if inspect.isfunction(value):
                continue
            odict[key] = value
        cls.dump_dict(odict)
        return odict

    @classmethod
    def inspect_model(cls, model, args, dataloader):
        import torch
        from torchinfo import summary

        for batch_x,batch_y,batch_x_mark,batch_y_mark in dataloader:
            break

        batch_x = batch_x.float()
        batch_y = batch_y.float()
        
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        summary(model, [batch_x.shape, batch_x_mark.shape, batch_y.shape, batch_y_mark.shape])

    
