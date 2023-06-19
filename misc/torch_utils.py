import torch
from torch.backends import cudnn


def init_device_pytorch(use_cudnn=True):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        gpus = list(range(torch.cuda.device_count()))
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')
    cudnn.benchmark = use_cudnn
    return device


def prepare_weights_dict(ckpt_path):
    """ Removes all unnecessary fields and convert weights from GPU to CPU device"""

    dct = torch.load(ckpt_path)
    if len(dct) > 1:
        if 'model_state' in dct:
            dct = dct['model_state']
        elif 'model' in dct:
            dct = dct['model']
        elif 'state_dict' in dct:
            dct = dct['state_dict']
            if next(iter(dct)).startswith('model'):
                new_dct = {}
                for k, v in dct.items():
                    name = k[6:]  # remove `model.`
                    new_dct[name] = v.to('cpu')
                dct = new_dct
        else:
            raise Exception("wrong format of the checkpoint/weights file")
    if ckpt_path.endswith('.ckpt'):
        ckpt_path = ckpt_path.replace('.ckpt', '.pth')
    torch.save(dct, ckpt_path)
    print("weights were succesfully saved")


if __name__ == '__main__':
    prepare_weights_dict('/home/noteme/data/results/logger/version_3/epoch=02-precision=0.9655.ckpt')
