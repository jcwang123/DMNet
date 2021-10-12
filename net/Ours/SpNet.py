import torch, math
import torch.nn as nn
import torch.nn.functional as F
import sys, time, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from net.utils.helpers import maybe_download
from net.utils.layer_factory import conv1x1, conv3x3, convbnrelu, CRPBlock
from net.LSTM.torch_convlstm import ConvLSTM
from net.LSTM.bottlenecklstm import BottleneckLSTM
from net.LSTM.grouplstm import GroupLSTM
from net.Ours.Module import *


class SPNet(nn.Module):
    def __init__(self, num_classes, global_n, spatial_layer):
        super(SPNet, self).__init__()
        c = 256 if spatial_layer == -1 else 96
        self.memory = Memory(c)
        self.global_n = global_n
        self.encoder = MobileEncoder()
        self.decoder = RefineDecoder(num_classes)
        self.spatial_layer = spatial_layer

    def forward(self, x):
        tic = time.perf_counter()

        b, t, _, w, h = x.size()

        seq = []
        for i in range(t):
            tensor = self.encoder(x[:, i])
            seq.append(tensor[self.spatial_layer].unsqueeze(1))
        seq = torch.cat(seq, dim=1)

        global_context = seq[:, :-1]
        current_context = seq[:, -1]
        if self.global_n > 0:
            st_outputs, st_p = self.memory(global_context, current_context)
        else:
            st_outputs = current_context
        tensor[self.spatial_layer] = st_outputs
        out_segm = self.decoder(tensor)
        return out_segm

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def spnet(num_classes, imagenet=False, pretrained=True, **kwargs):
    """Constructs the network.

    Args:

        num_classes (int): the number of classes for the segmentation head to output.

    """
    model = SPNet(num_classes, **kwargs)
    if imagenet:
        key = "mbv2_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "mbv2_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


if __name__ == '__main__':
    import torch
    net = spnet(11, imagenet=True, global_n=4, spatial_layer=-1).cuda()

    print('CALculate..')
    with torch.no_grad():
        y = net(torch.randn(2, 5, 3, 512, 640).cuda())
