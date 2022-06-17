import torch
import torch.nn as nn
import sys, time, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# from net.utils.helpers import maybe_download
# from net.utils.layer_factory import conv1x1, conv3x3, convbnrelu, CRPBlock
from net.LSTM.torch_convlstm import ConvLSTM
from net.LSTM.bottlenecklstm import BottleneckLSTM
from net.LSTM.grouplstm import GroupLSTM
from net.Ours.Module import *

from net.Ours.lib.non_local_dot_product import NONLocalBlock2D


class TemporalNet(nn.Module):
    def __init__(self, num_classes, batch_size, tag, group):
        super(TemporalNet, self).__init__()
        self.encoder = MobileEncoder()
        self.decoder = RefineDecoder(num_classes)
        self.lstm = TimeProcesser(256, 256, (16, 20), batch_size, tag, group)

    def forward(self, x):
        tic = time.perf_counter()
        b, t, _, w, h = x.size()  #

        seq = []

        for i in range(t):
            tensor = self.encoder(x[:, i])
            seq.append(tensor[-1].unsqueeze(1))
        tem = torch.cat(seq, dim=1)  # b,t,c,w,h

        temporal_output = self.lstm(tem)[:, -1]

        tensor[-1] = temporal_output
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


if __name__ == '__main__':
    import torch
    net = TemporalNet(11, batch_size=8, tag='btnlstm', group=1).cuda()

    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)

    print('CALculate..')
    with torch.no_grad():
        y = net(torch.randn(2, 5, 3, 512, 640).cuda())
