import torch, math
import torch.nn as nn
import torch.nn.functional as F
import sys, time, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from net.Ours.Module import *


class EffLA(nn.Module):
    def __init__(self, num_classes, tag):
        super(EffLA, self).__init__()
        self.encoder = MobileEncoder()
        self.decoder = RefineDecoder(num_classes)
        self.lstm = TimeProcesser(256, 256, (16, 20), 1, tag, 1)
        self.memory = Memory(256)

    def forward(self, x):
        tic = time.perf_counter()

        b, t, _, w, h = x.size()

        seq = []
        for i in range(t):
            tensor = self.encoder(x[:, i])
            seq.append(tensor[-1].unsqueeze(1))
        seq = torch.cat(seq, dim=1)

        temporal_output = self.lstm(seq)[:, -1:]
        densest_output, p = self.memory(temporal_output, temporal_output[:, 0])

        tensor[-1] = densest_output
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
    net = EffLA(11, tag='convlstm').cuda()

    print('CALculate..')
    with torch.no_grad():
        y = net(torch.randn(2, 5, 3, 512, 640).cuda())
