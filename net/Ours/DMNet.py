import torch, math
import torch.nn as nn
import torch.nn.functional as F
import sys, time, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from net.Ours.Module import *
from net.Ours.lib.non_local_dot_product import NONLocalBlock2D


class DMNet(nn.Module):
    def __init__(self,
                 num_classes,
                 batch_size,
                 tag,
                 group,
                 t,
                 global_n,
                 fusion_type='tandem'):
        super(DMNet, self).__init__()
        self.encoder = MobileEncoder()
        self.decoder = RefineDecoder(num_classes)
        self.lstm = TimeProcesser(256, 256, (16, 20), batch_size, tag, group)
        self.memory = Memory(256)
        self.t = t
        self.g = global_n
        self.ft = fusion_type

    def forward(self, x):
        tic = time.perf_counter()
        g = self.g
        t = self.t
        b, n, _, w, h = x.size()  #
        assert self.g + self.t == n

        seq = []

        for i in range(g):
            tensor = self.encoder(x[:, i])
            seq.append(tensor[-1].unsqueeze(1))
        global_mem = torch.cat(seq, dim=1)  # b,g,c,w,h

        seq = []
        for i in range(g, n):
            tensor = self.encoder(x[:, i])
            seq.append(tensor[-1].unsqueeze(1))
        local_mem = torch.cat(seq, dim=1)  # b,g,c,w,h

        if self.ft == 'tandem':
            local_output = self.lstm(local_mem)[:, -1]
            final_output, gdst_p = self.memory(global_mem, local_output)
        else:
            local_output = self.lstm(local_mem)[:, -1]
            global_output, _ = self.memory(global_mem, local_mem[:, -1])
            if self.ft == 'add':
                final_output = global_output + local_output
            else:
                raise NotImplementedError
        tensor[-1] = final_output
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
    net = DMNet(11, batch_size=8, tag='convlstm', group=1, t=5,
                global_n=4).cuda()

    print('CALculate..')
    with torch.no_grad():
        y = net(torch.randn(2, 9, 3, 512, 640).cuda())
