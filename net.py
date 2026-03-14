from pretrained.smt import smt_t
import torch
import torch.nn as nn
from thop import profile
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
import torch.nn.functional as F

class EFFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EFFM, self).__init__()

        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1)
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up_dwc(x)
        return x



class CCM(nn.Module):
    def __init__(self, dim, ratio=1, band_kernel_size=11, square_kernel_size=3,drop_path=0.):
        super(CCM, self).__init__()
        self.dim = dim
        self.dwconv_hw = nn.Conv2d(dim, dim, square_kernel_size, padding=square_kernel_size // 2, groups=dim)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = dim // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(dim, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, dim, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = ConvMlp(dim, int(2 * dim))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def sge(self, x):
        x_h = self.pool_h(x)    # C,H,1
        x_w = self.pool_w(x)    # C,1,W
        x_gather = x_h + x_w  # C,H,W
        ge = self.excite(x_gather)  # C,H,W
        return ge

    def forward(self, x):
        shortcut = x
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        x = self.mlp(self.norm(att * loc))
        x = x * self.gamma
        out = self.drop_path(x) + shortcut

        return out


class ConvMlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Enhance(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, dilation=2, groups=dim),
            nn.BatchNorm2d(dim)
        )
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1, padding=0, stride=1, dilation=1, groups=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1, padding=0, stride=1, dilation=1, groups=1)
        self.g = nn.Sequential(nn.Conv2d(mlp_ratio * dim,  dim, 1, padding=0, stride=1, dilation=1, groups=1), nn.BatchNorm2d(dim))
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, dilation=2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



def get_open_map(input, kernel_size):
    return F.max_pool2d(input, kernel_size=kernel_size,
                        stride=1, padding=kernel_size//2)

class EGR(nn.Module):
    def __init__(self, cur_channel, dep_channel, focus_background = True, opr_kernel_size = 7,iterations = 1):
        super(EGR, self).__init__()
        self.cur_channel = cur_channel
        self.dep_channel = dep_channel
        self.focus_background = focus_background

        self.block = Enhance(dim=cur_channel)
        self.input_map = nn.Sigmoid()

        self.output_map = nn.Conv2d(self.cur_channel, 1, 1)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.conv2 = nn.Sequential(nn.Conv2d(self.cur_channel, self.cur_channel, kernel_size=3, stride=1, padding=1, groups=self.cur_channel),
                                nn.Conv2d(self.cur_channel, self.cur_channel, kernel_size=1, stride=1, padding=0))

        self.conv_cur_dep = nn.Sequential(nn.Conv2d(2 * self.cur_channel, 2 * self.cur_channel, kernel_size=3, stride=1, padding=1, groups=2 * self.cur_channel, bias=False),  # 深度卷积
                                    nn.Conv2d(2 * self.cur_channel, self.cur_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(self.cur_channel), nn.ReLU())

        self.opr_kernel_size = opr_kernel_size
        self.iterations = iterations

        self.proj = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(self.dep_channel, self.cur_channel, 1, 1, 0, 1, 1, bias=False),
                                  nn.BatchNorm2d(self.cur_channel))

    def forward(self, cur_x, dep_x, in_map):
        input_map = self.input_map(F.interpolate(in_map, scale_factor=2, mode='bilinear', align_corners=True))
        if self.focus_background:
            map = get_open_map(input_map, self.opr_kernel_size)
            increase_map = map - input_map
            b_feature = cur_x * increase_map
        else:
            b_feature = cur_x * input_map

        fn = self.conv2(b_feature)
        dep_x_up = F.interpolate(dep_x, scale_factor=2, mode='bilinear', align_corners=True)
        fn1, fn2 = torch.split(dep_x_up, [self.cur_channel, self.cur_channel], dim=1)
        dep_x1 = self.block(fn1)
        dep_x12 = self.proj(torch.cat([dep_x1, fn2], dim=1))

        refine2 = self.conv_cur_dep(torch.cat((dep_x12, self.beta * fn), dim=1))
        output_map = self.output_map(refine2)
        return output_map


class CENet(nn.Module):
    def __init__(self):
        super(CENet, self).__init__()

        self.smt = smt_t()
        self.ccm_1 = CCM(dim=512)
        self.ccm_2 = CCM(dim=256)
        self.ccm_3 = CCM(dim=128)
        self.ccm_4 = CCM(dim=64)

        self.up_1 = EFFM(in_channels=512, out_channels=256)
        self.up_2 = EFFM(in_channels=256, out_channels=128)
        self.up_3 = EFFM(in_channels=128, out_channels=64)

        self.dwcon_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=256, bias=False),  # 深度卷积
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),  # 逐点卷积
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dwcon_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),  # 深度卷积
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),  # 逐点卷积
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dwcon_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),  # 深度卷积
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),  # 逐点卷积
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pred1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False),
            nn.Conv2d(512, 1, 1)
        )

        self.egr_1 = EGR(cur_channel=256, dep_channel=512, focus_background=True, opr_kernel_size=7, iterations=1)
        self.egr_2 = EGR(cur_channel=128, dep_channel=256, focus_background=True, opr_kernel_size=7, iterations=1)
        self.egr_3 = EGR(cur_channel=64, dep_channel=128, focus_background=False, opr_kernel_size=7, iterations=1)

    def forward(self, x):
        rgb_list = self.smt(x)
        r1 = rgb_list[3]
        r2 = rgb_list[2]
        r3 = rgb_list[1]
        r4 = rgb_list[0]

        xf_1 = self.ccm_1(r1)
        xf_1_up = self.up_1(xf_1)
        r2_conv = torch.cat((r2, xf_1_up), dim=1)
        r2_conv = self.dwcon_1(r2_conv)
        xf_2 = self.ccm_2(r2_conv)
        xf_2_up = self.up_2(xf_2)
        r3_conv = torch.cat((r3, xf_2_up), dim=1)
        r3_conv = self.dwcon_2(r3_conv)
        xf_3 = self.ccm_3(r3_conv)
        xf_3_up = self.up_3(xf_3)
        r4_conv = torch.cat((r4, xf_3_up), dim=1)
        r4_conv = self.dwcon_3(r4_conv)
        xf_4 = self.ccm_4(r4_conv)

        pred_1 = self.pred1(xf_1)
        pred_2 = self.egr_1(cur_x=xf_2, dep_x=xf_1, in_map=pred_1)
        pred_3 = self.egr_2(cur_x=xf_3, dep_x=xf_2, in_map=pred_2)
        pred_4 = self.egr_3(cur_x=xf_4, dep_x=xf_3, in_map=pred_3)

        y1 = F.interpolate(pred_1, scale_factor=32, mode='bilinear')
        y2 = F.interpolate(pred_2, scale_factor=16, mode='bilinear')
        y3 = F.interpolate(pred_3, scale_factor=8, mode='bilinear')
        y4 = F.interpolate(pred_4, scale_factor=4, mode='bilinear')
        return y1, y2, y3, y4

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")



import torch.backends.cudnn as cudnn
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    net = CENet().cuda()
    net.eval()
    x = torch.randn(1, 3, 384, 384).cuda()

    flops, params = profile(net, (x,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    import numpy as np
    from time import time
    with torch.no_grad():
        for _ in range(50):
            _ = net(x)

    torch.cuda.synchronize()

    frame_rate = []
    with torch.no_grad():
        for i in range(300):
            torch.cuda.synchronize()
            start = time()
            _ = net(x)
            torch.cuda.synchronize()
            end = time()
            frame_rate.append(1 / (end - start))
    print("FPS:", np.mean(frame_rate))




