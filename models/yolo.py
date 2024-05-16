# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative
from ultralytics.nn.modules import *
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


#Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape，
# 为后面的计算损失或者NMS作准备
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    # Detect层位于下图中的红框处
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
                detection layer 相当于yolov3中的YOLOLayer层
                :params nc: number of classes
                :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
                :params ch: [128, 256, 512] 3个输出feature map的channel
                """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers Detect的个数
        self.na = len(anchors[0]) // 2  # number of anchors每个feature map的anchor个数 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # register_buffer
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter;
        # 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 对每个输出的feature map都要调用一次conv1x1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # use in-place ops (e.g. slice assignment) 一般都是True 默认不使用AWS Inferentia加速
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                            分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                    1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                        [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        z = []  # inference output
        logits_ = []  # 修改---1
        #对三个feature map分别进行处理
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了记录每个grid的网格坐标 方面后面使用
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                logits = x[i][..., 5:]  # 修改---2
                y = x[i].sigmoid()
                if self.inplace:
                    # 默认执行 不使用AWS Inferentia
                    # 这里的公式和yolov3、v4中使用的不一样 是yolov5作者自己用的 效果更好
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z是一个tensor list 三个元素 分别是[1, 19200, 25] [1, 4800, 25] [1, 1200, 25]
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))  # 修改---3
        return x if self.training else (torch.cat(z, 1), x)
        # return x if self.training else (z[0], x) # (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


# ======================= 解耦头=============================#
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256, 1, 1)
        self.cls_convs1 = Conv(256, 256, 3, 1, 1)
        self.cls_convs2 = Conv(256, 256, 3, 1, 1)
        self.reg_convs1 = Conv(256, 256, 3, 1, 1)
        self.reg_convs2 = Conv(256, 256, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)  # 一个1x1的卷积，把通道数变成类别数，比如coco 80类（主要对目标框的类别，预测分数）
        self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)  # 一个1x1的卷积，把通道数变成4通道，因为位置是xywh
        self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)  # 一个1x1的卷积，把通道数变成1通道，通过一个值即可判断有无目标（置信度）

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)  # 把分类和回归结果按channel维度，即dim=1拼接
        return out


class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class CLLA(nn.Module):
    def __init__(self, range, c):
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
        # x2 = x2.squeeze(3).permute(0, 3, 1, 2).contiguous()
        return (out + x2) / 2
        # return out

class CLLABlock(nn.Module):
    def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0):
        super().__init__()
        self.range = range
        self.c_ = ch
        self.cout = out
        self.conv1 = nn.Conv2d(ch1, self.c_, 1)
        self.conv2 = nn.Conv2d(ch2, self.c_, 1)

        self.att = CLLA(range = range, c = self.c_)

        self.det = nn.Conv2d(self.c_, out, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        f = self.att(x1, x2)

        return self.det(f)


class CLLADetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.det = CLLABlock(range = 2, ch = ch[0], ch1 = ch[0], ch2 = ch[1], out = self.no * self.na)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[2:])  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        p = []
        for i in range(self.nl):
            if i == 0:
                p.append(self.det(x[0], x[1]))
            else:
                p.append(self.m[i-1](x[i+1]))  # conv
            bs, _, ny, nx = p[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != p[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = p[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return p if self.training else (torch.cat(z, 1), p)
        # return x if self.training else (z[0], x) # (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        """
                :params cfg:模型配置文件
                :params ch: input img channels 一般是3 RGB文件
                :params nc: number of classes 数据集的类别个数
                :anchors: 一般是None
                """
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, errors='ignore') as f:
                # model dict  取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)  # model dict


        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.model 初始化整个网络模型
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # default class names ['0', '1', '2',..., '19']
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect) or isinstance(m, CLLADetect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s),visualize=False)])  # forward
            # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            self.stride = m.stride
            # only run once 初始化偏置
            if isinstance(m, Detect) or isinstance(m, CLLADetect):
                self._initialize_biases()  # only run once
        if isinstance(m, Decoupled_Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), visualize=False)])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            try:
                self._initialize_biases()  # only run once
                LOGGER.info('initialize_biases done')
            except:
                LOGGER.info('decoupled no biase ')


        # Init weights, biases
        # 调用torch_utils.py下initialize_weights初始化模型权重
        initialize_weights(self)
        # 打印模型信息
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # augmented inference, None  上下flip/左右flip
        # 是否在测试时也使用数据增强  Test Time Augmentation(TTA)
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        # 正常前向推理
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 1, 0.83, 0.83, 0.67, 0.67]  # scales
        f = [None, 3, None, 3, None, 3]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # _descale_pred将推理结果恢复到相对原图图片尺寸
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
                :params x: 输入图像
                :params profile: True 可以做一些性能评估
                :params visualize: True 可以做一些特征可视化
                :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                               分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                """
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        for m in self.model:
            # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
            # if not from previous layer   m.f=当前层的输入来自哪一层的输出  s的m.f都是-1
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 打印日志信息  FLOPs time等
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = False
            x = m(x)  # run 正向推理
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                # print(visualize)
                # 特征可视化 可以自己改动想要哪层的特征进行可视化
                if m.type =='models.common.C3_CBAM' and m.i in [22,26,30,34]:
                    feature_visualization(x, m.type, m.i,save_dir=Path('runs/val/exp'))
                elif m.type =='models.common.SPPF':
                    feature_visualization(x, m.type, m.i,save_dir=Path('runs/val/exp'))

        return x

    def _descale_pred(self, p, flips, scale, img_size):
        """
        将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
               de-scale predictions following augmented inference (inverse operation)
               :params p: 推理结果
               :params flips:
               :params scale:
               :params img_size:
               """
        # 不同的方式前向推理使用公式不同 具体可看Detect函数
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace: # 默认执行 不使用AWS Inferentia
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        # 打印日志信息  FLOPs time等
        c = isinstance(m, Detect) or isinstance(m, Decoupled_Detect) # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module

        if isinstance(m, Detect) or isinstance(m, Decoupled_Detect):
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        else:
            for mi, s in zip(m.m, m.stride[1:]):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b = m.det.det.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / m.stride[0]) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            m.det.det.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        # 打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        m = self.model[-1]  # Detect() module
        if getattr(m.m, 'bias', False):
            for mi in m.m:  # from
                b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
                LOGGER.info(
                    ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        else:
            for mi in m.m:  # from
                b1 = mi.cls.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
                b2 = mi.bbox.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
                LOGGER.info(
                    ('%6g Conv2d.bias and %6g Conv2d.bias:' + '%10.3g' * 6) % (mi.bbox.weight.shape[1], mi.cls.weight.shape[1], *b2[:].mean(1).tolist(), b1[:].mean()))


    # def _print_weights(self):
    # 打印模型中Bottleneck层的权重参数weights信息(也可以任选哪些层weights信息)
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """用在detect.py、val.py
                fuse model Conv2d() + BatchNorm2d() layers
                调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
                """
        LOGGER.info('Fusing layers... ')
        # 遍历每一层结构
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv 融合
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播 update forward (反向传播不用管, 因为这种推理只用在推理阶段)
                m.forward = m.forward_fuse  # update forward
        self.info() # 打印conv+bn融合后的模型信息
        return self

    def autoshape(self):  # add AutoShape module
        # 直接调用common.py中的AutoShape模块  也是一个扩展模型功能的模块
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model扩展模型功能 此时模型包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        #  调用torch_utils.py下model_info函数打印模型信息
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Decoupled_Detect)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    """用在上面Model模块中
        解析模型文件(字典形式)，并搭建网络结构
        这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                              使用当前层的参数搭建当前层 =>
                              生成 layers + save
        :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
        :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
        :return nn.Sequential(*layers): 网络的每一层的层结构
        :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]
        """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: number of anchors 每一个predict head上的anchor数 = 3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: number of outputs 每一个predict head层的输出channel = anchors * (classes + 5) = 75(VOC)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # from(当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3STR, C3SPP, C3Ghost, ASPP, CBAM, BAM , nn.ConvTranspose2d,DWConvTranspose2d, C3_CBAM,C3_BAM,C3_CA, C3_SCBAM,
                 C3CR, C2f,C2fBAM, SPPELAN, SPPCSPC, ResBlock_CBAM, se_block,RepVGGBlock, ADown,C3x, RepC3,C2fAttn,Conv2Former]:
            # c1: 当前层的输入的channel数
            # c2: 当前层的输出的channel数(初定)
            # ch: 记录着所有层的输出channel
            c1, c2 = ch[f], args[0]
            # if not output  no=75  只有最后一层c2=no  最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # 控制宽度  如v5s: c2*0.5  c2: 当前层的最终输出的channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)
            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            # [in_channel, out_channel, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR, 则需要在args中加入bottleneck的个数
            # [in_channel, out_channel, Bottleneck的个数n, bool(True表示有shortcut 默认，反之无)]
            if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost,C3CR, C2f,C2fBAM, Conv2Former,SPPCSPC,C2fAttn, C3x, RepC3,C3_CBAM,C3_BAM, C3_SCBAM]:
                args.insert(2, n)  # number of repeats，在第二个位置插入bottleneck个数n
                n = 1  # 恢复默认值1
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        elif m in [SimAM]:
            args = [*args[:]]
        elif m in [ODConv_3rd,ODConv]:
            #[-1, 1, ODConv_3rd, [512, 3, 2,2]],
            #[-1, 1, ODConv, [1024]]
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]

        elif m is DownSimper:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2]
        elif m is CARAFE:
            c2 = ch[f]
            args = [c2, *args]
        # 添加bifpn_add结构
        elif m in [BiFPN_Add2, BiFPN_Add3]:
            c2 = max([ch[x] for x in f])
        elif m is CAM:
            #[-1, 1, CAM, ['concat']],  # 11
            c1, c2 = ch[f], (ch[f] * 3 if args[0] == 'concat' else ch[f])
            args = [c1, args[0]]
        elif m is BAM:
            args = [ch[f]]
        elif m in [S2Attention]:
            c1 = ch[f]
            args = [c1]
        elif m in [NAMAttention]:
            c1 = ch[f]
            args = [c1]
        elif m in [DySample]:
            args.insert(0, ch[f])
        elif m is BiFusion:
            c2 = args[3]
            print(c2)
        elif m is Zoom_cat:
            c2 = 3 * args[0]
        elif m is attention_model:
            c2 = args[0]
        elif m is ScalSeq:
            c2 = args[0]
        elif m is SF:
            c2 = sum(ch[x] for x in f)
            print(c2)
        elif m in [SKAttention]:  # channels args
            c2 = ch[f]
            args = [c2, *args[0:]]
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        elif m is CShortcut:
            c2 = sum(ch[x] for x in f)
        elif m is space_to_depth:
            c2 = 4 * ch[f]
            # 将输入通道数 ch[f] 增加了四倍
            # 4 是基于 SPD 的 block_size 为 2 的假设。如果 block_size 有所不同，这个数字应该是 block_size 的平方
            # ch 是一个包含前面所有层输出通道数的列表，f 是指向前面某层的索引
        elif m is Detect: # Detect（YOLO Layer）层
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is CLLADetect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * (len(f) - 1)
        elif m is CoorAttention:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]

        elif m is Decoupled_Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is eca_block:
            args=[*args[:]]
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            # Upsample args不变
            c2 = ch[f]
        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params 计算这一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # append to savelist  把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel [3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model 创建一个模型
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter('.')
    LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
