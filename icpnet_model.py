'''ICPNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as eo
# from timm.models.layers import DropPath
import numpy as np


def shuffle(x, groups):
    _, c, _, _ = x.shape
    assert c % groups == 0, 'Channels are not divisible by groups'
    # channels_per_group = int(c / groups)
    x = eo.rearrange(x, 'b (groups channels_per_group) h w -> b groups channels_per_group h w', groups=groups)
    x = eo.rearrange(x, 'b groups channels_per_group h w -> b channels_per_group groups h w')
    x = eo.rearrange(x, 'b channels_per_group groups h w -> b (channels_per_group groups) h w')

    return x


# conv_stem
class features(nn.Module):
    def __init__(self, ics, ocs, bias=True):
        super(features, self).__init__()
        self.conv_5 = nn.Conv2d(ics, ocs, 3, 2, 2, bias=bias, dilation=2)
        self.conv_9 = nn.Conv2d(ics, ocs, 3, 2, 4, bias=bias, dilation=4)
        self.conv_13 = nn.Conv2d(ics, ocs, 3, 2, 6, bias=bias, dilation=6)
        self.conv_15 = nn.Conv2d(ics, ocs, 3, 2, 7, bias=bias, dilation=7)

        self.layernorm = LayerNorm(ocs)

    def forward(self, x):
        y = self.layernorm(self.conv_5(x) + self.conv_9(x) + self.conv_13(x) + self.conv_15(x))
        y = shuffle(y, 8)
        return y  # (B, C, H/2, W/2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  #(channels, 1, 1)
            return x


class Conv_Pooling(nn.Module):
    def __init__(self, ics=64, ocs=128, dim=112*112):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.reduction = nn.Conv2d(4 * ics, ocs, 1, 1, 0)

    def forward(self, x):
        '''
        x: B, C, H, W, this class function come from Swin
        '''
        _, H, W, _ = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
        b, c, h, w = x.shape
        x = self.layernorm(x.reshape(b, c, -1)).reshape(b, c, h, w)
        x = self.reduction(x)

        return x  # (B, C, H/2, W/2)


class Depthwise_separable_conv(nn.Module):
    def __init__(self, ics, ocs, kernel_sizes=3, padding=1, dim=112*112, rate_dilation=1):
        super(Depthwise_separable_conv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=ics, out_channels=ics, kernel_size=kernel_sizes, padding=padding, stride=1, groups=ics, dilation=rate_dilation)
        # self.LN_1 = LayerNorm(ics)
        self.LN_1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(in_channels=ics, out_channels=ocs, kernel_size=1, padding=0, stride=1)
        # self.LN_2 = LayerNorm(ocs)
        self.LN_2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_1(x)
        x = self.act(self.LN_1(x.reshape(b, c, -1)).reshape(b, c, h, w))
        x = self.conv_2(x)
        b, c, h, w = x.shape
        x = self.LN_2(x.reshape(b, c, -1)).reshape(b, c, h, w)

        return x


class IPA(nn.Module):
    def __init__(self, ics, ocs, dim, i=0):
        super(IPA, self).__init__()
        # up-sampling
        self.conv_alignment = nn.Conv2d(ics, ocs, 1, 1, 0, 1)
        self.LN_1 = nn.LayerNorm(dim)
        self.i = i
        if self.i == 0:
            self.upsample = nn.ConvTranspose2d(ics, ics, 3, 2, 1, 1)  # dilation=1: h'=(h-1)s+k-2p+out_padding

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(ocs, int(ocs*4), 1, 1, 0)
        self.conv2 = nn.Conv2d(int(ocs*4), ocs, 1, 1, 0)
        self.act_1 = nn.GELU()

        # spatial attention
        self.conv_ver_1 = nn.Conv2d(2, 1, (7, 3), 1, (3, 1))
        self.conv_ver_2 = nn.Conv2d(2, 1, (13, 5), 1, (6, 2))
        self.conv_hor_1 = nn.Conv2d(2, 1, (3, 7), 1, (1, 3))
        self.conv_hor_2 = nn.Conv2d(2, 1, (5, 13), 1, (2, 6))

    def forward(self, input_1, input_2):  # input_1（B, C, H/2, W/2），input_2 (B, 2C, H/4, W/4)
        # up-sampling
        if self.i == 0:
            input_2 = self.upsample(input_2)
            input_2 = self.conv_alignment(input_2)
            b, c, h, w = input_2.shape
            input_2 = self.LN_1(input_2.reshape(b, c, -1)).reshape(b, c, h, w)

        # channel_attention
        output = (self.conv2(self.act_1(self.conv1(self.avg_pool(input_1)))) + self.conv2(self.act_1(self.conv1(self.max_pool(input_1))))).sigmoid() * input_1 + input_2

        # spatial attention
        max_map, _ = torch.max(output, dim=1, keepdim=True)
        avg_map = torch.mean(output, dim=1, keepdim=True)
        maps = torch.cat([max_map, avg_map], dim=1)
        hor_map = self.conv_hor_1(maps) + self.conv_hor_2(maps)
        ver_map = self.conv_ver_1(maps) + self.conv_ver_2(maps)
        output = (hor_map + ver_map).sigmoid() * output + output
        output = shuffle(output, 8)
        return output


class block_1(nn.Module):
    def __init__(self, ics, ocs, extension=2, kernel_size=(5, 3), padding=(2, 1), dim=112*112, i=0, rate_dilation=(1, 1)):
        super(block_1, self).__init__()
        self.i = i
        if i == 0:
            self.dw_1 = Depthwise_separable_conv(ics, ocs, kernel_size[0], padding[0], dim, rate_dilation[0])
            self.dw_2 = Depthwise_separable_conv(ocs, ocs*extension, kernel_size[1], padding[1], dim, rate_dilation[1])
            self.pwconv = nn.Conv2d(ocs*extension, ics, 1, 1, 0)
        else:
            self.conv11 = nn.Conv2d(ics, ocs, 1, 1, 0)
            self.LN_1 = nn.LayerNorm(dim)
            self.act = nn.GELU()
            self.dw_1 = Depthwise_separable_conv(ics, ocs, kernel_size[0], padding[0], dim, rate_dilation[0])
            self.dw_2 = Depthwise_separable_conv(ocs, ocs*extension, kernel_size[1], padding[1], dim, rate_dilation[1])
            self.pwconv = nn.Conv2d(ocs*extension, ocs, 1, 1, 0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.i == 0:
            input = x
            x = self.dw_1(x)
            x = self.dw_2(x)
            x = self.pwconv(x)
        else:
            x = self.conv11(x)
            b, c, h, w = x.shape
            x = self.act(self.LN_1(x.reshape(b, c, -1)).reshape(b, c, h, w))
            input = x
            x = self.dw_1(x)
            x = self.dw_2(x)
            x = self.pwconv(x)
        x = input + self.dropout(x)

        return x


class block_2(nn.Module):
    def __init__(self, ics, ocs, extension=2, kernel_size=4, padding=3, dim=112*112, i=0, rate_dilation=2):
        super(block_2, self).__init__()
        self.i = i
        if i == 0:
            self.dw_1 = Depthwise_separable_conv(ics, ics*extension, kernel_size, padding, dim, rate_dilation)
            self.pwconv = nn.Conv2d(ics*extension, ocs, 1, 1, 0)
        else:
            self.conv11 = nn.Conv2d(ics, ocs, 1, 1, 0)
            self.LN_1 = nn.LayerNorm(dim)
            self.act = nn.GELU()
            self.dw_1 = Depthwise_separable_conv(ocs, ocs*extension, kernel_size, padding, dim, rate_dilation)
            self.pwconv = nn.Conv2d(ics*extension, ocs, 1, 1, 0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.i == 0:
            input = x
            x = self.dw_1(x)
            x = self.pwconv(x)
            x = input + self.dropout(x)
        else:
            x = self.conv11(x)
            b, c, h, w = x.shape
            x = self.act(self.LN_1(x.reshape(b, c, -1)).reshape(b, c, h, w))
            # input = x
            x = self.dw_1(x)
            x = self.pwconv(x)
        # x = input + self.dropout(x)

        return x


class block_3(nn.Module):
    def __init__(self, ics, ocs, kernel_size=4, padding=3, dim=112*112, rate_dilation=2, i=1):
        super(block_3, self).__init__()
        self.i = i
        self.conv11 = nn.Conv2d(ics, ocs, 1, 1, 0)
        self.LN_1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dw_1 = Depthwise_separable_conv(ics, ocs, kernel_size, padding, dim, rate_dilation)

    def forward(self, x):
        if self.i == 0:
            x = self.conv11(x)
            b, c, h, w = x.shape
            x = self.act(self.LN_1(x.reshape(b, c, -1)).reshape(b, c, h, w))
            x = self.dw_1(x)
        else:
            x = self.dw_1(x)
        return x


class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dim=112*112):
        super(adap_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.LN = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.act(self.LN(x.reshape(b, c, -1)).reshape(b, c, h, w))
        return x


class Refine_block(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False, dim=[112*112, 56*56]):
        super(Refine_block, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel, dim=dim[0])
        self.pre_conv2 = adap_conv(in_channel[1], out_channel, dim=dim[1])
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor

    def forward(self, input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2


class ICPNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ICPNet, self).__init__()
        self.ic = [3,  64, 128, 256, 512]  # input chanels
        self.oc = [64, 64, 128, 256, 512]  # output chanels
        self.hw = [112*112, 56*56, 28*28, 14*14]

        self.convpooling_1 = Conv_Pooling(ics=self.ic[1], ocs=self.oc[2], dim=self.hw[1])
        self.convpooling_2 = Conv_Pooling(ics=self.ic[2], ocs=self.oc[3], dim=self.hw[2])
        self.convpooling_3 = Conv_Pooling(ics=self.ic[3], ocs=self.oc[4], dim=self.hw[3])

        self.convpooling_d1 = Conv_Pooling(ics=self.ic[1], ocs=self.oc[2], dim=self.hw[1])
        self.convpooling_d2 = Conv_Pooling(ics=self.ic[2], ocs=self.oc[3], dim=self.hw[2])
        self.convpooling_d3 = Conv_Pooling(ics=self.ic[3], ocs=self.oc[4], dim=self.hw[3])

        self.IPA_FP = IPA(self.oc[1], self.ic[1], dim=self.hw[0], i=1)
        self.IPA_1 = IPA(self.oc[2], self.ic[1], dim=self.hw[0])
        self.IPA_2 = IPA(self.oc[3], self.ic[2], dim=self.hw[1])
        self.IPA_3 = IPA(self.oc[4], self.ic[3], dim=self.hw[2])
        self.IPA_d1 = IPA(self.ic[2], self.oc[2], dim=self.hw[1], i=1)
        self.IPA_d2 = IPA(self.ic[3], self.oc[3], dim=self.hw[2], i=1)
        self.IPA_d3 = IPA(self.ic[4], self.oc[4], dim=self.hw[3], i=1)

        for i in range(len(self.ic)):
            if i == 0:
                self.conv_99 = features(self.ic[0], self.oc[0])
            if i == 1:
                self.s1_g = block_1(self.ic[i], self.oc[i], kernel_size=(5, 3), padding=(2, 2), dim=self.hw[0], rate_dilation=(1, 2))
                self.s1_sg = block_2(self.ic[i], self.oc[i], kernel_size=4, padding=3, dim=self.hw[0], rate_dilation=2)
                self.s1_ig = block_3(self.ic[i], self.oc[i], kernel_size=4, padding=3, dim=self.hw[0], rate_dilation=2)
            if i == 2:
                self.s2_g = block_3(self.ic[i], self.oc[i], kernel_size=6, padding=5, dim=self.hw[1], rate_dilation=2)
                self.s2_sg = block_1(self.ic[i], self.oc[i], kernel_size=(7, 3), padding=(6, 6), dim=self.hw[1], rate_dilation=(2, 6))
                self.s2_ig = block_2(self.ic[i], self.oc[i], kernel_size=7, padding=6, dim=self.hw[1], rate_dilation=2)
            if i == 3:
                self.s3_g = block_3(self.ic[i], self.oc[i], kernel_size=7, padding=9, dim=self.hw[2], rate_dilation=3)
                self.s3_sg = block_1(self.ic[i], self.oc[i], kernel_size=(9, 3), padding=(12, 12), dim=self.hw[2], rate_dilation=(3, 12))
                self.s3_ig = block_2(self.ic[i], self.oc[i], kernel_size=9, padding=12, dim=self.hw[2], rate_dilation=3)
            if i == 4:
                self.s4_g = block_3(self.ic[i], self.oc[i], kernel_size=7, padding=9, dim=self.hw[3], rate_dilation=3)
                self.s4_sg = block_1(self.ic[i], self.oc[i], kernel_size=(9, 3), padding=(12, 12), dim=self.hw[3], rate_dilation=(3, 12))
                self.s4_ig = block_2(self.ic[i], self.oc[i], kernel_size=9, padding=12, dim=self.hw[3], rate_dilation=3)

        self.LNend = nn.LayerNorm(self.oc[-1])
        self.linear = nn.Linear(self.oc[-1], num_classes)

        # edge detection decoding
        self.level_1 = Refine_block((256, 512), 256, 2, dim=[self.hw[2], self.hw[3]])
        self.level_2 = Refine_block((128, 256), 128, 2, dim=[self.hw[1], self.hw[2]])
        self.level_3 = Refine_block((64, 128), 64, 2, dim=[self.hw[0], self.hw[1]])
        self.output = nn.Conv2d(64, 1, 3, 1, 1)
        self.up_weight = nn.Parameter(bilinear_upsample_weights(2, self.oc[0]), requires_grad=False)
        self.act = nn.GELU()

        self._initialize_weights()

    def forward(self, x, states):
        x_2, x_3, x_4 = states  # (b, 128, h//4, w//4), (b, 256, h//8, w//8), (b, 512, h//16, w//16)
        b, _, h, w = x.shape
        if x.shape[1] == 1:
            x = x.expand(b, 3, h, w)
        else:
            b, _, _, _ = x.shape
        feature_stem = self.conv_99(x)  # conv_stem
        feature_stem = self.IPA_FP(feature_stem, feature_stem)

        for i in range(2):
            # stage 1
            if i == 0:
                g_1 = self.s1_g(feature_stem)
            x = self.IPA_1(g_1, x_2)
            x = self.s1_sg(x)
            if i == 1:
                x_1 = self.s1_ig(x)
            # pooling
            x = self.convpooling_1(x)
            x = shuffle(x, 8)

            # stage 2
            x = self.s2_g(x)
            x = self.IPA_2(x, x_3)
            x = self.s2_sg(x)
            x_2 = self.s2_ig(x)  # h 192 h/2 w/2
            # pooling
            x = self.convpooling_2(x)
            x = shuffle(x, 8)

            # stage 3
            x = self.s3_g(x)
            x = self.IPA_3(x, x_4)
            x = self.s3_sg(x)
            x_3 = self.s3_ig(x)  # h 384 h/4 w/4
            # pooling
            x = self.convpooling_3(x)
            x = shuffle(x, 8)

            # stage 4
            x = self.s4_g(x)
            x = self.s4_sg(x)
            x_4 = self.s4_ig(x)  # h 768 h/8 w/8

        out_1 = self.IPA_d1(self.convpooling_d1(x_1), x_2)  # b, 128, h/2, w/2
        out_2 = self.IPA_d2(self.convpooling_d2(out_1), x_3)  # b, 256, h/4, w/4
        out_3 = self.IPA_d3(self.convpooling_d3(out_2), x_4)  # b, 512, h/4, w/4
        out = F.avg_pool2d(out_3, out_3.shape[-1])  # (b, c, 1, 1)
        out = self.LNend(out.reshape(b, -1))
        out = self.linear(out)  # (b, num_classes)] 

        # edge detection decoding
        e_2 = self.level_1([out_2, out_3])
        e_1 = self.level_2([out_1, e_2])
        e_1 = self.level_3([feature_stem, e_1])

        # output
        e_1 = F.conv_transpose2d(e_1, self.up_weight, stride=2, padding=int(2/2), output_padding=(h-e_1.size(2)*2, w-e_1.size(3)*2))
        e_1 = self.output(e_1).sigmoid()

        return out, e_1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0, 0.02) 
                m.bias.data.zero_()  # type: ignore
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.02)
                m.bias.data.zero_()
        print('Weights Initialization complete!')


def learning_rate_decay(optimizer, epoch, decay_rate=0.1):
    for param_group in optimizer.param_groups:
        if epoch == 0:
            break
        elif epoch == 20:
            param_group['lr'] = param_group['lr'] * decay_rate
        elif epoch == 50:
            param_group['lr'] = param_group['lr'] * decay_rate
        elif epoch == 80:
            param_group['lr'] = param_group['lr'] * decay_rate
        else:
            pass


class edge_cls_loss(nn.Module):
    def __init__(self, reduction, label_smoothing: float = 0.0):
        super(edge_cls_loss, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target, pred=0, labels=0, flag='train'):
        cls_loss = F.cross_entropy(input, target, reduction=self.reduction, label_smoothing=self.label_smoothing)
        if flag == 'train':
            edge_loss = cross_entropy_per_image(pred, labels)
            total_loss = cls_loss + 0.01 * edge_loss
            return total_loss, cls_loss, edge_loss
        else:
            return cls_loss

def cross_entropy_per_image(preds, labels):
    total_loss = 0.0
    for _, (_pred, _label) in enumerate(zip(preds, labels)):
        total_loss += cross_entropy_with_weight_original(_pred, _label)
    return total_loss / len(preds)


def cross_entropy_with_weight_original(logits, labels, threshold=0.2, weight=1):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > threshold].clamp(eps, 1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    weight_pos = len(pred_neg)/(len(pred_neg)+len(pred_pos))
    weight_neg = len(pred_pos)/(len(pred_neg)+len(pred_pos))
    cross_entropy = (-weight_pos * pred_pos.log()).sum() + (-weight * weight_neg * (1.0 - pred_neg).log()).sum()

    return cross_entropy


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)
