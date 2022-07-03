import torch
import os
import sys
sys.path.append(os.getcwd())


def l1_reg(y_pred, y_true_conv):
    return torch.abs(y_pred)


def l1_comp(y_pred, y_true_conv):
    return torch.abs(y_true_conv-y_pred)


def mse(y_pred, y_true_conv):
    loss_mse = (y_true_conv-y_pred) ** 2
    return loss_mse


def ll1(y_pred, y_true_conv):
    loss_l1 = torch.abs(torch.log(1+y_true_conv)-torch.log(1+y_pred))
    return loss_l1


def create_manual_loss(conf):
    losses = conf.losses
    loss_weights = conf.loss_weights
    loss_dict = {"l1_comp": l1_comp, "l1_reg": l1_reg, "ll1": ll1}

    def loss_func(y_pred, y_true, span=9, sig=1):
        x_g, y_g, z_g = torch.meshgrid(torch.arange(-span, span + 1), torch.arange(-span, span + 1),
                                       torch.arange(-span, span + 1))
        w = torch.exp(-((x_g ** 2 + y_g ** 2 + z_g ** 2) ** 2 / (2. * sig ** 2))).to(y_true.device)
        y_gaus = torch.nn.functional.conv3d(y_true,
                                            w.unsqueeze(0).unsqueeze(0),
                                            bias=None,
                                            stride=1,
                                            padding=int((w.size()[-1] - 1) / 2),
                                            dilation=1,
                                            groups=1)

        loss = 0
        for loss_type, weight in zip(losses, loss_weights):
            cur_loss = loss_dict[loss_type](y_pred, y_gaus)
            loss += weight * torch.mean(cur_loss)
        return loss

    return loss_func



