import torch
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import random

class self_attention_memory_module(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape

        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)


        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)

        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)

        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SA_Convlstm_cell(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.input_channels = params['hidden_dim']
        self.hidden_dim = params['hidden_dim']
        self.kernel_size = params['kernel_size']
        self.padding = params['padding']
        self.device = params['device']
        self.attention_layer = self_attention_memory_module(params['hidden_dim'], params['att_hidden_dim'],
                                                            self.device)  # 32, 16
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.hidden_dim, out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size, padding=self.padding)
            , nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim))  # (num_groups, num_channels)

    def forward(self, x, hidden):
        h, c, m = hidden
        device = x.device
        h = h.to(device)
        c = c.to(device)
        m = m.to(device)

        combined = torch.cat([x, h], dim=1)  # (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])

        combined_conv = self.conv2d(combined)  # (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o, g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)

    def init_hidden(self, batch_size, img_size):  # h, c, m initalize
        h, w = img_size

        return (torch.zeros(batch_size, self.hidden_dim, h, w),
                torch.zeros(batch_size, self.hidden_dim, h, w),
                torch.zeros(batch_size, self.hidden_dim, h, w))



class Encode2Decode(nn.Module):  # SA_Convlstm
    def __init__(self, params):
        super(Encode2Decode, self).__init__()
        # hyper parameters
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns, self.decoderCells = [], [], []
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_dim']

        # use seq2seq model
        self.img_encode = nn.Sequential(
            nn.Conv2d(in_channels=params['input_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1)
        )

        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['input_dim'])
        )

        for i in range(params['n_layers']):
            params['input_dim'] == params['hidden_dim'] if i == 0 else params['hidden_dim']
            params['hidden_dim'] == params['hidden_dim']
            self.cells.append(SA_Convlstm_cell(params))
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 32, 32)))  # Use layernorm

        self.cells = nn.ModuleList(self.cells)

        self.bns = nn.ModuleList(self.bns)
        self.decoderCells = nn.ModuleList(self.decoderCells)

        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, frames, target, mask_true, is_training, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(batch_size=frames.size(0), img_size=self.img_size)

        if is_training:
            frames = torch.cat([frames, target], dim=1)

        predict_temp_de = []

        for t in range(11):

            if is_training:
                if t < self.input_window_size:
                    x = frames[:, t, :, :, :]
                else:  # sampling schedule
                    x = mask_true[:, t - self.input_window_size, :, :, :] * frames[:, t] + \
                        (1 - mask_true[:, t - self.input_window_size, :, :, :]) * out
            else:
                if t < self.input_window_size:
                    x = frames[:, t, :, :, :]
                else:  # if the mode is testing, the mask is generated by the model itself
                    x = out

            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):
                out, hidden[i] = cell(x, hidden[i])
                out = self.bns[i](out)

            out = self.img_decode(out)
            predict_temp_de.append(out)
        predict_temp_de = torch.stack(predict_temp_de, dim=1)

        predict_temp_de = predict_temp_de[:, 5:, :, :, :]

        return predict_temp_de

    def init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))

        return states


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.LeakyReLU(0.2))

        self.lay2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.LeakyReLU(0.2))

        self.lay3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2))

        self.lay4 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2))

        self.lay5 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dense1 = nn.Linear(256, 1)

    def forward(self, x):
        # input [batch_size, 12(seq_length), channel, 128, 128]
        x = x.view(-1, 1, 12, 128, 128)
        # input [batch_size, channel, 12(seq_length), 128, 128]
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.avgpool(x)
        dense_input = x.contiguous().view(x.size(0), -1)
        dense_output_1 = self.dense1(dense_input)
        return dense_output_1



class generator_loss_function(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.loss_ssim = ssim

    def forward(self, gen_img, target, gen_D):
        L_rec = self.l1_loss(gen_img, target) + self.l2_loss(gen_img, target)


        output_np = gen_img.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        ssim_value = 0
        for i in range(output_np.shape[0]):
            ssim_seq = 0
            for k in range(output_np.shape[1]):
                result = self.loss_ssim(output_np[i, k, 0, :, :] * 255, target_np[i, k, 0, :, :] * 255, data_range=255)
                ssim_seq += result
            ssim_value += ssim_seq / 6

        L_ssim = ssim_value / output_np.shape[0]
        L_adv = -torch.mean(gen_D)

        return L_rec + 1e-2 * (1 - L_ssim) + 1e-4 * L_adv, L_rec, L_ssim, L_adv


def sa_lstm_loss(output, target):
    loss_mae = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_ssim = ssim

    L_rec = loss_mae(output, target) + loss_mse(output, target)

    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_value = 0
    for i in range(output_np.shape[0]):
        ssim_seq = 0
        for k in range(output_np.shape[1]):
            result = loss_ssim(output_np[i, k, 0, :, :] * 255, target_np[i, k, 0, :, :] * 255, data_range=255)
            ssim_seq += result
        ssim_value += ssim_seq / 6

    L_ssim = ssim_value / output_np.shape[0]

    return L_rec + 0.01 * (1 - L_ssim), L_rec, L_ssim
