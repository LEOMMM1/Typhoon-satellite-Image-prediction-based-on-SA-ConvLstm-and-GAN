import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, params):


        super(ConvLSTMCell, self).__init__()

        self.input_dim = params['hidden_dim']
        self.hidden_dim = params['hidden_dim']

        self.kernel_size = params['kernel_size']
        self.padding = params['kernel_size'] // 2, params['kernel_size'] // 2
        self.bias = params['bias']

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLstmEncode2Decode(nn.Module):
    def __init__(self, params):
        super(ConvLstmEncode2Decode, self).__init__()
        
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns, self.decoderCells = [], [], []
        self.n_layers = params['n_layers']

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
            self.cells.append(ConvLSTMCell(params)) 
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 32, 32)))

        self.cells = nn.ModuleList(self.cells)

        self.bns = nn.ModuleList(self.bns)
        self.decoderCells = nn.ModuleList(self.decoderCells)

        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, frames, hidden=None):
        if hidden == None:
            hidden = self._init_hidden(batch_size=frames.size(0), img_size=self.img_size)

        predict_temp_de = []

        for t in range(6):
            x = frames[:, t, :, :, :]

            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):
                out, hidden[i] = cell(x, hidden[i])
                out = self.bns[i](out)

            out = self.img_decode(out)
            predict_temp_de.append(out)
        predict_temp_de = torch.stack(predict_temp_de, dim=1)

        return predict_temp_de

    def _init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))

        return states


if __name__ == '__main__':
    batch_size = 8
    epochs = 100
    hid_dim = 64
    n_layers = 4
    input_window_size, output = 6, 6
    img_size = (32, 32)
    att_hid_dim = 64
    bias = True
    strides = img_size
    sampling_start_value = 1.0
    params = {'input_dim': 1, 'batch_size': batch_size, 'padding': 1, 'device': 'cuda',
              'att_hidden_dim': att_hid_dim, 'kernel_size': 3, 'img_size': img_size, 'hidden_dim': hid_dim,
              'n_layers': n_layers, 'output_dim': output, 'input_window_size': input_window_size, 'bias': bias}
    convlstm = ConvLstmEncode2Decode(params)
    x = torch.rand(8, 6, 1, 128, 128)
    output = convlstm(x)
    print(output.shape)
