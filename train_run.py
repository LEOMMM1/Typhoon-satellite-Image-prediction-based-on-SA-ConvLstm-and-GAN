import torch
from torch import nn
import torchvision
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, Dataset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from data_loader import loaded_train_Dataset, loaded_test_Dataset, loaded_valid_Dataset
from training import train_sa_lstm_with_gan, train_sa_lstm, train_convlstm
from testing import model_test
import gc
from io import BytesIO
from PIL import Image
import zipfile
import matplotlib.pyplot as plt
import numpy as np


def train_model(model_name, params, train_loaders, valid_loaders, device, checkpoint = None, writer = None, is_training = False):
    if model_name == 'convlstm':
        print('Training ConvLSTM model')
        model = train_convlstm(params, train_loaders, valid_loaders, device, checkpoint, writer, is_training)

    if model_name == 'sa_lstm':
        print('Training SA-LSTM model')
        model = train_sa_lstm(params, train_loaders, valid_loaders, device, checkpoint, writer, is_training)

    if model_name == 'sa_lstm_with_gan':
        print('Training SA-LSTM with GAN model')
        model, _ = train_sa_lstm_with_gan(params, train_loaders, device, checkpoint, writer, is_training)

    print('Training finished')
    return model



def plot_test_img(model, zip_file_path, frame_num, device):

    '''
    My dataset is a zip file, so I need to extract the image from the zip file.
    You can change the code according to your dataset.
    '''

    test_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for hour in range(73, 85):
            with zip_ref.open(f'U_Hour_{hour}_Band_09.png') as file:
                data = BytesIO(file.read())
                test_img = Image.open(data)
                test_img = torch.tensor(np.array(test_img)) / 255.0
                test_img = test_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                test_list.append(test_img)

    test_data = torch.cat(test_list, 1).to(device)

    model.eval()
    with torch.no_grad():
        test_input = test_data[:, :frame_num//2, :, :].cuda()
        test_target = test_data[:, frame_num//2:, :, :].cuda()
        test_output = model(test_input, test_target, mask_true=None, is_training=False)


    # Plot the test images
    concat_img = torch.cat([test_output, test_target], 1)
    concat_img_reshape = concat_img.detach().cpu().reshape(-1, 1, 128, 128)  # reshape [B * 6, 1, 128, 128]
    grid = make_grid(concat_img_reshape.float(), nrow=6, padding=1, normalize=True, scale_each=True)

    grid = grid.permute(1, 2, 0)
    grid = grid.numpy().astype(float)

    plt.figure(figsize=(15, 10))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    batch_size = 8
    epochs = 150
    hid_dim = 64
    n_layers = 4
    input_window_size, output = 6, 6
    img_size = (32, 32)
    att_hid_dim = 64
    bias = True
    strides = img_size
    is_load_lstm = False
    sampling_start_value = 1.0
    frame_num = 12 # 12 frames
    time_freq = 1 # the interval between two frames

    # use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = {'input_dim': 1, 'batch_size': batch_size, 'padding': 1, 'device': device, 'train_epoch': epochs,
              'att_hidden_dim': att_hid_dim, 'kernel_size': 3, 'img_size': img_size, 'hidden_dim': hid_dim,
              'n_layers': n_layers, 'output_dim': output, 'input_window_size': input_window_size, 'bias': bias,
              'sampling_start_value': sampling_start_value, 'frame_num': frame_num}


    # load data
    train_dir = 'train_img.zip' # the path of training data
    test_dir = 'test_img.zip' # the path of testing data

    train_HEADS = ['A', 'B', 'C', 'E']
    test_HEADS = ['U', 'V', 'W', 'X', 'Y']

    train_datasets = [None] * len(train_HEADS)
    train_loaders = [None] * len(train_HEADS)

    valid_datasets = [None] * len(train_HEADS)
    valid_loaders = [None] * len(train_HEADS)

    test_datasets = [None] * len(test_HEADS)
    test_loaders = [None] * len(test_HEADS)

    for i in range(len(train_HEADS)):
        train_datasets[i] = loaded_train_Dataset(train_dir, train_HEADS[i], frame_num, time_freq, 128)
        train_loaders[i] = DataLoader(
            train_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True)

        valid_datasets[i] = loaded_valid_Dataset(train_dir, train_HEADS[i], frame_num, time_freq, 128)
        valid_loaders[i] = DataLoader(
            valid_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True)


    for i in range(len(test_HEADS)):
        test_datasets[i] = loaded_test_Dataset(test_dir, test_HEADS[i], frame_num, time_freq, 128)
        test_loaders[i] = DataLoader(
            test_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True)

    print(len(train_datasets[0]), len(train_datasets[1]), len(train_datasets[2]), len(train_datasets[3]))
    print(len(valid_datasets[0]), len(valid_datasets[1]), len(valid_datasets[2]), len(valid_datasets[3]))
    print(len(test_datasets[0]), len(test_datasets[1]), len(test_datasets[2]), len(test_datasets[3]), len(test_datasets[4]))

    del train_datasets, valid_datasets, test_datasets
    gc.collect()

    # =================

    # checkpoint = torch.load('checkpoint_path') # if you want to continue training, you can load the checkpoint
    # writer = SummaryWriter('tf_logs) # use the tensorboard to monitor the training process

    # train model
    model_name = 'sa_lstm_with_gan'
    # model_name = 'sa_lstm' # train the model without GAN
    # model_name = 'convlstm' # train convlstm model

    model = train_model(model_name, params, train_loaders, valid_loaders, device, is_training=False)

    # test model
    model_test(model_name, model, test_loaders, frame_num, batch_size, device)

    # Plot the img
    
    # zip_file_path = 'test_img.zip'
    # plot_test_img(model, zip_file_path, frame_num, device)



