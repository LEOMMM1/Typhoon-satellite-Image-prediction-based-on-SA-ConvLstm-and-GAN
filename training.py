import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd

from PIL import Image
import numpy as np
import glob
import gc
import re
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from convlstm_model import ConvLstmEncode2Decode
from sa_lstm_model import Encode2Decode, Discriminator, generator_loss_function, sa_lstm_loss

# create the mask for the input
def schedule_sampling(eta, itr, batch_size = 8, total_length = 12, input_length = 6, img_width = 128, img_channel = 1, sampling_stop_iter = 50000, sampling_changing_rate = 0.00002):
    zeros = np.zeros((batch_size,
                      total_length - input_length - 1,
                      img_channel,
                      img_width,
                      img_width))


    if itr < sampling_stop_iter:
        eta -= sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, total_length - input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_channel,
                    img_width,
                    img_width))
    zeros = np.ones((img_channel,
                    img_width,
                    img_width))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(total_length - input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  total_length - input_length - 1,
                                  img_channel,
                                  img_width,
                                  img_width))
    return eta, real_input_flag


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha =  torch.rand(real_samples.size(0), 1, 1, 1, 1).cuda().expand_as(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_l2norm = gradients.norm(2, dim=[1,2,3,4])
    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)
    return gradient_penalty




# train the model with gan
def train_sa_lstm_with_gan(params, train_loaders, device, checkpoint = None, writer = None, is_training = False):
    frame_num = params['frame_num']
    batch_size = params['batch_size']
    HEADS = ['A', 'B', 'C', 'E']
    generator = Encode2Decode(params).to(device)
    discriminator = Discriminator().to(device)
    criterion_G = generator_loss_function()

    if is_training:
        # if checkpoint is not None, load the checkpoint
        if checkpoint is not None:
            generator_checkpoint = checkpoint[0]
            discriminator_checkpoint = checkpoint[1]

            generator.load_state_dict(generator_checkpoint['model_state_dict'])
            discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
            eta = generator_checkpoint['eta']
            start_epoch = generator_checkpoint['epoch'] + 1
            itr = generator_checkpoint['itr']   # model iteration

            # load optimizer
            optimizer_G = generator_checkpoint['optimizer_state_dict']
            optimizer_D = discriminator_checkpoint['optimizer_state_dict']


        else:
            eta = params['sampling_start_value']
            start_epoch = 0
            itr = 0
            optimizer_G = Adam(generator.parameters(), lr=0.002, betas=(0.9, 0.999))
            optimizer_D = Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

        # start training
        lambda_gp = 10 # the weight of gradient penalty


        generator.train()
        discriminator.train()

        train_epoch = params['train_epoch']
        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            start_time = time.time()

            total_loss_D = []
            total_loss_G = []

            for headid in range(len(HEADS)):
                for ind, (ids, frames) in enumerate(train_loaders[headid]):

                    input_list = []
                    for input_tensor in frames[:-(frame_num // 2)]:
                        input_list.append(input_tensor.unsqueeze(1))
                    input = torch.cat(input_list, 1).to(device)  # (batch, seq, channel, height, width)

                    if input.size(0) != batch_size:
                        continue

                    target_list = []
                    for target_tensor in frames[-(frame_num // 2):]:
                        target_list.append(target_tensor.unsqueeze(1))
                    target = torch.cat(target_list, 1).to(device)

                    eta, real_input_flag = schedule_sampling(eta, itr)
                    real_input_flag = torch.Tensor(real_input_flag).cuda()

                    #############start training gennerater and discriminator#######################

                    optimizer_D.zero_grad()
                    gen_img = generator(input, target, real_input_flag, is_training=True)

                    pred_discri_input = torch.cat([input, gen_img], 1)
                    true_discri_input = torch.cat([input, target], 1)

                    gen_D = discriminator(pred_discri_input.detach())
                    true_D = discriminator(true_discri_input)

                    gradient_penalty = compute_gradient_penalty(discriminator, true_discri_input.data,
                                                                pred_discri_input.data)
                    loss_D = -torch.mean(true_D) + torch.mean(gen_D) + lambda_gp * gradient_penalty
                    loss_D.backward()
                    optimizer_D.step()

                    total_loss_D.append(loss_D.item())

                    if writer is not None:
                        writer.add_scalar('Loss/sa_lstm_wgan_loss_D', loss_D.item(), itr)
                    optimizer_G.zero_grad()
                    if ind % 2 == 0:
                        output = discriminator(pred_discri_input)

                        loss_G, L_rec, L_ssim, L_adv = criterion_G(gen_img, target, output)

                        loss_G.backward()
                        optimizer_G.step()
                        total_loss_G.append(loss_G.item())
                        if writer is not None:
                            writer.add_scalar('Loss/sa_lstm_wgan_loss_G', loss_G.item(), itr)
                            writer.add_scalar('Loss/sa_lstm_wgan_L_rec', L_rec.item(), itr)
                            writer.add_scalar('Loss/sa_lstm_wgan_L_ssim', L_ssim.item(), itr)
                            writer.add_scalar('Loss/sa_lstm_wgan_L_adv', L_adv.item(), itr)

                    itr += 1

            end_time = time.time()
            print('Epoch: ', epoch + 1 + start_epoch, ' | time: ', end_time - start_time, ' | D loss: ',
                  sum(total_loss_D) / len(total_loss_D), ' | G loss: ', sum(total_loss_G) / len(total_loss_G))

            # save the model checkpoint every 5 epoch
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + start_epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'itr': itr,
                    'eta': eta
                }, 'sa_lstm_wgan_generator_checkpoint_' + str(epoch + 1 + start_epoch) + '.pt')

                torch.save({
                    'epoch': epoch + start_epoch,
                    'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizer_D.state_dict(),
                    'itr': itr,
                    'eta': eta
                }, 'sa_lstm_wgan_discriminator_checkpoint_' + str(epoch + 1 + start_epoch) + '.pt')

                # save the latest generated image
                gen_img_reshape = gen_img.detach().cpu().reshape(-1, 1, 128, 128)  # 重塑为 [B*6, 1, 128, 128]
                save_image(gen_img_reshape.float().data[:48], f"sa_lstm_wgan_epoch{epoch + 1 + start_epoch}.png", nrow=6,
                           normalize=True)

        return generator, discriminator

    else:
        model = Encode2Decode(params).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, None




def train_sa_lstm(params, train_loaders, valid_loaders, device, checkpoint = None, writer = None, is_training = False):

    frame_num = params['frame_num']
    batch_size = params['batch_size']
    HEADS = ['A', 'B', 'C', 'E']
    model = Encode2Decode(params).to(device)
    criterion = sa_lstm_loss

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        eta = checkpoint['eta']
        start_epoch = checkpoint['epoch'] + 1
        itr = checkpoint['itr']   


        optimizer = checkpoint['optimizer_state_dict']


    else:
        eta = params['sampling_start_value']
        start_epoch = 0
        itr = 0
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if is_training:

        train_epoch = params['train_epoch']
        for epoch in range(train_epoch):
            model.train()

            torch.cuda.empty_cache()
            start_time = time.time()
            total_train_loss = []
            total_valid_mse_loss = []

            for headid in range(len(HEADS)):
                for ind, (ids, frames) in enumerate(train_loaders[headid]):

                    input_list = []
                    for input_tensor in frames[:-(frame_num // 2)]:
                        input_list.append(input_tensor.unsqueeze(1))
                    input = torch.cat(input_list, 1).to(device)  # (batch, seq, channel, height, width)

                    if input.size(0) != batch_size:
                        continue

                    target_list = []
                    for target_tensor in frames[-(frame_num // 2):]:
                        target_list.append(target_tensor.unsqueeze(1))
                    target = torch.cat(target_list, 1).to(device)

                    eta, real_input_flag = schedule_sampling(eta, itr)
                    real_input_flag = torch.Tensor(real_input_flag).cuda()

                    optimizer.zero_grad()
                    train_output = model(input, target, real_input_flag, is_training=True)

                    loss, L_rec, L_ssim = criterion(train_output, target)
                    loss.backward()
                    optimizer.step()

                    total_train_loss.append(loss.item())
                    if writer is not None:
                        writer.add_scalar('Loss/sa_lstm_loss', loss.item(), itr)
                        writer.add_scalar('Loss/sa_lstm_L_rec', L_rec.item(), itr)
                        writer.add_scalar('Loss/sa_lstm_L_ssim', L_ssim.item(), itr)

                    itr += 1


            model.eval()
            for headid in range(len(HEADS)):
                for ind, (ids, frames) in enumerate(valid_loaders[headid]):
                    input_list = []
                    for input_all_list in frames[:-(frame_num // 2)]:
                        input_list.append(input_all_list[1].unsqueeze(1))
                    input = torch.cat(input_list, 1).to(device)

                    target_list = []
                    for target_tensor in frames[-(frame_num // 2):]:
                        target_list.append(target_tensor.unsqueeze(1))
                    target = torch.cat(target_list, 1).to(device)

                    with torch.no_grad():
                        valid_output = model(input, target, None, is_training=False)
                        valid_loss = nn.MSELoss(valid_output, target)
                        total_valid_mse_loss.append(valid_loss.item())

            end_time = time.time()
            print('Epoch: ', epoch + 1 + start_epoch, ' | time: ', end_time - start_time, ' | train loss: ',
                  sum(total_train_loss) / len(total_train_loss), ' | valid loss: ', sum(total_valid_mse_loss) / len(total_valid_mse_loss))

            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + start_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'itr': itr,
                    'eta': eta
                }, 'sa_lstm_checkpoint_' + str(epoch + 1 + start_epoch) + '.pt')

                gen_img_reshape = train_output.detach().cpu().reshape(-1, 1, 128, 128)  
                save_image(gen_img_reshape.float().data[:48], f"sa_lstm_epoch{epoch + 1 + start_epoch}.png", nrow=6,
                           normalize=True)

        return model

    else:
        return model





def train_convlstm(params, train_loaders, valid_loaders, device, checkpoint = None, writer = None, is_training = False):
    frame_num = params['frame_num']
    batch_size = params['batch_size']
    HEADS = ['A', 'B', 'C', 'E']
    model = Encode2Decode(params).to(device)


    if checkpoint is not None:
        model_checkpoint = checkpoint[0]
        model.load_state_dict(model_checkpoint['model_state_dict'])
        eta = model_checkpoint['eta']
        start_epoch = model_checkpoint['epoch'] + 1
        itr = model_checkpoint['itr']   

        optimizer = model_checkpoint['optimizer_state_dict']


    else:
        eta = params['sampling_start_value']
        start_epoch = 0
        itr = 0
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if is_training:
        train_epoch = params['train_epoch']
        for epoch in range(train_epoch):
            model.train()
            torch.cuda.empty_cache()
            start_time = time.time()

            total_train_loss = []
            total_valid_mse_loss = []

            for headid in range(len(HEADS)):
                for ind, (ids, frames) in enumerate(train_loaders[headid]):

                    input_list = []
                    for input_tensor in frames[:-(frame_num // 2)]:
                        input_list.append(input_tensor.unsqueeze(1))
                    input = torch.cat(input_list, 1).to(device)

                    if input.size(0) != batch_size:
                        continue

                    target_list = []
                    for target_tensor in frames[-(frame_num // 2):]:
                        target_list.append(target_tensor.unsqueeze(1))
                    target = torch.cat(target_list, 1).to(device)

                    optimizer.zero_grad()
                    train_output = model(input)

                    loss, L_rec, loss_ssim = sa_lstm_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    total_train_loss.append(loss.item())

                    if writer is not None:
                        writer.add_scalar('Loss/sa_convlstm_loss', loss.item(), itr)
                        writer.add_scalar('Loss/sa_convlstm_L_rec', L_rec.item(), itr)
                        writer.add_scalar('Loss/sa_convlstm_L_ssim', loss_ssim.item(), itr)

                    itr += 1


            model.eval()

            for headid in range(len(HEADS)):
                for ind, (ids, frames) in enumerate(valid_loaders[headid]):
                    input_list = []
                    for input_all_list in frames[:-(frame_num // 2)]:
                        input_list.append(input_all_list[1].unsqueeze(1))
                    input = torch.cat(input_list, 1).to(device)

                    target_list = []
                    for target_tensor in frames[-(frame_num // 2):]:
                        target_list.append(target_tensor.unsqueeze(1))
                    target = torch.cat(target_list, 1).to(device)

                    with torch.no_grad():
                        valid_output = model(input)
                        valid_loss = nn.MSELoss(valid_output, target)
                        total_valid_mse_loss.append(valid_loss.item())

            end_time = time.time()

            print('Epoch: ', epoch + 1 + start_epoch, ' | time: ', end_time - start_time, ' | train loss: ',
                  sum(total_train_loss) / len(total_train_loss), ' | valid loss: ', sum(total_valid_mse_loss) / len(total_valid_mse_loss))

            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + start_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'itr': itr,
                    'eta': eta
                }, 'sa_convlstm_checkpoint_' + str(epoch + 1 + start_epoch) + '.pt')

                gen_img_reshape = train_output.detach().cpu().reshape(-1, 1, 128, 128)
                save_image(gen_img_reshape.float().data[:48], f"sa_convlstm_epoch{epoch + 1 + start_epoch}.png", nrow=6,
                           normalize=True)

        return model

    else:
        return model










