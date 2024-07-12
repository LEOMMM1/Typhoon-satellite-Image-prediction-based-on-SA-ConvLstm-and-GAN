import torch
from torch import nn

from PIL import Image
import numpy as np
import glob
import gc
import os
import re
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from convlstm_model import ConvLstmEncode2Decode
from sa_lstm_model import Encode2Decode, Discriminator, generator_loss_function, sa_lstm_loss
from metric_function import ssim_metric, mse_metric, psnr_metric, sharpness_metric


def model_test(model_name, model, test_loader, frame_num, batch_size, device, writer = None):
    model.eval()

    HEADS = ['U', 'V', 'W', 'X', 'Y']
    ssim_sum, ssim_first_sum = 0, 0
    mse_sum, mse_first_sum = 0, 0
    psnr_sum, psnr_first_sum = 0, 0
    sharpness_sum, sharpness_first_sum = 0, 0

    itr = 0
    for headid in range(len(HEADS)):
        for ind, (ids, frames) in enumerate(test_loader[headid]):
            input_list = []
            for input_all_list in frames[:-(frame_num // 2)]:
                input_list.append(input_all_list.unsqueeze(1))
            input = torch.cat(input_list, 1).to(device)

            if input.size(0) != batch_size:
                continue

            target_list = []
            for target_tensor in frames[-(frame_num // 2):]:
                target_list.append(target_tensor.unsqueeze(1))
            target = torch.cat(target_list, 1).to(device)

            with torch.no_grad():
                if model_name == 'convlstm':
                    output = model(input)
                else:
                    output = model(input, target, None, is_training = False)

            # calculate metrics
            ssim_value, ssim_first = ssim_metric(output, target)
            mse_value, mse_first = mse_metric(output, target)
            psnr_value, psnr_first = psnr_metric(output, target)
            sharpness_value, sharpness_first = sharpness_metric(output)

            ssim_sum += ssim_value; ssim_first_sum += ssim_first
            mse_sum += mse_value; mse_first_sum += mse_first
            psnr_sum += psnr_value; psnr_first_sum += psnr_first
            sharpness_sum += sharpness_value; sharpness_first_sum += sharpness_first

            if writer is not None:
                writer.add_scalar('ssim', ssim_value, itr)
                writer.add_scalar('mse', mse_value, itr)
                writer.add_scalar('psnr', psnr_value, itr)
                writer.add_scalar('sharpness', sharpness_value, itr)

            itr += 1


    print('ssim:', ssim_sum / 80, ssim_first_sum / 80)
    print('mse:', mse_sum / 80, mse_first_sum / 80)
    print('psnr:', psnr_sum / 80, psnr_first_sum / 80)
    print('sharpness:', sharpness_sum / 80, sharpness_first_sum / 80)



