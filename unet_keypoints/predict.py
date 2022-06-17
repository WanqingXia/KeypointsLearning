import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.utils import plot_img_and_mask
import cv2
from utils.data_loading import create_dataloader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils.calc_loss import nms
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M
dt_string = now.strftime("%Y-%m-%d_%H:%M")

def process_pred(preds, threshold):
    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    real_pred = detectors[0, :, :]
    gene_pred = detectors[1, :, :]
    real_des_pred = torch.sigmoid(descriptors[0, :, :])
    gene_des_pred = torch.sigmoid(descriptors[1, :, :])
    # Set all detection result smaller than threshold to 0
    real_masked = (torch.sigmoid(real_pred) > threshold) * real_pred
    gene_masked = (torch.sigmoid(gene_pred) > threshold) * gene_pred

    # find all non-zero predictions, rank from highest to lowest and unravel for their index
    v_r, i_r = torch.topk(real_masked.flatten(), 500)
    i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
    v_g, i_g = torch.topk(gene_masked.flatten(), 500)
    i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

    # Non-max suppression
    real_masked, v_r, i_r = nms(real_masked, v_r, i_r)
    gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g)
    matches_r, matches_g = [], []

    for point_r in i_r:
        des_r = real_des_pred[:, point_r[0], point_r[1]]
        min_distance = 100
        min_point_g = []
        for point_g in i_g:
            des_g = gene_des_pred[:, point_g[0], point_g[1]]
            distance = torch.dist(des_r, des_g)
            # max distance 16 (square root), 1.6 is 10% distance
            if distance <= 1.6 and distance < min_distance:
                min_distance = distance
                min_point_g = point_g
        if min_distance == 100:
            pass
        else:
            matches_r.append(point_r)
            matches_g.append(min_point_g)
    return matches_r, matches_g


def predict_img(dir_ycb, test_folder, output, threshold, net, device, save):
    net.eval()
    test_loader = create_dataloader(dir_ycb, 'test', test_folder, num_workers=8, pin_memory=True)
    # for num, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    for num, data in enumerate(test_loader):
        images = []
        images.append((data[0])[0].get('real'))
        images.append((data[0])[0].get('gene'))
        images_T = torch.stack(images)
        images_T = images_T.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            preds = net(images_T)

        matches_r, matches_g = process_pred(preds, threshold)
        if save:
            dir_r = data[2][0]
            dir_g = data[3][0]
            out_dir = output / dir_r.name.replace('color', 'result')
            image_r = cv2.imread(str(dir_r))
            image_g = cv2.imread(str(dir_g))
            out_img = np.concatenate((image_r, image_g), axis=1)
            for point_r, point_g in zip(matches_r[:50], matches_g[:50]):
                out_img = cv2.line(out_img, (point_r[1], point_r[0]), (point_g[1] + image_r.shape[1], point_g[0]),
                                   (0, 255, 0), 1)
            cv2.imwrite(str(out_dir), out_img)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/2022-06-15_13:58/checkpoint_epoch1.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--save', '-s', action='store_true', default=True, help='Save the output images')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--descriptors', '-c', type=int, default=256, help='Channels of descriptors')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()

    # test_folders = ['0006', '0009', '0010', '0011', '0012', '0018', '0024', '0030', '0037', '0038', '0050',
    # '0054', '0056', '0059', '0077', '0081', '0083', '0086', '0088']
    test_folders = ['0006']
    ycb_dir = Path('/data/Wanqing/YCB_Video_Dataset/')
    out_dir = Path('./output')/dt_string

    net = UNet(n_channels=4, n_descriptors=args.descriptors, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(test_folders):
        logging.info(f'\n Processing folder {ycb_dir}/{filename} ...')
        out = out_dir/filename
        out.mkdir(parents=True, exist_ok=True)
        file = []
        file.append(filename)
        predict_img(ycb_dir, file, out, args.threshold, net=net, device=device, save=args.save)
