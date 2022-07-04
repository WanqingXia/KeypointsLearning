import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from time import time
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
    tic = time()

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
    v_r, i_r = torch.topk(real_masked.flatten(), torch.count_nonzero(real_masked))
    i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
    v_g, i_g = torch.topk(gene_masked.flatten(), torch.count_nonzero(real_masked))
    i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

    # Non-max suppression
    real_masked, v_r, i_r = nms(real_masked, v_r, i_r, limit=False)
    gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g, limit=False)
    matches_r, matches_g = [], []

    for point_r in i_r:
        des_r = real_des_pred[:, point_r[0], point_r[1]]
        min_distance = 100
        min_point_g = []
        for point_g in i_g:
            des_g = gene_des_pred[:, point_g[0], point_g[1]]
            # tic = time()
            distance = torch.dist(des_r, des_g)
            # toc = time()
            # print(f'the process took {toc - tic} seconds')
            # max distance 16 (square root), 1.6 is 10% distance
            if distance <= 1.6 and distance < min_distance:
                min_distance = distance
                min_point_g = point_g
        if min_distance == 100:
            pass
        else:
            matches_r.append(point_r)
            matches_g.append(min_point_g)

    filter_r, filter_g = [], []
    for point_r, point_g in zip(matches_r, matches_g):
        if point_g[0] - 2 < point_r[0] < point_g[0] + 3 and point_g[1] - 2 < point_r[1] < point_g[1] + 3:
            filter_r.append(point_r)
            filter_g.append(point_g)
    toc = time()
    print(f'the process took {toc - tic} seconds')

    return matches_r, matches_g, filter_r, filter_g


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

        matches_r, matches_g, filter_r, filter_g = process_pred(preds, threshold)
        if save:
            dir_r = data[2][0]
            dir_g = data[3][0]
            out_dir1 = output / dir_r.name.replace('color', 'resulto')
            out_dir2 = output / dir_r.name.replace('color', 'resultf')
            image_r = cv2.imread(str(dir_r))
            image_g = cv2.imread(str(dir_g))
            out_img1 = np.concatenate((image_r, image_g), axis=1)
            out_img2 = np.concatenate((image_r, image_g), axis=1)
            for point_r, point_g in zip(matches_r, matches_g):
                out_img1 = cv2.line(out_img1, (point_r[1], point_r[0]), (point_g[1] + image_r.shape[1], point_g[0]),
                                   (0, 255, 0), 1)
            for point_r, point_g in zip(filter_r, filter_g):
                out_img2 = cv2.line(out_img2, (point_r[1], point_r[0]), (point_g[1] + image_r.shape[1], point_g[0]),
                                   (0, 255, 0), 1)

            cv2.imwrite(str(out_dir1), out_img1)
            cv2.imwrite(str(out_dir2), out_img2)
            print('before filter real' + str(len(matches_r)))
            print('before filter gene' + str(len(matches_g)))
            print('after filter real' + str(len(filter_r)))
            print('after filter gene' + str(len(filter_g)))
            print('true positive rate ' + str(len(filter_r)/len(matches_r)))
            print('false positive rate ' + str((len(matches_r)-len(filter_r))/len(matches_r)))


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/sunny-armadillo-232/checkpoint_epoch30.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--save', '-s', action='store_true', default=True, help='Save the output images')
    parser.add_argument('--threshold', '-t', type=float, default=0.9,
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
    test_folders = ['0000']
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
