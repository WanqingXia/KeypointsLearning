import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

from utils.data_loading import create_dataloader
from utils.calc_loss import keypoint_loss
from evaluate import evaluate
from unet import UNet


from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M
dt_string = now.strftime("%Y-%m-%d_%H:%M")

dir_ycb = Path('/data/Wanqing/YCB_Video_Dataset/')
dir_checkpoint = Path('./checkpoints/' + dt_string)
# list contains all dataset folders will be used as testing dataset, selected since they all
# have "power_drill", see list.py
test_folders = ['0006', '0009', '0010', '0011', '0012', '0018', '0024', '0030', '0037', '0038', '0050', '0054', '0056',
                '0059', '0077', '0081', '0083', '0086', '0088']

l1loss = nn.L1Loss()
def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataloaders
    train_loader, val_loader, train_set, val_set = create_dataloader(dir_ycb, 'train', test_folders, val_percent,
                                batch_size=batch_size, sample_num=5400, num_workers=8, pin_memory=True)
    # test_loader, test_set = YCBDataset(dir_ycb, 'test', test_folders, batch_size=batch_size,
    #                        num_workers=8, pin_memory=True)  # test dataset

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 2. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # 3. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=1000, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for num, batch in enumerate(train_loader):
                images = []
                for item in batch[0]:
                    images.append(item.get('real'))
                    images.append(item.get('gene'))
                images_T = torch.stack(images)
                labels = batch[1]

                assert images_T.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images_T.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images_T = images_T.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    points_pred = net(images_T)
                    loss = keypoint_loss(points_pred, images_T, labels)
                    input = torch.randn(3, 5, requires_grad=True).to(device=device, dtype=torch.float32)
                    target = torch.randn(3, 5).to(device=device, dtype=torch.float32)
                    loss = l1loss(input, target)

                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round, evaluate every 100 batches
                division_step = 1000
                if division_step > 0 and global_step % division_step == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(net, val_loader, device)
                    scheduler.step(val_score)

                    logging.info('Validation Score: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Score': val_score,
                        # 'images': wandb.Image(images[0].cpu()),
                        # 'masks': {
                        #     'true': wandb.Image(true_masks[0].float().cpu()),
                        #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        # },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--descriptors', '-d', type=int, default=256, help='Channels of descriptors')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=4 for RGBD images
    # n_descriptors is the number of descriptor depth you want to get per pixel
    net = UNet(n_channels=4, n_descriptors=args.descriptors, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_descriptors+1} output channels (detector + descriptors)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    summary(net, (4, 480, 640))
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
