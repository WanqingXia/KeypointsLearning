import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.calc_loss import keypoint_loss

#TODO: implement the evaluation function, gain points when matches are close and in object area

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            points_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                points_pred = (F.sigmoid(points_pred) > 0.5).float()
                # compute the Dice score
                dice_score += keypoint_loss(points_pred)
            else:
                points_pred = F.one_hot(points_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += keypoint_loss(points_pred)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
