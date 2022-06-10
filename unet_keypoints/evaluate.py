import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.calc_loss import keypoint_loss

#TODO: implement the evaluation function, gain points when matches are close and in object area

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        images = []
        for item in batch[0]:
            images.append(item.get('real'))
            images.append(item.get('gene'))
        images_T = torch.stack(images)
        labels = batch[1]

        images_T = images_T.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            points_pred = net(images_T)
            score += keypoint_loss(points_pred, images_T, labels)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score
    return score / num_val_batches
