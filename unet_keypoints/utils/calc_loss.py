import torch
from torch import Tensor
from torch.nn import BCELoss


def keypoint_loss(preds, images, labels):
    BCE = BCELoss(reduction='mean')
    detect_threah = 0.7
    detect_weight = 0.1
    descrip_weight = 0.05
    detect_loss = 0
    descrip_loss = 0

    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    for i, label in enumerate(labels):
        real_pred = detectors[i * 2, :, :, :]
        gene_pred = detectors[i * 2 + 1, :, :, :]
        label = label[label.sum(dim=0) != 0]

        loss_sum = 0
        for num in range(label.shape[0]):
            l = label[num]
            real_local_max = max(real_pred[max(0, l[0] - 2): min(l[0] + 2, 479), max(0, l[1] - 2): min(l[1] + 2, 639)])
            gene_local_max = max(gene_pred[max(0, l[2] - 2): min(l[2] + 2, 479), max(0, l[3] - 2): min(l[3] + 2, 639)])
            loss_sum += torch.log(real_local_max) + torch.log(gene_local_max)
        detect_loss += -loss_sum/(label.shape[0]*2)

        masked = gene_pred[gene_pred > detect_threah]
        masked = masked * (images[i * 2 + 1, 3, :, :] == 0)
        detect_loss += BCE(masked, torch.zeros((1, 1, 480, 640)))

    return detect_weight*detect_loss + descrip_weight*descrip_loss
