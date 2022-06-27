import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np
import time


# the nms function takes in a 2d array, a list of topk values and a list of topk indices
# return the array, topk values topk indices after suppress the local non-max (2 pixels nearby)
def nms(array2d, values, indices):
    for value, i in zip(values, indices):
        # this value is suppressed already, pass
        if array2d[i[0], i[1]] == 0:
            pass
        # this location is the highest score in neighbours, suppress neighbours
        else:
            # set all local values to 0
            array2d[max(0, i[0] - 2): min(i[0] + 3, 479), max(0, i[1] - 2): min(i[1] + 3, 639)] = 0
            # set the middle value back
            array2d[i[0], i[1]] = value
    values_new, indices_new = torch.topk(array2d.flatten(), 500)
    indices_new = np.array(np.unravel_index(indices_new.detach().to("cpu").numpy(), array2d.shape)).T

    return array2d, values_new, indices_new

def keypoint_loss(preds, images, labels, device):
    # set hyperpatameters
    BCEL = BCEWithLogitsLoss(reduction='mean')
    MSE = MSELoss(reduction='mean')
    detect_threah = 0.7
    detect_weight = 0.1 # weight of the detect loss
    descrip_weight = 0.05 # weight of the description loss loss
    detect_loss = torch.tensor(0, dtype=torch.float32, device=device)
    description_loss = torch.tensor(0, dtype=torch.float32, device=device)
    distance_loss = torch.tensor(0, dtype=torch.float32, device=device)
    quantity_loss = torch.tensor(0, dtype=torch.float32, device=device)

    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    for i, label in enumerate(labels):
        real_pred = detectors[i * 2, :, :]
        gene_pred = detectors[i * 2 + 1, :, :]
        label = label[label.sum(axis=1) != 0].detach().cpu().numpy().astype(int)

        # Calculate loss by positive training labels
        max_values = []
        for num in range(label.shape[0]):
            l = label[num]
            real_local_max = torch.max(real_pred[max(0, l[0] - 2): min(l[0] + 3, 479), max(0, l[1] - 2): min(l[1] + 3, 639)])
            gene_local_max = torch.max(gene_pred[max(0, l[2] - 2): min(l[2] + 3, 479), max(0, l[3] - 2): min(l[3] + 3, 639)])
            max_values.append(real_local_max)
            max_values.append(gene_local_max)

        # need to divide by shape again since previous loss is calculated individually
        if max_values:
            max_values = torch.stack(max_values)
            detect_loss += BCEL(max_values, torch.ones_like(max_values))

        # Calculate loss by negative prediction results
        # masked = (torch.sigmoid(gene_pred) > detect_threah) * gene_pred
        # masked = (masked * (images[i * 2 + 1, 3, :, :] == 0)).to(device=device)
        # detect_loss += BCEL(masked, torch.zeros_like(masked))

        # Calculate loss by matching descriptors
        real_des_pred = descriptors[i * 2, :, :]
        gene_des_pred = descriptors[i * 2 + 1, :, :]
        # Set all detection result smaller than threshold to 0
        real_masked = (torch.sigmoid(real_pred) > detect_threah) * real_pred
        # extra step to remove points in empty area
        gene_masked = (gene_pred * (images[i * 2 + 1, 3, :, :] != 0)).to(device=device)
        gene_masked = (torch.sigmoid(gene_masked) > detect_threah) * gene_masked


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
            for point_g in i_g:
                if point_g[0]-2 < point_r[0] < point_g[0]+3 and point_g[1]-2 < point_r[1] < point_g[1]+3:
                    matches_r.append(real_des_pred[:, point_r[0], point_r[1]])
                    matches_g.append(gene_des_pred[:, point_g[0], point_g[1]])

        if matches_r:
            matches_r = torch.stack(matches_r).to(device=device)
            matches_g = torch.stack(matches_g).to(device=device)
            description_loss += MSE(torch.sigmoid(matches_r), torch.sigmoid(matches_g))
            distance_loss += 1 / MSE(torch.sigmoid(matches_r), torch.sigmoid(torch.flip(matches_r, dims=(0,))))
            distance_loss += 1 / MSE(torch.sigmoid(matches_g), torch.sigmoid(torch.flip(matches_g, dims=(0,))))
            quantity_loss += 1 / len(matches_r)
        else:
            # apply the empty loss when no keypoint pairs can be found
            quantity_loss += 0.25
        # t4 = time.time()
        # print(f'found {len(v_r)} real points, {len(v_g)} gene points, {len(matches_r)} matches\n')
        # print(f'the nms took {t3 - t2} seconds\n')
        # print(f'the matching took {t4 - t3} seconds\n')
    detect_loss = detect_loss * 1
    description_loss = description_loss * 1
    distance_loss = distance_loss * 0.001
    quantity_loss = quantity_loss * 1

    return detect_loss+description_loss+distance_loss+quantity_loss, detect_loss, description_loss, distance_loss, quantity_loss

def keypoint_score(preds, images, labels, device):
    MSE = MSELoss(reduction='mean')
    detect_threah = 0.7
    score = torch.tensor(0, dtype=torch.float32, device=device)

    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    for i, label in enumerate(labels):
        real_pred = detectors[i * 2, :, :]
        gene_pred = detectors[i * 2 + 1, :, :]

        # Calculate loss by matching descriptors
        real_des_pred = descriptors[i * 2, :, :]
        gene_des_pred = descriptors[i * 2 + 1, :, :]
        # Set all detection result smaller than threshold to 0
        real_masked = (torch.sigmoid(real_pred) > detect_threah) * real_pred
        # extra step to remove points in empty area
        gene_masked = (gene_pred * (images[i * 2 + 1, 3, :, :] != 0)).to(device=device)
        gene_masked = (torch.sigmoid(gene_masked) > detect_threah) * gene_masked

        # find all non-zero predictions, rank from highest to lowest and unravel for their index
        v_r, i_r = torch.topk(real_masked.flatten(), 500) #torch.count_nonzero(real_masked)
        i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
        v_g, i_g = torch.topk(gene_masked.flatten(), 500)
        i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

        # Non-max suppression
        real_masked, v_r, i_r = nms(real_masked, v_r, i_r)
        gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g)

        matches_r, matches_g = [], []
        for point_r in i_r:
            for point_g in i_g:
                if point_g[0] - 2 < point_r[0] < point_g[0] + 3 and point_g[1] - 2 < point_r[1] < point_g[1] + 3:
                    matches_r.append(real_des_pred[:, point_r[0], point_r[1]])
                    matches_g.append(gene_des_pred[:, point_g[0], point_g[1]])

        if matches_r:
            matches_r = torch.stack(matches_r).to(device=device)
            matches_g = torch.stack(matches_g).to(device=device)
            loss = MSE(torch.sigmoid(matches_r), torch.sigmoid(matches_g))
            score += (1-loss)*matches_r.shape[0]
        else:
            pass

    return score