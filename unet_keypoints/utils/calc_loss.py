import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss, TripletMarginWithDistanceLoss, PairwiseDistance
import numpy as np
import time


# the nms function takes in a 2d array, a list of topk values and a list of topk indices
# return the array, topk values topk indices after suppress the local non-max (2 pixels nearby)
def nms(array2d, values, indices, limit=True):
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
    if limit:
        values_new, indices_new = torch.topk(array2d.flatten(), 500)
    else:
        values_new, indices_new = torch.topk(array2d.flatten(), torch.count_nonzero(array2d))
    indices_new = np.array(np.unravel_index(indices_new.detach().to("cpu").numpy(), array2d.shape)).T

    return array2d, values_new, indices_new

def new_loss(preds, images, labels, device):
    # set hyperpatameters
    BCEL = BCEWithLogitsLoss(reduction='mean')
    TRP = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(2), margin=10)
    detect_threah = 0.8
    detect_loss = torch.tensor(0, dtype=torch.float32, device=device)
    triplet_loss = torch.tensor(0, dtype=torch.float32, device=device)
    quantity = torch.tensor(0, dtype=torch.float32, device=device)
    num_ig = 0
    num_ir = 0

    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    for i, label in enumerate(labels):
        real_pred = torch.sigmoid(detectors[i * 2, :, :])
        gene_pred = torch.sigmoid(detectors[i * 2 + 1, :, :])

        real_des_pred = torch.sigmoid(descriptors[i * 2, :, :])
        gene_des_pred = torch.sigmoid(descriptors[i * 2 + 1, :, :])
        label = label[label.sum(axis=1) != 0].detach().cpu().numpy().astype(int)

        # Calculate loss by positive training labels
        matches_a, matches_p, matches_n, pos_detect = [], [], [], []
        for num in range(label.shape[0]):
            l = label[num]
            real_local_max = torch.max(real_pred[max(0, l[0] - 2): min(l[0] + 3, 479), max(0, l[1] - 2): min(l[1] + 3, 639)])
            gene_local_max = torch.max(gene_pred[max(0, l[2] - 2): min(l[2] + 3, 479), max(0, l[3] - 2): min(l[3] + 3, 639)])
            pos_detect.append(real_local_max)
            pos_detect.append(gene_local_max)

            matches_p.append(real_des_pred[:, l[0], l[1]])
            matches_a.append(gene_des_pred[:, l[2], l[3]])

        for x, ma in enumerate(matches_a):
            min_distance = 16
            min_point_n = []
            for y, mp in enumerate(matches_p):
                if x != y:
                    distance = torch.dist(mp, ma, 2)
                    if distance < min_distance:
                        min_distance = distance
                        min_point_n = torch.clone(ma)
            if min_distance == 16:
                matches_n.append(matches_p[0])
            else:
                matches_n.append(min_point_n)

        # Calculate loss by negative prediction results
        masked = (gene_pred * (images[i * 2 + 1, 3, :, :] == 0)).to(device=device)
        detect_loss += BCEL(masked, torch.zeros_like(masked))

        # Set all detection result smaller than threshold to 0
        real_masked = (real_pred > detect_threah) * real_pred
        # extra step to remove points in empty area
        gene_masked = (gene_pred * (images[i * 2 + 1, 3, :, :] != 0)).to(device=device)
        gene_masked = (gene_masked > detect_threah) * gene_masked

        # find all non-zero predictions, rank from highest to lowest and unravel for their index
        v_r, i_r = torch.topk(real_masked.flatten(), torch.count_nonzero(real_masked))
        i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
        v_g, i_g = torch.topk(gene_masked.flatten(), torch.count_nonzero(gene_masked))
        i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

        # Non-max suppression
        real_masked, v_r, i_r = nms(real_masked, v_r, i_r, limit=False)
        gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g, limit=False)


        for point_g in i_g:
            des_g = gene_des_pred[:, point_g[0], point_g[1]]
            min_distance = 16
            min_point_p = torch.zeros(256, dtype=torch.float32)
            min_point_n = []
            for point_r in i_r:
                des_r = real_des_pred[:, point_r[0], point_r[1]]
                distance = torch.dist(des_r, des_g, 2)
                # max distance 16 (square root), 1.6 is 10% distance
                if distance <= 1.6 and distance < min_distance:
                    min_distance = distance
                    min_point_n = min_point_p
                    min_point_p = point_r.copy()
            if min_distance == 16:
                pass
            else:
                if point_g[0] - 2 < point_r[0] < point_g[0] + 3 and point_g[1] - 2 < point_r[1] < point_g[1] + 3:
                    matches_a.append(gene_des_pred[:, point_g[0], point_g[1]])
                    matches_p.append(real_des_pred[:, min_point_p[0], min_point_p[1]])
                    matches_n.append(real_des_pred[:, min_point_n[0], min_point_n[1]])
                    pos_detect.append(real_pred[min_point_p[0], min_point_p[1]])
                    pos_detect.append(gene_pred[point_g[0], point_g[1]])

        if matches_a:
            anchor = torch.stack(matches_a).to(device=device)
            positive = torch.stack(matches_p).to(device=device)
            negative = torch.stack(matches_n).to(device=device)
            pos_detect = torch.stack(pos_detect)
            detect_loss += BCEL(pos_detect, torch.ones_like(pos_detect))
            triplet_loss += TRP(anchor, positive, negative)
            quantity += len(matches_a)
            num_ig += len(i_g)
            num_ir += len(i_r)

            if len(matches_n)==1:
                print('only one label available')
        else:
            detect_loss += 2
            triplet_loss += 10
            quantity += len(matches_a)
            num_ig += len(i_g)
            num_ir += len(i_r)


    total_loss = detect_loss + triplet_loss

    return total_loss, detect_loss, triplet_loss, [quantity, num_ig, num_ir]

def keypoint_loss(preds, images, labels, device):
    # set hyperpatameters
    BCEL = BCEWithLogitsLoss(reduction='mean')
    MSE = MSELoss(reduction='mean')
    TRP = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(2), margin=10)
    detect_threah = 0.8
    detect_loss = torch.tensor(0, dtype=torch.float32, device=device)
    triplet_loss = torch.tensor(0, dtype=torch.float32, device=device)
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

        if max_values:
            max_values = torch.stack(max_values)
            detect_loss += BCEL(max_values, torch.ones_like(max_values))

        # Calculate loss by negative prediction results
        # masked = (torch.sigmoid(gene_pred) > detect_threah) * gene_pred
        masked = torch.sigmoid(gene_pred)
        masked = (masked * (images[i * 2 + 1, 3, :, :] == 0)).to(device=device)
        detect_loss += BCEL(masked, torch.zeros_like(masked))

        # Calculate loss by matching descriptors
        real_des_pred = descriptors[i * 2, :, :]
        gene_des_pred = descriptors[i * 2 + 1, :, :]
        # Set all detection result smaller than threshold to 0
        real_masked = (torch.sigmoid(real_pred) > detect_threah) * real_pred
        # extra step to remove points in empty area
        gene_masked = (gene_pred * (images[i * 2 + 1, 3, :, :] != 0)).to(device=device)
        gene_masked = (torch.sigmoid(gene_masked) > detect_threah) * gene_masked


        # find all non-zero predictions, rank from highest to lowest and unravel for their index
        v_r, i_r = torch.topk(real_masked.flatten(), torch.count_nonzero(real_masked))
        i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
        v_g, i_g = torch.topk(gene_masked.flatten(), torch.count_nonzero(gene_masked))
        i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

        # Non-max suppression
        real_masked, v_r, i_r = nms(real_masked, v_r, i_r, limit=False)
        gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g, limit=False)

        matches_r, matches_g = [], []
        # randomly select 500 points from i_g
        if i_g.shape[0] > 500:
            perm = torch.randperm(i_g.shape[0])
            idx = perm[:500]
            i_g = i_g[idx]

        for point_r in i_r:
            for point_g in i_g:
                if point_g[0]-2 < point_r[0] < point_g[0]+3 and point_g[1]-2 < point_r[1] < point_g[1]+3:
                    matches_r.append(real_des_pred[:, point_r[0], point_r[1]])
                    matches_g.append(gene_des_pred[:, point_g[0], point_g[1]])

        if matches_r:
            negative = matches_r.copy()
            first = matches_r[0]
            negative.pop(0)
            negative.append(first)

            matches_r = torch.stack(matches_r).to(device=device)
            matches_g = torch.stack(matches_g).to(device=device)
            negative = torch.stack(negative).to(device=device)

            triplet_loss = TRP(matches_g, matches_r, negative)

            quantity_loss += 1 / len(matches_r)
        else:
            # apply the empty loss when no keypoint pairs can be found
            quantity_loss += 2
        # t4 = time.time()
        # print(f'found {len(v_r)} real points, {len(v_g)} gene points, {len(matches_r)} matches\n')
        # print(f'the nms took {t3 - t2} seconds\n')
        # print(f'the matching took {t4 - t3} seconds\n')
    detect_loss = detect_loss * 0.1
    triplet_loss = triplet_loss * 1
    quantity_loss = quantity_loss * 1
    total_loss = detect_loss+triplet_loss

    return total_loss, detect_loss, triplet_loss, quantity_loss

def keypoint_score(preds, images, labels, device):
    MSE = MSELoss(reduction='mean')
    detect_threah = 0.8
    score = torch.tensor(0, dtype=torch.float32, device=device)

    detectors = preds[:, 0, :, :]
    descriptors = preds[:, 1:, :, :]

    for i, label in enumerate(labels):
        real_pred = torch.sigmoid(detectors[i * 2, :, :])
        gene_pred = torch.sigmoid(detectors[i * 2 + 1, :, :])

        # Calculate loss by matching descriptors
        real_des_pred = torch.sigmoid(descriptors[i * 2, :, :])
        gene_des_pred = torch.sigmoid(descriptors[i * 2 + 1, :, :])
        # Set all detection result smaller than threshold to 0
        real_masked = (real_pred > detect_threah) * real_pred
        # extra step to remove points in empty area
        gene_masked = (gene_pred * (images[i * 2 + 1, 3, :, :] != 0)).to(device=device)
        gene_masked = (gene_masked > detect_threah) * gene_masked

        # find all non-zero predictions, rank from highest to lowest and unravel for their index
        v_r, i_r = torch.topk(real_masked.flatten(), torch.count_nonzero(real_masked)) #torch.count_nonzero(real_masked)
        i_r = np.array(np.unravel_index(i_r.detach().to("cpu").numpy(), real_masked.shape)).T
        v_g, i_g = torch.topk(gene_masked.flatten(), torch.count_nonzero(gene_masked))
        i_g = np.array(np.unravel_index(i_g.detach().to("cpu").numpy(), gene_masked.shape)).T

        # Non-max suppression
        real_masked, v_r, i_r = nms(real_masked, v_r, i_r, limit=False)
        gene_masked, v_g, i_g = nms(gene_masked, v_g, i_g, limit=False)

        matches_r, matches_g, = [], []
        for point_g in i_g:
            des_g = gene_des_pred[:, point_g[0], point_g[1]]
            min_distance = 16
            min_point_r = []
            for point_r in i_r:
                des_r = real_des_pred[:, point_r[0], point_r[1]]
                distance = torch.dist(des_r, des_g, 2)
                # max distance 16 (square root), 1.6 is 10% distance
                if distance <= 1.6 and distance < min_distance:
                    min_distance = distance
                    min_point_r = point_r.copy()
            if min_distance == 100:
                pass
            else:
                if point_g[0] - 2 < point_r[0] < point_g[0] + 3 and point_g[1] - 2 < point_r[1] < point_g[1] + 3:
                    matches_g.append(gene_des_pred[:, point_g[0], point_g[1]])
                    matches_r.append(real_des_pred[:, min_point_r[0], min_point_r[1]])

        if matches_r:
            matches_r = torch.stack(matches_r).to(device=device)
            matches_g = torch.stack(matches_g).to(device=device)
            loss = MSE(matches_r, matches_g)
            score += (1/loss)*matches_r.shape[0]
        else:
            pass

    return score