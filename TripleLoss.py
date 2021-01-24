import torch
#source : https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py inspired by https://omoindrot.github.io/triplet-loss

def _pairwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 -mask) * torch.sqrt(distances)
    return distances

def _get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels, device):
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, device='cuda'):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss
