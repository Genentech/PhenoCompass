"""Loss functions for multimodal contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(x1, x2):
    """Compute cosine distance between two tensors."""
    return (F.normalize(x1, p=1) * F.normalize(x2, p=1)).sum(1)


def get_loss(name="gmc"):
    """Get loss function by name.

    Parameters
    ----------
    name : str
        Name of the loss function. Options: 'gmc', 'mocop', 'mtl_bceloss',
        'triple_infonce', 'barlow_twins', 'gmc_sigmoid',
        'gmc_soft_nearest_neighbors_loss', 'gmc_soft_nearest_neighbors_out_loss',
        'gmc_supervised_metric_learning', 'gmc_ccl', 'gmc_consistency'.

    Returns
    -------
    nn.Module
        Loss function module.
    """
    if name == "gmc":
        return GMCLoss()
    elif name == "mocop":
        return MOCOPLoss()
    elif name == "mtl_bceloss":
        return MultiTaskLoss()
    elif name == "triple_infonce":
        return TripleInfoNCELoss()
    elif name == "barlow_twins":
        return BarlowTwinsLoss(None)
    elif name == "gmc_sigmoid":
        return GMCSigmoidLoss()
    elif name == "gmc_soft_nearest_neighbors_loss":
        return GMCSoftNearestNeighborsLoss()
    elif name == "gmc_soft_nearest_neighbors_out_loss":
        return GMCSoftNearestNeighborsOutLoss()
    elif name == "gmc_supervised_metric_learning":
        return GMCSupervisedMetricLearningLoss()
    elif name == "gmc_ccl":
        return ConditionalContrastiveLoss()
    elif name == "gmc_consistency":
        return GMCLossConsistency()
    else:
        raise ValueError(f"Unknown loss name: {name}")


class TripleInfoNCELoss(nn.Module):
    """Triple InfoNCE loss for three modalities."""

    def __init__(self, learnable_w=None):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, normalize=True):
        emb_x, emb_y, emb_z = list(batch_representations.values())
        loss = 0
        for emb_i, emb_j in [(emb_x, emb_y), (emb_x, emb_z), (emb_y, emb_z)]:
            CL_loss_1 = do_CL(emb_i, emb_j, normalize, temp, batch_size)
            CL_loss_2 = do_CL(emb_j, emb_i, normalize, temp, batch_size)
            loss += (CL_loss_1 + CL_loss_2) / 2

        return loss


class GMCLoss(nn.Module):
    """GMC Loss for grouped multimodal contrastive learning.

    Compares each modality to the joint representation.

    Parameters
    ----------
    batch_representations : dict
        Dictionary mapping modality names to batch representations.
    temp : float
        Temperature parameter for calculating the loss.
    batch_size : int
        Batch size.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, learnable_w=None):
        loss_list = gmc_loss_terms(batch_representations, temp, batch_size)
        if learnable_w is None:
            loss_total = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_total = torch.mean(
                torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0
            )
        return loss_total


class GMCSigmoidLoss(nn.Module):
    """GMC Sigmoid Loss for triple contrastive learning."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, learnable_w=None):
        loss_list = gmc_sigmoid_loss_terms(batch_representations, temp, batch_size)
        if learnable_w is None:
            loss_total = torch.mean(torch.stack(loss_list))
        else:
            raise NotImplementedError
        return loss_total


class GMCSoftNearestNeighborsLoss(nn.Module):
    """GMC Soft Nearest Neighbors Loss for triple contrastive learning."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, pairwise_similarity, temp, batch_size, learnable_w=None):
        loss_list = gmc_soft_nearest_neighbors_loss_terms(
            batch_representations, pairwise_similarity, temp, batch_size
        )
        if learnable_w is None:
            loss_total = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_total = torch.mean(
                torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0
            )
        return loss_total


class GMCSoftNearestNeighborsOutLoss(nn.Module):
    """GMC Soft Nearest Neighbors Loss (sum outside log)."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, pairwise_similarity, temp, batch_size, learnable_w=None):
        loss_list = gmc_soft_nearest_neighbors_out_loss_terms(
            batch_representations, pairwise_similarity, temp, batch_size
        )
        if learnable_w is None:
            loss_total = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_total = torch.mean(
                torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0
            )
        return loss_total


class GMCSupervisedMetricLearningLoss(nn.Module):
    """GMC supervised metric learning loss."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, pairwise_similarity, temp, batch_size, learnable_w=None):
        loss_list = gmc_supervised_metric_learning_loss_terms(
            batch_representations, pairwise_similarity, temp, batch_size
        )
        if learnable_w is None:
            loss_total = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_total = torch.mean(
                torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0
            )
        return loss_total


class ConditionalContrastiveLoss(nn.Module):
    """Conditional contrastive learning loss."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, batch_meta, temp, batch_size, learnable_w=None):
        loss_list = conditional_contrastive_gmc_loss_terms(
            batch_representations, batch_meta, temp, batch_size
        )
        if learnable_w is None:
            loss_total = torch.mean(torch.stack(loss_list))
        else:
            loss_total = torch.sum((torch.stack(loss_list) * learnable_w))
        return loss_total


class GMCLossConsistency(nn.Module):
    """GMCLoss for triple contrastive loss with consistency regularization.

    Parameters
    ----------
    batch_representations : dict
        Dictionary mapping modality names to batch representations.
    temp : float
        Temperature parameter for calculating the loss.
    batch_size : int
        Batch size.
    alpha : float
        Weight for the consistency loss term. Default is 10.
    """

    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def forward(self, batch_representations, temp, batch_size, learnable_w=None):
        loss_list, consistency_loss = gmc_loss_terms_with_consistency(batch_representations, temp, batch_size)
        if learnable_w is None:
            loss_contrastive = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_contrastive = torch.mean(torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0)
        loss_total = loss_contrastive + self.alpha * consistency_loss
        return loss_total


class MOCOPLoss(nn.Module):
    """MOCOP Loss for dual contrastive learning."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, normalize=True, learnable_w=None):
        emb_x, emb_y = list(batch_representations.values())
        CL_loss_1 = do_CL(emb_x, emb_y, normalize, temp, batch_size)
        CL_loss_2 = do_CL(emb_y, emb_x, normalize, temp, batch_size)
        loss = (CL_loss_1 + CL_loss_2) / 2
        return loss


class TripletLoss(nn.Module):
    """Triplet margin loss."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, learnable_w=None):
        triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_distance, margin=1.0
        )

        anchor = batch_representations["struct"]
        positive = batch_representations["morph"]
        negative = batch_representations["ctrl"]

        return triplet_loss(anchor, positive, negative)


class MultiTaskLoss(torch.nn.Module):
    """Multi-task binary cross-entropy loss with NaN handling."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, label):
        # Create a mask to identify where labels are not NaN
        mask = ~torch.isnan(label)

        # Select only the elements corresponding to non-NaN labels
        valid_output = torch.masked_select(output, mask)
        valid_label = torch.masked_select(label, mask)

        # Use BCELoss to compute the loss on the valid elements
        valid_label = valid_label.to(torch.float32)
        valid_output = valid_output.to(torch.float32)

        loss = F.binary_cross_entropy_with_logits(
            valid_output, valid_label, reduction=self.reduction
        )

        return loss


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins self-supervised loss.

    Parameters
    ----------
    lamda : float
        Weight on off-diagonal terms. Default is 0.0051.
    """

    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        if lamda is None:
            self.lamda = 0.0051
        print(f"Using loss: BarlowTwins with lamda: {self.lamda}")

    def forward(self, batch_representations, batch_size, *args, **kwargs):
        emb_joint = batch_representations["joint"]
        emb_mods = [v for k, v in batch_representations.items() if k != "joint"]

        loss = 0
        for emb_m in emb_mods:
            BarlowTwins_loss = self._do_BarlowTwins(emb_m, emb_joint, batch_size)
            loss += BarlowTwins_loss

        loss /= len(emb_mods)

        return loss

    def _do_BarlowTwins(self, x, y, batch_size):
        # empirical cross-correlation matrix
        c = x.T @ y
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        loss = on_diag + self.lamda * off_diag

        return loss


def do_CL(X, Y, normalize, temp, batch_size):
    """Compute symmetric contrastive loss."""
    if normalize:
        X = F.normalize(X)
        Y = F.normalize(Y)

    criterion = nn.CrossEntropyLoss()
    B = batch_size

    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, temp)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss


def gmc_loss_terms(batch_representations, temperature, batch_size, target_mod="joint"):
    """Compute GMC loss terms for each modality."""
    src_mod_list = [k for k in batch_representations.keys() if k != target_mod]

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        # Negative pairs: everything that is not in the current joint-modality pair
        out_joint_mod = torch.cat(
            [batch_representations[target_mod], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = torch.exp(
            torch.sum(
                batch_representations[target_mod] * batch_representations[mod], dim=-1
            )
            / temperature
        )
        # [2*B]
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))
        joint_mod_loss_terms.append(loss_joint_mod)

    return joint_mod_loss_terms


def gmc_loss_terms_with_consistency(batch_representations, temperature, batch_size, target_mod="joint"):
    """Compute GMC loss terms with consistency regularization.

    Parameters
    ----------
    batch_representations : dict
        Dictionary mapping modality names to batch representations.
    temperature : float
        Temperature parameter for calculating the loss.
    batch_size : int
        Batch size.
    target_mod : str
        Target modality name. Default is "joint".

    Returns
    -------
    tuple
        (joint_mod_loss_terms, consistency_loss) where joint_mod_loss_terms is a list
        of loss tensors for each modality pair and consistency_loss is the
        consistency regularization term.
    """
    src_mod_list = ['morph', 'struct']

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        # Negative pairs: everything that is not in the current joint-modality pair
        out_joint_mod = torch.cat(
            [batch_representations[target_mod], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
            mask_joint_mod
        ).view(2 * batch_size, -1)

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = torch.exp(
            torch.sum(
                batch_representations[target_mod] * batch_representations[mod], dim=-1
            )
            / temperature
        )
        # [2*B]
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(
            pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
        )
        joint_mod_loss_terms.append(loss_joint_mod)

    # Consistency regularization
    temperature_2 = 0.5
    dist_list = []
    for mod in src_mod_list:
        joint_embeddings = batch_representations[target_mod]
        unimodal_embeddings = batch_representations[mod]
        sim_matrix_joint_unimodal = torch.exp(
            torch.mm(joint_embeddings, unimodal_embeddings.t().contiguous()) / temperature_2
        )
        sim_matrix_unimodal = torch.exp(
            torch.mm(unimodal_embeddings, unimodal_embeddings.t().contiguous()) / temperature_2
        )
        mask = (
                torch.ones_like(sim_matrix_joint_unimodal)
                - torch.eye(batch_size, device=sim_matrix_joint_unimodal.device)
        ).bool()

        sim_matrix_joint_unimodal = sim_matrix_joint_unimodal.masked_select(
            mask
        ).view(batch_size, -1)

        sim_matrix_unimodal = sim_matrix_unimodal.masked_select(
            mask
        ).view(batch_size, -1)

        log_probs_joint_unimodal = torch.nn.functional.log_softmax(sim_matrix_joint_unimodal, dim=-1)
        log_probs_unimodal = torch.nn.functional.log_softmax(sim_matrix_unimodal, dim=-1)
        dist_list.append([log_probs_joint_unimodal, log_probs_unimodal])

    cross_m_p, with_m_p = dist_list[0]
    cross_m_q, with_m_q = dist_list[1]
    kl_div = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    consistency_loss_1 = 0.5 * (kl_div(with_m_p, cross_m_p) + kl_div(cross_m_p, with_m_p))
    consistency_loss_2 = 0.5 * (kl_div(with_m_q, cross_m_q) + kl_div(cross_m_q, with_m_q))
    consistency_loss = 0.5 * (consistency_loss_1 + consistency_loss_2)

    return joint_mod_loss_terms, consistency_loss


def gmc_sigmoid_loss_terms(batch_representations, temperature, batch_size, target_mod="joint"):
    """Compute GMC sigmoid loss terms."""
    src_mod_list = [k for k in batch_representations.keys() if k != target_mod]
    labels_tiled = (
        2 * torch.tile(
            torch.eye(batch_size, device=batch_representations[target_mod].device), (2, 2)
        )
        - 1
    )

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        out_joint_mod = torch.cat(
            [batch_representations[target_mod], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = (
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        loss_joint_mod = -torch.log(
            1.0 / (1 + torch.exp(-labels_tiled * sim_matrix_joint_mod))
        )

        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(loss_joint_mod)
            - torch.eye(2 * batch_size, device=loss_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        loss_joint_mod = loss_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        joint_mod_loss_terms.append(loss_joint_mod.mean())

    return joint_mod_loss_terms


def conditional_contrastive_gmc_loss_terms(
    batch_representations, batch_meta, temperature, batch_size, target_mod="joint"
):
    """Compute conditional contrastive GMC loss terms."""
    src_mod_list = [k for k in batch_representations.keys() if k != target_mod]

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        joint_mod_loss_terms_per_cond = []
        for meta_label in batch_meta.unique():
            idx = torch.where(batch_meta == meta_label)[0]
            src_representations = batch_representations[mod][idx]
            tgt_representations = batch_representations[target_mod][idx]

            batch_label_size = len(idx)

            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat([src_representations, tgt_representations], dim=0)
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_label_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
                2 * batch_label_size, -1
            )

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(tgt_representations * src_representations, dim=-1) / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_terms_per_cond.append(loss_joint_mod.mean())

        joint_mod_loss_terms_per_cond = torch.stack(joint_mod_loss_terms_per_cond)
        joint_mod_loss_terms.append(torch.sum(joint_mod_loss_terms_per_cond))

    return joint_mod_loss_terms


def gmc_soft_nearest_neighbors_loss_terms(
    batch_representations, pairwise_similarity, temperature, batch_size, target_mod="joint"
):
    """Compute GMC soft nearest neighbors loss terms."""
    src_mod_list = [k for k in batch_representations.keys() if k != target_mod]
    pairwise_similarity.fill_diagonal_(1)
    pairwise_similarity_tiled = torch.tile(pairwise_similarity, (2, 2))

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        out_joint_mod = torch.cat(
            [batch_representations[target_mod], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        pairwise_similarity_tiled_masked = pairwise_similarity_tiled.masked_select(
            mask_joint_mod
        ).view(2 * batch_size, -1)

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = (sim_matrix_joint_mod * pairwise_similarity_tiled_masked).mean(
            1
        )
        # [2*B]
        loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))
        joint_mod_loss_terms.append(loss_joint_mod)

    return joint_mod_loss_terms


def gmc_soft_nearest_neighbors_out_loss_terms(
    batch_representations, pairwise_similarity, temperature, batch_size, target_mod="joint"
):
    """Compute GMC soft nearest neighbors loss terms (sum outside log)."""
    src_mod_list = [k for k in batch_representations.keys() if k != target_mod]
    pairwise_similarity.fill_diagonal_(1)
    pairwise_similarity_tiled = torch.tile(pairwise_similarity, (2, 2))

    joint_mod_loss_terms = []
    for mod in src_mod_list:
        out_joint_mod = torch.cat(
            [batch_representations[target_mod], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        pairwise_similarity_tiled_masked = pairwise_similarity_tiled.masked_select(
            mask_joint_mod
        ).view(2 * batch_size, -1)

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = sim_matrix_joint_mod * pairwise_similarity_tiled_masked

        # [2*B]
        loss_joint_mod = -torch.log(
            pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1).unsqueeze(1)
        ).mean(1)
        joint_mod_loss_terms.append(loss_joint_mod)

    return joint_mod_loss_terms


def gmc_supervised_metric_learning_loss_terms(
    batch_representations, pairwise_similarity, temperature, batch_size
):
    """Compute GMC supervised metric learning loss terms."""
    src_mod_list = list(batch_representations.keys())

    matrix_mask = 1 - torch.eye(
        pairwise_similarity.shape[0], device=pairwise_similarity.device
    )

    mod_loss_terms = []
    for mod in src_mod_list:
        sim_matrix_mod = torch.mm(
            batch_representations[mod], batch_representations[mod].t().contiguous()
        )

        sim_mse = (sim_matrix_mod - pairwise_similarity).pow(2)
        loss_mod = (sim_mse * matrix_mask).sum(1)
        mod_loss_terms.append(loss_mod)

    return mod_loss_terms


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
