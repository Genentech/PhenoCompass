"""Neural network building blocks for multimodal contrastive learning."""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    ModuleDict,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.nn.functional import logsigmoid
from torch_geometric.nn import (
    GINEConv,
    JumpingKnowledge,
    global_add_pool,
    global_mean_pool,
    GraphMultisetTransformer,
    GCNConv,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from tqdm import tqdm

from phenocompass.models.loss import get_loss, MultiTaskLoss
from phenocompass.models import utils as ut


class MultiLayerPerceptron(nn.Module):
    """Standard multi-layer perceptron with non-linearity and dropout.

    Parameters
    ----------
    num_input_features : int
        Input dimension.
    hidden_layer_dimensions : list of int
        List of hidden layer dimensions.
    output_size : int
        Number of output features.
    nonlin : str or nn.Module, default "ReLU"
        Name of a nonlinearity in torch.nn, or a pytorch Module.
    p_dropout : float, default 0.0
        Dropout probability for dropout layers.
    """

    def __init__(
        self,
        num_input_features: int,
        hidden_layer_dimensions: Optional[List[int]],
        output_size: int,
        nonlin: Union[str, nn.Module] = "ReLU",
        p_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)

        if isinstance(nonlin, str):
            nonlin = getattr(torch.nn, nonlin)()

        hidden_layer_dimensions = [dim for dim in hidden_layer_dimensions if dim != 0]
        layer_inputs = [num_input_features] + hidden_layer_dimensions
        modules = []
        for i in range(len(hidden_layer_dimensions)):
            modules.extend(
                [
                    nn.Linear(layer_inputs[i], layer_inputs[i + 1]),
                    nn.BatchNorm1d(layer_inputs[i + 1]),
                    nonlin,
                    nn.Dropout(p=p_dropout),
                ]
            )

        modules.append(nn.Linear(layer_inputs[-1], output_size))

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class CommonEncoder(torch.nn.Module):
    """Common encoder for modality-specific and joint embeddings.

    Applies MLP followed by L2 normalization.

    Parameters
    ----------
    common_dim : int
        Input dimensions.
    hidden_layer_dimensions : list of int
        Hidden layer dimensions.
    latent_dim : int
        Output embedding dimension.
    nonlin : str, default "ReLU"
        Nonlinearity name.
    """

    def __init__(self, common_dim, hidden_layer_dimensions, latent_dim, nonlin="ReLU"):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.nonlin = nonlin
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)

        self.feature_extractor = MultiLayerPerceptron(
            num_input_features=self.common_dim,
            hidden_layer_dimensions=hidden_layer_dimensions,
            output_size=self.latent_dim,
            nonlin=self.nonlin,
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class JointEncoder(torch.nn.Module):
    """Joint encoder that concatenates modality-specific encodings.

    Parameters
    ----------
    encoders_mod : dict
        Dictionary mapping modality names to encoder modules.
    """

    def __init__(self, encoders_mod):
        super().__init__()
        self.encoders = ModuleDict()
        if isinstance(encoders_mod, Dict):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        self.encoders.update(encoders_mod)

    def forward(self, x_dict):
        list_emb_mod_ = []
        for mod_name, x in x_dict.items():
            list_emb_mod_.append(self.encoders[mod_name](x))

        return torch.cat(list_emb_mod_, dim=-1)


class JointEncoderBatch(torch.nn.Module):
    """Joint encoder compatible with batch labels (tuple inputs)."""

    def __init__(self, encoders_mod):
        super().__init__()
        self.encoders = ModuleDict()
        if isinstance(encoders_mod, Dict):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        self.encoders.update(encoders_mod)

    def forward(self, x_dict):
        list_emb_mod_ = []
        for mod_name, x in x_dict.items():
            x_feat, x_meta = x
            list_emb_mod_.append(self.encoders[mod_name](x_feat))

        return torch.cat(list_emb_mod_, dim=-1)


class JointEncoderCtrl(torch.nn.Module):
    """Joint encoder with control modality subtraction."""

    def __init__(self, encoders_mod, mod_to_normalize, ctrl_mod="ctrl"):
        super().__init__()
        self.encoders = ModuleDict()
        if isinstance(encoders_mod, Dict):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        self.encoders.update(encoders_mod)
        self.mod_to_normalize = mod_to_normalize
        self.ctrl_mod = ctrl_mod

    def forward(self, x_dict):
        list_emb_mod_ = []
        for mod_name, x in x_dict.items():
            if mod_name != self.ctrl_mod:
                emb_mod_ = self.encoders[mod_name](x)

                # normalize embedding
                if mod_name == self.mod_to_normalize:
                    x_ctrl = x_dict[self.ctrl_mod]
                    emb_ctrl_mod_ = self.encoders[self.mod_to_normalize](x_ctrl)
                    emb_mod_ -= emb_ctrl_mod_

                list_emb_mod_.append(emb_mod_)

        return torch.cat(list_emb_mod_, dim=-1)


class GNEpropGIN(torch.nn.Module):
    """Graph Neural Network (GIN-based) for molecular property prediction.

    Parameters
    ----------
    in_channels : int
        Node feature dimension.
    edge_dim : int
        Edge feature dimension.
    hidden_channels : int
        Hidden layer dimension.
    ffn_hidden_channels : int, optional
        FFN hidden dimension.
    num_layers : int
        Number of GIN layers.
    out_channels : int
        Output dimension.
    num_readout_layers : int
        Number of readout MLP layers.
    dropout : float
        Dropout probability.
    mol_features_size : int
        Size of additional molecular features.
    aggr : str, default "mean"
        Aggregation method ("mean", "sum", "global_attention", "gmt").
    jk : str, default "cat"
        Jumping knowledge mode ("cat" or "none").
    gmt_args : dict, optional
        Arguments for GraphMultisetTransformer.
    use_proj_head : bool, default False
        Whether to use projection head.
    proj_dims : tuple, optional
        Projection head dimensions.
    skip_last_relu : bool, default False
        Skip ReLU in last conv layer.
    num_classes : int, optional
        Number of output classes.
    """

    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden_channels,
        ffn_hidden_channels,
        num_layers,
        out_channels,
        num_readout_layers,
        dropout,
        mol_features_size,
        aggr="mean",
        jk="cat",
        gmt_args=None,
        use_proj_head=False,
        proj_dims=(512, 256),
        skip_last_relu=False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        if num_classes is None:
            self.has_final_layer = False
        else:
            self.has_final_layer = True

        # graph encoder
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels)

        self.convs = ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1 and skip_last_relu:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    Dropout(p=dropout),
                )
            else:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(inplace=True),
                    Dropout(p=dropout),
                )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)

        self.jk_mode = jk
        if self.jk_mode == "none":
            self.jk = None
        else:
            self.jk = JumpingKnowledge(
                mode=self.jk_mode, channels=hidden_channels, num_layers=num_layers
            )

        # classifier
        self.classifier = ModuleList()

        if self.jk_mode == "none":
            hidden_channels_mol = hidden_channels + mol_features_size
        elif self.jk_mode == "cat":
            hidden_channels_mol = hidden_channels * (num_layers + 1) + mol_features_size
        else:
            raise NotImplementedError

        ffn_hidden_size = (
            int(ffn_hidden_channels) if ffn_hidden_channels is not None else int(hidden_channels_mol)
        )

        for layer in range(num_readout_layers):
            input_dim = int(hidden_channels_mol) if layer == 0 else int(ffn_hidden_size)
            mlp = Sequential(
                Linear(input_dim, ffn_hidden_size),
                BatchNorm1d(ffn_hidden_size),
                ReLU(inplace=True),
                Dropout(p=dropout),
            )
            self.classifier.append(mlp)

        # last layer (classifier)
        input_dim = hidden_channels_mol if num_readout_layers == 0 else ffn_hidden_size
        self.classifier.append(
            Linear(input_dim, int(out_channels)),
        )

        self.aggr = aggr
        self.global_pool = None
        if aggr == "mean":
            self.global_pool = global_mean_pool
        elif aggr == "sum":
            self.global_pool = global_add_pool
        elif aggr == "global_attention":
            hidden_channels_without_mol = hidden_channels_mol - mol_features_size
            hidden_ga_channels = int(hidden_channels_without_mol / 2)
            gate_nn = Sequential(
                Linear(hidden_channels_without_mol, hidden_ga_channels),
                BatchNorm1d(hidden_ga_channels),
                ReLU(inplace=True),
                Linear(hidden_ga_channels, 1),
            )
            self.global_pool = AttentionalAggregation(gate_nn)
        elif aggr == "gmt":
            assert gmt_args is not None

            gmt_sequences = [
                ["GMPool_I"],
                ["GMPool_G"],
                ["GMPool_G", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            ]

            gmt_sequence = gmt_sequences[gmt_args["gmt_sequence"]]

            self.global_pool = GraphMultisetTransformer(
                in_channels=hidden_channels_mol,
                hidden_channels=gmt_args["hidden_channels"],
                out_channels=hidden_channels_mol,
                Conv=GCNConv,
                num_nodes=200,
                pooling_ratio=gmt_args["gmt_pooling_ratio"],
                pool_sequences=gmt_sequence,
                num_heads=gmt_args["gmt_num_heads"],
                layer_norm=gmt_args["gmt_layer_norm"],
            )

        self.use_proj_head = use_proj_head
        self.proj_dims = proj_dims
        if self.use_proj_head and self.proj_dims is not None:
            self.proj_head = ModuleList()

            input_dim = hidden_channels_mol
            for proj_dim in self.proj_dims[:-1]:
                mlp = Sequential(
                    Linear(input_dim, proj_dim),
                    BatchNorm1d(proj_dim),
                    ReLU(inplace=True),
                    Dropout(p=dropout),
                )
                self.proj_head.append(mlp)
                input_dim = proj_dim

            # last proj layer
            self.proj_head.append(
                Linear(input_dim, proj_dims[-1]),
            )

    def load_pretrained(self, pretrain_path):
        """Load pretrained weights from checkpoint.

        Parameters
        ----------
        pretrain_path : str
            Path to checkpoint file or directory.
        """
        from phenocompass.models.clr import SimCLR

        print(pretrain_path)
        if os.path.isfile(pretrain_path):
            model_pretrained = SimCLR.load_from_checkpoint(pretrain_path)
        elif os.path.isdir(pretrain_path):
            pretrain_full_path = ut.get_checkpoint_paths(checkpoint_dir=pretrain_path)[0]
            model_pretrained = SimCLR.load_from_checkpoint(pretrain_full_path)
        else:
            raise ValueError("Invalid argument: pretrain path")

        self.load_state_dict(model_pretrained.encoder.state_dict(), strict=False)

        # freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def compute_representations(self, x, edge_index, edge_attr, batch, perturb=None):
        """Compute graph-level representations."""
        list_graph_encodings = []

        x_encoded = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if perturb is not None:
            if "perturb_a" in perturb:
                x_encoded += perturb["perturb_a"]
            if "perturb_b" in perturb:
                edge_attr += perturb["perturb_b"]

        if self.jk_mode != "none":
            list_graph_encodings.append(x_encoded)

        for conv in self.convs:
            x_encoded = conv(x_encoded, edge_index, edge_attr)
            if self.jk_mode != "none":
                list_graph_encodings.append(x_encoded)

        if self.jk_mode != "none":
            x_encoded = self.jk(list_graph_encodings)

        if self.aggr in ["gmt"]:
            out = self.global_pool(x_encoded, batch, edge_index)
        else:
            out = self.global_pool(x_encoded, batch)  # [batch_size, hidden_channels]

        if perturb is not None:
            if "perturb_graph" in perturb:
                out += perturb["perturb_graph"]
        return out

    def forward(
        self, x_struct, mol_features=None, restrict_output_layers=0, output_type=None, perturb=None, **kwargs
    ):
        """Forward pass.

        Parameters
        ----------
        x_struct : torch_geometric.data.Data
            Molecular graph batch.
        mol_features : torch.Tensor, optional
            Additional molecular features.
        restrict_output_layers : int, default 0
            Number of classifier layers to use.
        output_type : str, optional
            Output type ("prob", "log_prob", etc.).
        perturb : dict, optional
            Perturbation parameters.

        Returns
        -------
        torch.Tensor
            Output embeddings or predictions.
        """
        x = x_struct.x
        edge_index = x_struct.edge_index
        edge_attr = x_struct.edge_attr
        batch = x_struct.batch

        # compute graph emb
        out_repr = self.compute_representations(x, edge_index, edge_attr, batch, perturb=perturb)

        if mol_features is not None:
            out_repr = torch.cat((out_repr, mol_features), dim=1)

        # compute classifier
        out = out_repr

        if self.has_final_layer:
            for mlp in self.classifier[
                : None if restrict_output_layers == 0 else restrict_output_layers
            ]:
                out = mlp(out)

            # (optionally) compute proj head
            if self.use_proj_head:
                if self.proj_dims is None:
                    out_proj = out_repr[-500:]
                else:
                    out_proj = out_repr
                    for mlp in self.proj_head:
                        out_proj = mlp(out_proj)

            if output_type is not None:
                if output_type == "prob":
                    out = torch.sigmoid(out)
                elif output_type == "log_prob":
                    out = torch.nn.functional.logsigmoid(out)
                elif output_type == "prob_multiclass":
                    out = torch.sigmoid(out)
                    return torch.cat((1 - out, out), dim=-1)
                elif output_type == "log_prob_multiclass":
                    return torch.cat(
                        (logsigmoid(-out), logsigmoid(out)), dim=-1
                    )  # equivalent to log(1-sigmoid(x)), log(sigmoid(x))
                else:
                    raise NotImplementedError

        if self.use_proj_head:
            return out, out_proj
        else:
            return out

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mlp in self.classifier:
            mlp.reset_parameters()
        self.input_batch_norm.reset_parameters()


class MultiTask_model(torch.nn.Module):
    """Multi-task classification model with frozen backbone."""

    def __init__(self, backbone, loss_name, num_tasks, mod_name, lr=0.001):
        super().__init__()
        self.backbone = backbone
        self.optimizer = None
        input_dim = self.backbone.hparams.latent_dim
        self.linear_layer = nn.Linear(input_dim, num_tasks)
        self.loss = get_loss(name=loss_name)
        self.mod_name = mod_name
        self.optimizer = torch.optim.Adam(self.linear_layer.parameters(), lr=lr)

    def forward(self, batch):
        probs = self._forward_with_sigmoid(batch, device="cuda", return_mod=None)
        return probs

    def _forward_with_sigmoid(self, batch, device="cuda", mod_name=None, return_mod=None):
        label = batch["labels"]
        label = label.to(device)
        x_dict = batch["inputs"]
        x_dict = ut.move_batch_input_to_device(x_dict, device=device)

        self.backbone.freeze()
        shared_reprs = self.backbone.compute_representations(x_dict, mod_name=mod_name)
        logits = self.linear_layer(shared_reprs)
        probs = torch.sigmoid(logits)
        loss = self.loss(logits, label)

        if return_mod == "label":
            return probs, label
        elif return_mod == "logits":
            return probs, logits
        elif return_mod is None:
            return probs

    def predict_probs_dataloader(
        self,
        dataloader,
        device="cuda",
        return_mod="label",
        mod_name=None,
        disable_progress_bar=False,
        return_mol=False,
    ):
        probs_list = []
        label_list = []
        mols = []
        with torch.no_grad():
            for batch in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                probs, label = self._forward_with_sigmoid(
                    batch, device=device, mod_name=mod_name, return_mod=return_mod
                )
                probs_list.append(probs.cpu().data.numpy())
                label_list.append(label.cpu().data.numpy())
                if return_mol:
                    mols.extend([*batch["inputs"]["struct"].mols])
        if return_mol:
            return np.vstack(probs_list), np.vstack(label_list), mols
        else:
            return np.vstack(probs_list), np.vstack(label_list)
