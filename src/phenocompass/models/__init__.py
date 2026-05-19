"""Model components for multimodal contrastive learning."""

from phenocompass.models.components import (
    CommonEncoder,
    GNEpropGIN,
    JointEncoder,
    JointEncoderBatch,
    JointEncoderCtrl,
    MultiLayerPerceptron,
    MultiTask_model,
)
from phenocompass.models.contrastive import (
    DualContrastive_PL,
    MultiModalContrastive_PL,
    MultiTask_PL,
    TripleContrastive_PL,
    TripleContrastive_PL_Batch,
    TripleContrastive_PL_Neighbors,
)
from phenocompass.models.loss import (
    BarlowTwinsLoss,
    ConditionalContrastiveLoss,
    GMCLoss,
    GMCSigmoidLoss,
    GMCSoftNearestNeighborsLoss,
    MOCOPLoss,
    MultiTaskLoss,
    TripleInfoNCELoss,
    TripletLoss,
    get_loss,
)

__all__ = [
    # Components
    "MultiLayerPerceptron",
    "CommonEncoder",
    "JointEncoder",
    "JointEncoderBatch",
    "JointEncoderCtrl",
    "GNEpropGIN",
    "MultiTask_model",
    # Contrastive models
    "MultiModalContrastive_PL",
    "TripleContrastive_PL",
    "DualContrastive_PL",
    "MultiTask_PL",
    "TripleContrastive_PL_Neighbors",
    "TripleContrastive_PL_Batch",
    # Loss functions
    "get_loss",
    "TripleInfoNCELoss",
    "GMCLoss",
    "GMCSigmoidLoss",
    "GMCSoftNearestNeighborsLoss",
    "ConditionalContrastiveLoss",
    "MOCOPLoss",
    "TripletLoss",
    "MultiTaskLoss",
    "BarlowTwinsLoss",
]
