"""PhenoCompass: Multimodal deep learning for phenotypic virtual screening.

This package provides tools for scoring molecules against mechanism-of-action (MoA)
anchor sets using multimodal contrastive learning on Cell Painting morphological
profiles and chemical structure embeddings.

Main API:
    PhenoCompass: Score query molecules against curated MoA anchor sets.

Example:
    >>> from phenocompass import PhenoCompass
    >>> model = PhenoCompass(final_model_dir="./final_model")
    >>> scores = model.score_against_anchors(["CCO", "c1ccccc1"])
    >>> struct_emb = model.compute_struct_embeddings(["CCO", "c1ccccc1"])
"""

__version__ = "0.1.0"

from phenocompass.inference.scorer import PhenoCompass

__all__ = ["PhenoCompass", "__version__"]
