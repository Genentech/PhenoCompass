# PhenoCompass

PhenoCompass is a multimodal deep learning framework designed to enable phenotypic virtual screening at the scale of billions of compounds. It combines Cell Painting morphological profiles with chemical structure embeddings to score molecules against mechanism-of-action (MoA) anchor sets.

Pretrained models and input data are available on Zenodo: [10.5281/zenodo.20367744](https://doi.org/10.5281/zenodo.20367744).

## Installation

```bash
pip install -e .
```

## Directory Structure

```
phenocompass/
├── src/phenocompass/      # Python package
│   ├── models/            # Neural network modules
│   ├── data/              # Dataset and featurization
│   ├── inference/         # Scoring API
│   ├── evaluation/        # Metrics
│   ├── training/          # Training utilities
│   └── configs/           # Hydra configuration files
├── notebooks/             # Manuscript figure generation notebooks and utilities
├── data/                  # Intermediate and cached notebook outputs
├── figures/               # Publication figures and data
├── final_model/           # Pretrained model checkpoints and anchor embeddings
│   ├── model_ckpts/       # Model checkpoint files
│   └── anchors/           # Precomputed anchor embeddings
├── final_data/            # Consolidated input data for notebooks (deposited separately)
│   ├── chembl_drugs/      # ChEMBL reference compounds
│   ├── compound_annotations/  # Anchor and JUMP compound annotations
│   ├── crossmodal_retrieval/  # Cross-modal retrieval inputs
│   ├── embeddings/        # Precomputed embeddings (image, structure, coembedding)
│   ├── enamine/           # Enamine compound results
│   ├── historical_hts/    # Historical HTS results 
│   ├── jump_map/          # JUMP morphological map (parquets, H5, splits)
│   ├── library_similarities/  # Pairwise library Tanimoto similarities
│   ├── loo_comparisons/   # Leave-one-out evaluation outputs
│   └── validation/        # Cell Painting, FOXO3A, kinase inhibition validation data
├── scripts/               # Utility scripts (preprocessing, baselines)
└── pyproject.toml         # Package configuration
```

## Notebooks

All notebooks live in `notebooks/` and are prefixed with their manuscript figure number. Shared configuration and I/O utilities are in `notebooks/utils.py`.

| Notebook | Description |
| --- | --- |
| `Fig1_jump_phenotypic_map.ipynb` | JUMP phenotypic map: evaluation metrics + UMAP visualizations |
| `Fig2_bioactivity_predictions.ipynb` | Bioactivity prediction accuracy comparisons |
| `Fig2_crossmodal_retrieval.ipynb` | Scaffold-cluster splits, Tanimoto similarity, multimodal retrieval |
| `Fig3_anchor_compounds_diversity.ipynb` | Anchor set clustermaps, cosine/Tanimoto heatmaps |
| `Fig3_jump_simulated_vs.ipynb` | Leave-one-out MoA classification baselines |
| `Fig4_historical_hts_simulated_vs.ipynb` | HTS violin plots, statistical tests, drug-anchor similarity |
| `Fig5_cellpainting_foxo3a_validation.ipynb` | Cell Painting, FOXO3A validation |
| `Fig5_hit_analysis.ipynb` | Hit compound summary heatmaps, Tanimoto analysis, molecule grids |
| `FigS1_jump_map_cross_source_reproducibility.ipynb` | Cross-source reproducibility clustermaps |
| `FigS9_jump_hts_enamine_distribution_shift.ipynb` | JUMP-Enamine-HTS Tanimoto similarity distributions |
| `FigS10_enamine_score_threshold.ipynb` | Enamine compound scoring threshold analysis |

## Requirements

- Python >= 3.11
- PyTorch >= 1.12.0
- PyTorch Lightning >= 1.9.0
- PyTorch Geometric >= 2.0.0
- RDKit >= 2022.03.1

See `pyproject.toml` for the complete list of dependencies.

## License

MIT License
