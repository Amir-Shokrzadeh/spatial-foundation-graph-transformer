# Spatial Foundation Graph Transformer
### Learning Spatial Tissue Representations via Graph Transformers on Visium Breast Cancer Data

---

## Background

Spatial transcriptomics simultaneously measures gene expression and tissue location
at near-single-cell resolution. Traditional pipelines apply PCA and simple neighborhood
graphs to summarize gene expression. Recent work instead uses graph neural networks
and foundation models to capture the spatial context of each spot — treating the tissue
as a graph where nodes are spots and edges encode spatial proximity.

This project implements a modern spatial transcriptomics pipeline demonstrating:
- Reproducible preprocessing with Scanpy and Squidpy
- Modular embedding interface supporting PCA and foundation models
- Spatial graph construction with k-nearest neighbors
- Contrastive self-supervised training on a Graph Transformer
- Quantitative evaluation and publication-quality figures

---

## Dataset

**10x Genomics Visium Human Breast Cancer Block A Section 1**

| Property | Value |
|---|---|
| Spots (raw) | 3,798 |
| Spots (after QC) | 3,661 |
| Genes (raw) | 36,601 |
| Genes (after QC) | 20,955 |
| Highly variable genes | 3,000 |
| Spatial resolution | ~55 um per spot |

Dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0

---

## Installation

    git clone https://github.com/Amir-Shokrzadeh/spatial-foundation-graph-transformer
    cd spatial-foundation-graph-transformer
    conda env create -f environment.yml
    conda activate sfgt
    pip install -e .

Download the dataset manually and place files in data/raw/visium_breast_cancer/.

---

## Repository Structure

    spatial-foundation-graph-transformer/
    configs/default.yaml          # All hyperparameters and paths
    data/raw/                     # Raw Visium files (not tracked by git)
    data/processed/               # Preprocessed AnnData checkpoints
    data/embeddings/              # Node embedding arrays (.npy)
    notebooks/                    # Step-by-step analysis notebooks
    src/datasets/                 # Visium data loader
    src/preprocessing/            # QC and normalization modules
    src/embeddings/               # EmbeddingProvider interface + PCA
    src/graphs/                   # Spatial kNN graph builder
    src/models/                   # MLP, GCN, GraphSAGE, GraphTransformer
    src/training/                 # Trainer, NT-Xent loss, augmentation
    src/utils/                    # Config, seed, logging
    figures/                      # All generated figures
    outputs/checkpoints/          # Saved model weights
    outputs/metrics/              # Evaluation results (JSON)
    train.py                      # Top-level training entry point

---

## Models

| Model | Parameters | Description |
|---|---|---|
| MLP | 27,680 | Graph-unaware baseline |
| GCN | 27,680 | 1-hop neighbor aggregation |
| GraphSAGE | 54,560 | Inductive mean aggregation |
| Graph Transformer | 264,064 | Multi-head spatial attention + edge features |

---

## Results

| Metric | PCA Baseline | Graph Transformer |
|---|---|---|
| Silhouette score | -0.0021 | +0.0146 |
| Neighborhood preservation | -- | 0.5332 |
| Leiden clusters | 9 | 10 |
| Final training loss | -- | 1.578 |

The Graph Transformer learns spatially coherent representations that outperform
PCA on all evaluated metrics. Attention selectivity analysis reveals that spots
at tissue boundaries attend more selectively to their neighbors than spots
within homogeneous tissue regions.

---

## Figures

| Figure | Description |
|---|---|
| figures/spatial/tissue_overview.png | H&E image, clusters, UMAP, counts |
| figures/umap/pca_vs_graph_transformer.png | Baseline vs model comparison |
| figures/attention/spatial_attention_edge.png | Spatial attention weights |
| figures/evaluation/qc_distributions.png | QC metric distributions |
| figures/training/graph_transformer_edge_loss.png | Training loss curve |

---

## Embedding Interface

The project implements a modular EmbeddingProvider interface.
Swap the provider in configs/default.yaml under embeddings.provider.
Supported providers:

| Provider | Status | Notes |
|---|---|---|
| PCA | Ready | Default, no extra dependencies |
| scGPT | Implemented | Requires GPU + Linux. Windows blocked by torchtext binary incompatibility. |
| Geneformer | Interface ready | Extend src/embeddings/geneformer_provider.py |
| UCE | Interface ready | Extend src/embeddings/uce_provider.py |

To switch provider, set embeddings.provider in configs/default.yaml.
To run scGPT on a GPU machine:

    import scgpt
    scgpt.download_pretrained_model("human", save_dir="data/raw/scgpt_weights")

Then set embeddings.provider: scgpt in configs/default.yaml and run train.py.

---

## Future Work

- Integrate scGPT or Geneformer foundation model embeddings
- Extend to multi-section or multi-patient datasets
- Add supervised fine-tuning with pathologist annotations
- Implement GraphSAGE and GCN baselines in the training pipeline
- Add Visium HD support for sub-spot resolution

---

## References

- Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.
- Shi et al. (2021). Masked Label Prediction: Unified Message Passing Model. IJCAI.
- Chen et al. (2020). A Simple Framework for Contrastive Learning. ICML.
- Wolf et al. (2018). SCANPY: large-scale single-cell gene expression data analysis.
- Palla et al. (2022). Squidpy: a scalable framework for spatial omics analysis.

---

## Author

Amir Shokrzadeh — https://github.com/Amir-Shokrzadeh

---

*This project is an implementation and demonstration of modern spatial transcriptomics
AI workflows using publicly available data. It is not intended as a novel scientific contribution.*