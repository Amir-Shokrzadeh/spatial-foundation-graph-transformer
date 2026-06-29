```markdown
# Spatial Graph Transformer for Spatial Transcriptomics Representation Learning

### Learning spatial tissue representations by combining Graph Transformers with contrastive self-supervised learning on 10x Visium breast cancer data.

---

## Background

Spatial transcriptomics simultaneously measures gene expression and tissue location at near-single-cell resolution. Traditional pipelines apply PCA and simple neighborhood graphs to summarize gene expression. Recent work instead uses graph neural networks to capture the spatial context of each spot — treating the tissue as a graph where nodes are spots and edges encode spatial proximity.

This project implements a modern spatial transcriptomics pipeline demonstrating:
- Reproducible preprocessing with Scanpy and Squidpy
- Modular embedding interface supporting PCA and foundation models (scGPT, Geneformer)
- Spatial graph construction with k-nearest neighbors and inverse-distance edge weights
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
| Spatial resolution | ~55 µm per spot |

Dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0

---

## Installation

```bash
git clone https://github.com/Amir-Shokrzadeh/spatial-foundation-graph-transformer
cd spatial-foundation-graph-transformer
conda env create -f environment.yml
conda activate sfgt
pip install -e .
```

Download the dataset manually and place files in `data/raw/visium_breast_cancer/`. See `notebooks/01_data_loading.ipynb` for details.

---

## Pipeline

```
Raw Visium Data
      |
      v
Quality Control          (Scanpy, Squidpy)
      |
      v
Normalization + HVGs     (log1p, seurat_v3, 3000 HVGs)
      |
      v
Node Embeddings          (PCA 50-dim | scGPT | Geneformer)
      |
      v
Spatial Graph            (kNN k=6, inverse-distance edge weights)
      |
      v
Graph Transformer        (4 layers, 8 heads, NT-Xent contrastive loss)
      |
      v
Clustering + Evaluation  (Leiden, silhouette, neighborhood preservation)
      |
      v
Visualization            (UMAP, spatial maps, attention weights)
```

---

## Repository Structure

```
spatial-foundation-graph-transformer/
├── configs/default.yaml          # All hyperparameters and paths
├── data/
│   ├── raw/                      # Raw Visium files (not tracked by git)
│   ├── processed/                # Preprocessed AnnData checkpoints
│   └── embeddings/               # Node embedding arrays (.npy)
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_quality_control.ipynb
│   ├── 03_normalization.ipynb
│   ├── 04_embedding.ipynb
│   ├── 05_graph_construction.ipynb
│   ├── 06_models.ipynb
│   ├── 06b_retrain_with_edge_features.ipynb
│   ├── 07_visualization.ipynb
│   └── 08_scgpt_embedding.ipynb
├── src/
│   ├── datasets/                 # Visium data loader
│   ├── preprocessing/            # QC and normalization modules
│   ├── embeddings/               # EmbeddingProvider interface + PCA + scGPT
│   ├── graphs/                   # Spatial kNN graph builder
│   ├── models/                   # MLP, GCN, GraphSAGE, Graph Transformer
│   ├── training/                 # Trainer, NT-Xent loss, augmentation
│   └── utils/                    # Config, seed, logging
├── figures/                      # All generated figures
├── outputs/
│   ├── checkpoints/              # Saved model weights
│   └── metrics/                  # Evaluation results (JSON)
├── train.py                      # Top-level training entry point
├── environment.yml
└── requirements.txt
```

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
| Neighborhood preservation (k=15) | -- | 0.5332 |
| Leiden clusters | 9 | 10 |
| Final training loss | -- | 1.578 |
| Model parameters | -- | 264,064 |

**A note on metrics:** silhouette scores are intentionally low in spatial transcriptomics. Tissue regions have continuous rather than discrete boundaries, so clusters naturally overlap in embedding space. Neighborhood preservation of 0.53 at k=15 is the more informative metric — it means the model preserves over half of each spot's spatial neighborhood in the learned embedding space, without ever being given spatial coordinates as direct input.

The visual story is more informative than clustering metrics alone — see the figures section below.

---

## Biological Interpretation

The 10 Leiden clusters discovered by the Graph Transformer correspond to spatially coherent tissue compartments visible in the H&E image:

- **Tumor core** (large contiguous regions, high total counts) — densely packed malignant epithelial cells with high transcriptional activity
- **Stromal regions** (surrounding tumor mass) — fibroblasts and extracellular matrix components with distinct gene expression signatures
- **Immune border zones** (cluster boundaries, high attention selectivity) — transition regions between tumor and immune infiltrate where the model attends most selectively to neighbors, consistent with active tumor-immune crosstalk
- **Adipose tissue** (peripheral low-count regions) — fat tissue surrounding the tumor with characteristically lower gene expression

Attention selectivity analysis (std of attention weights across neighbors) reveals that boundary spots between tissue compartments show higher selectivity than spots within homogeneous regions. This is consistent with the biological expectation that boundary spots must integrate signals from transcriptionally distinct neighbors.

> Note: cluster annotations above are based on spatial position and expression patterns, not pathologist labels.

---

## Figures

Figures are the primary output of this project. All are generated automatically by the pipeline and saved to `figures/`.

| Figure | Description |
|---|---|
| `figures/spatial/tissue_overview.png` | 2x2 overview: H&E tissue image with spot overlay, Leiden cluster spatial map, UMAP of learned embeddings, total counts heatmap |
| `figures/umap/pca_vs_graph_transformer.png` | Side-by-side comparison of PCA baseline vs Graph Transformer in both UMAP and spatial space — the key result figure |
| `figures/attention/spatial_attention_edge.png` | Mean incoming attention weight and attention selectivity (std across neighbors) mapped onto the tissue — boundary spots show highest selectivity |
| `figures/evaluation/qc_distributions.png` | QC metric distributions with filtering thresholds |
| `figures/evaluation/pca_variance.png` | Per-component and cumulative PCA variance explained |
| `figures/spatial/spatial_graph.png` | kNN spatial graph overlay on tissue spot coordinates |
| `figures/training/graph_transformer_edge_loss.png` | Contrastive training loss curve (NT-Xent, 200 epochs) |

---

## Embedding Interface

The project implements a modular `EmbeddingProvider` interface. Swap the provider by setting `embeddings.provider` in `configs/default.yaml`.

| Provider | Status | Notes |
|---|---|---|
| PCA | Ready | Default, no extra dependencies |
| scGPT | Implemented | Requires GPU + Linux. Windows blocked by torchtext binary incompatibility. |
| Geneformer | Interface ready | Extend `src/embeddings/geneformer_provider.py` |
| UCE | Interface ready | Extend `src/embeddings/uce_provider.py` |

To run scGPT on a GPU machine:

```python
import scgpt
scgpt.download_pretrained_model("human", save_dir="data/raw/scgpt_weights")
```

Then set `embeddings.provider: scgpt` in `configs/default.yaml` and run `train.py`.

---

## Future Work

- Run scGPT embeddings on GPU and benchmark against PCA
- Extend to multi-section or multi-patient datasets
- Add supervised fine-tuning with pathologist annotations
- Implement GraphSAGE and GCN baselines in the full training pipeline
- Add Visium HD support for sub-spot resolution

---

## Author

Amir J. Shokrzadeh — https://github.com/Amir-Shokrzadeh

---
