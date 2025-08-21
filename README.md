# DH-MEA  

This is source code for the paper *"Dual-Branch Heterophily-Aware Graph Neural Network with Seed Iterative Optimization for Multimodal Entity Alignment"*.  

![The Proposed DH-MEA Framework](image/framework.png)

# Dependency

- **python-3.10**

- **torch-2.6.0+cpu**

- **torch-scatter-2.1.2+pt26cpu**

- **scipy-1.15.3**

- **tqdm-4.67.1**

- **numpy-2.2.3**

# Dataset

We utilize three multimodal knowledge graph datasets:

- **MMKG:** FB15K-DB15K, FB15K-YAGO15K

- **Multi-OpenEA:** EN-FR-V1, EN-DE-V1, D-W-V1, D-W-V2

- **DBP15K:** ZH-EN, JA-EN, FR-EN

Basic information of these datasets is introduced above. For additional visual information related to them, please refer to the content placed [here]<https://translate.google.hk/?sl=auto&tl=en&text=%E6%AD%A4%E5%A4%96%EF%BC%8C%E7%94%A8%E4%BA%8E%E8%8E%B7%E5%8F%96%E5%B1%9E%E6%80%A7%E5%B5%8C%E5%85%A5%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%9C%A8%E8%BF%99%E9%87%8C%E3%80%82&op=translate)>.

Additionally, the model used to obtain attribute embeddings is [here](docs/visual-info.md).
