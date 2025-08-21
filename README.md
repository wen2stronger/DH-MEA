# DH-MEA  

This is source code for the paper *"Dual-Branch Heterophily-Aware Graph Neural Network with Seed Iterative Optimization for Multimodal Entity Alignment"*.  

![The Proposed DH-MEA Framework](image/framework.png)

# Dependency

python-3.10

torch-2.6.0+cpu

torch-scatter-2.1.2+pt26cpu

scipy-1.15.3

tqdm-4.67.1

numpy-2.2.3

# Dataset

## 1. MMKG  
- **Source**：Constructed from DBpedia, Freebase, YAGO  
- **Subsets**：FB15K-DB15K、FB15K-YAGO15K  
- **Content**：15,000 entities per subset (relational facts + numerical attributes + image embeddings)  
- **Use Case**：Evaluate alignment in rich structural/semantic contexts  

## 2. Multi-OpenEA  
- **Enhancement**：Augments OpenEA with multimodal info (entity images from Google search)  
- **Subsets**：2 monolingual (\textit{En-En}, \textit{Zh-Zh}) + 2 cross-lingual (\textit{En-Fr}, \textit{En-De})  
- **Use Case**：Assess alignment across linguistic + visual modalities  

## 3. DBP15K  
- **Source**：Derived from multilingual DBpedia versions  
- **Subsets**：\textit{ZH-EN}, \textit{JA-EN}, \textit{FR-EN}  
- **Use Case**：Evaluate cross-language barrier alignment
