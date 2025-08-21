import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from config import *

parser = argparse.ArgumentParser(description='MSP process (Vis) for PathFusion')
parser.add_argument('--dataset', type=str, default='fr_en', help='dataset name', choices=['OEA_D_W_15K_V1', 'OEA_D_W_15K_V2', 'OEA_EN_DE_15K_V1', 'OEA_EN_FR_15K_V1',
                                 'fr_en', 'ja_en', 'zh_en'])

args = parser.parse_args()

# 1) Load image embeddings (entity_id -> embedding vector)
with open(os.path.join(data_root, 'Vis', f'{args.dataset}.pkl'), 'rb') as f:
    id_img_feature = pickle.load(f)

# 2) Load entity ID lists, taking only the first column (entity ID)
with open(os.path.join(data_root, args.dataset, 'ent_ids_1'), 'rb') as f:
    ent_ids_1 = [int(line.split()[0].strip()) for line in f.readlines()]  # take only the first column (ID)

with open(os.path.join(data_root, args.dataset, 'ent_ids_2'), 'rb') as f:
    ent_ids_2 = [int(line.split()[0].strip()) for line in f.readlines()]  # take only the first column (ID)

# 3) Probe one embedding to get the dimensionality (assumes uniform length)
#    Iterate over KG1 entities and pick the first one that has an embedding
sample_embedding = None
for ent_id in ent_ids_1:
    sample_embedding = id_img_feature.get(ent_id, None)
    if sample_embedding is not None:
        break  # found one with an embedding, stop scanning

# Ensure at least one entity in KG1 has a visual embedding
if sample_embedding is None:
    raise ValueError("The entities in Graph 1 do not have any visual embeddings")

# Determine embedding dimension
embedding_length = len(sample_embedding)

# 4) Build feature vectors for KG1 and KG2
#    For entities without embeddings, use a zero vector of length embedding_length
kg1_features = [id_img_feature.get(ent_id, np.zeros(embedding_length)) for ent_id in ent_ids_1]
kg2_features = [id_img_feature.get(ent_id, np.zeros(embedding_length)) for ent_id in ent_ids_2]

# 5) Stack embeddings in entity order and save
all_features = kg1_features + kg2_features  # concatenate KG1 and KG2 features
embedding_array = np.array(all_features)    # shape: (len(ent_ids_1) + len(ent_ids_2), embedding_length)
np.save(os.path.join(data_root, args.dataset, 'Emb', 'vis_embedding.npy'), embedding_array)

# 6) Compute cross-graph visual similarity matrix (KG1 vs KG2)
similarity_matrix = cosine_similarity(np.array(kg1_features), np.array(kg2_features))

# 7) Save the similarity matrix
np.save(os.path.join(data_root, args.dataset, 'Score Matrix', 'Vis.npy'), similarity_matrix)
