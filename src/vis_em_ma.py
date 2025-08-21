import argparse
import numpy as np
from tqdm import tqdm
from sinkhorn import matrix_sinkhorn
import sparse_eval
import torch
import pickle
from config import *

# Setup argument parser
parser = argparse.ArgumentParser(description='MSP process (Vis) for PathFusion')
parser.add_argument('--dataset', type=str, default='YAGO15K-FB15K', help='Dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])
parser.add_argument('--max_image_num', type=int, default=6, help='Max number of images per entity', choices=[1, 2, 3, 4, 5, 6])

args = parser.parse_args()

# Parse source and target datasets from argument
source_dataset, target_dataset = args.dataset.split('-')
source_dataset = source_dataset.lower()
target_dataset = target_dataset.lower()
use_img_num = args.max_image_num

# Load source and target embeddings, and entity to image mappings
with open(os.path.join(data_root, 'Vis', f'{source_dataset}.npy'), 'rb') as f:
    source_embedding = pickle.load(f)  # Shape: [#images, dim]
with open(os.path.join(data_root, 'Vis', f'{source_dataset}'), 'rb') as f:
    source_id2img = pickle.load(f)  # Key: entity ID -> list of image indices

with open(os.path.join(data_root, 'Vis', f'{target_dataset}.npy'), 'rb') as f:
    target_embedding = pickle.load(f)
with open(os.path.join(data_root, 'Vis', f'{target_dataset}'), 'rb') as f:
    target_id2img = pickle.load(f)

# Load entity IDs from file
def load_entity_ids(path):
    """Load entity IDs from the specified file."""
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:  # Skip empty lines
                ids.append(parts[0])  # Assuming the ID is the first element
    return ids

# Define paths for source and target entity IDs
src_path = os.path.join(data_root, args.dataset, "ent_ids_1")
tgt_path = os.path.join(data_root, args.dataset, "ent_ids_2")

source_ids = load_entity_ids(src_path)
target_ids = load_entity_ids(tgt_path)

source_len = len(source_ids)
target_len = len(target_ids)

# Load triples and build adjacency matrix for entities
def load_triples_and_build_adjacency(triples_path):
    """Build adjacency matrix from triples in the specified file."""
    adj_matrix = {}
    with open(triples_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue  # Skip malformed lines
            h, r, t = parts
            h, t = int(h), int(t)  # Ensure entity IDs are integers
            if h not in adj_matrix:
                adj_matrix[h] = []
            if t not in adj_matrix:
                adj_matrix[t] = []
            adj_matrix[h].append(t)
            adj_matrix[t].append(h)  # Assuming undirected relations
    return adj_matrix

# Define paths for source and target triples
triples_path1 = os.path.join(data_root, args.dataset, 'triples_1')
triples_path2 = os.path.join(data_root, args.dataset, 'triples_2')

# Load adjacency matrices for both source and target entities
adj_matrix1 = load_triples_and_build_adjacency(triples_path1)
adj_matrix2 = load_triples_and_build_adjacency(triples_path2)

# Initialize embeddings as zero vectors
def initialize_embeddings(ent_ids, dim=1536):
    """Initialize embeddings for entities with zero vectors."""
    return np.zeros((len(ent_ids), dim))

# Aggregate visual features for each entity
def aggregate_attributes(ent_ids, id2img, vis_embeddings, initial_embeddings):
    """Aggregate image embeddings based on entity's image indices."""
    for ent_id in ent_ids:
        ent_id_ = int(ent_id)
        # Check for valid entity indices and aggregate images for source
        if ent_id_ < source_len and ent_id_ in id2img:
            indices = [index for index in id2img[ent_id_]]
            if indices:
                aggregated_embedding = np.mean(vis_embeddings[indices], axis=0)
                initial_embeddings[ent_id_] = aggregated_embedding
        # For target entities
        elif ent_id_ >= source_len and ent_id_ - source_len in id2img:
            indices = [index for index in id2img[ent_id_ - source_len]]
            if indices:
                aggregated_embedding = np.mean(vis_embeddings[indices], axis=0)
                initial_embeddings[ent_id_ - source_len] = aggregated_embedding
    return initial_embeddings

# Update embeddings for entities whose embeddings are all zeros
def update_embeddings(ent_ids, adj_matrix, embeddings):
    """Update embeddings for entities with zero embeddings based on their neighbors."""
    for ent_id in ent_ids:
        ent_id_ = int(ent_id)
        # Update embeddings for source entities
        if ent_id_ < source_len and np.all(embeddings[ent_id_] == 0):
            if ent_id_ in adj_matrix:
                valid_neighbors = [embeddings[nbr] for nbr in adj_matrix[ent_id_] if np.any(embeddings[nbr] != 0)]
                if valid_neighbors:
                    embeddings[ent_id_] = np.mean(valid_neighbors, axis=0)
        # Update embeddings for target entities
        elif ent_id_ >= source_len and np.all(embeddings[ent_id_ - source_len] == 0):
            if ent_id_ in adj_matrix:
                valid_neighbors = [embeddings[nbr - source_len] for nbr in adj_matrix[ent_id_] if np.any(embeddings[nbr - source_len] != 0)]
                if valid_neighbors:
                    embeddings[ent_id_ - source_len] = np.mean(valid_neighbors, axis=0)
    return embeddings

# Initialize embeddings for source and target entities
initial_embeddings1 = initialize_embeddings(source_ids)
initial_embeddings2 = initialize_embeddings(target_ids)

# Aggregate visual embeddings for both source and target
new_embeddings1 = aggregate_attributes(source_ids, source_id2img, source_embedding, initial_embeddings1)
new_embeddings2 = aggregate_attributes(target_ids, target_id2img, target_embedding, initial_embeddings2)

# Update embeddings for entities with zero embeddings
final_embeddings1 = update_embeddings(source_ids, adj_matrix1, new_embeddings1)
final_embeddings2 = update_embeddings(target_ids, adj_matrix2, new_embeddings2)

# Stack the final embeddings of source and target entities
final_embeddings = np.vstack([final_embeddings1, final_embeddings2])

# Save the final embeddings to file
np.save(os.path.join(data_root, args.dataset, 'Emb', 'vis_embedding.npy'), final_embeddings)

# Calculate similarity scores between source and target embeddings
image_scores = np.zeros((source_len, target_len))
image_scores = -float('inf') * np.ones((source_len, target_len))

for i in tqdm(range(source_len)):
    for j in range(target_len):
        for ii in range(min(use_img_num, len(source_id2img[i]))):
            for jj in range(min(use_img_num, len(target_id2img[j]))):
                image_scores[i, j] = max(image_scores[i, j], np.dot(source_embedding[source_id2img[i][ii]], target_embedding[target_id2img[j][jj]]))

# Convert image scores to final score matrix, replacing -inf with 0
scores = np.zeros((len(source_ids), len(target_ids)), dtype=np.float32)
for i in range(len(source_ids)):
    for j in range(len(target_ids)):
        scores[i][j] = image_scores[i][j] if image_scores[i][j] != -float('inf') else 0

# Normalize the scores to [0, 1]
scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# Save the final similarity scores to file
np.save(os.path.join(data_root, args.dataset, 'Score Matrix', 'Vis.npy'), scores)
