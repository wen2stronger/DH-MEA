import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import argparse
import os
from config import *  

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer(model_att_root).to(device)

# Parse command line arguments
parser = argparse.ArgumentParser(description='MSP process (Vis) for PathFusion')
parser.add_argument('--dataset', type=str, default='zh_en', help='Dataset name', choices=['OEA_D_W_15K_V1', 'OEA_D_W_15K_V2', 'OEA_EN_DE_15K_V1', 'OEA_EN_FR_15K_V1',
                                 'fr_en', 'ja_en', 'zh_en'])
args = parser.parse_args()

# Function to get embedding for an attribute (text)
def get_embedding_for_attr(attr_text):
    """Encode attribute text using the SentenceTransformer model."""
    emb = model.encode(attr_text, convert_to_tensor=True, device=device)
    return emb.cpu()  # Move tensor to CPU for further processing

# Function to extract the entity name from a URI
def extract_entity_name(uri):
    """Extract the entity name from a URI (get the last part after the final slash)."""
    if not uri:
        return uri
    return uri.rstrip('/').split('/')[-1]

# Function to load entity IDs and their names from a file
def load_entity_ids(filepath):
    """
    Reads an entity ID file, where each line contains: entity_id \t entity_name (URL format)
    Returns two lists: entity_ids and processed entity names (last part of URL).
    """
    entity_ids = []
    entity_names = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            eid = parts[0].strip()
            raw_name = parts[1].strip()
            clean_name = extract_entity_name(raw_name)
            entity_ids.append(eid)
            entity_names.append(clean_name)
    return entity_ids, entity_names

# Function to load the attribute dictionary with entity names as keys
def load_attr_dict_with_name_key(attr_file):
    """
    Reads an attribute file where each line contains: entity_name (URL format) \t attribute1 \t attribute2 ...
    Returns a dictionary with the processed entity name as the key and a list of attributes as the value.
    """
    d = {}
    with open(attr_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            raw_entity_name = parts[0].strip()
            entity_name = extract_entity_name(raw_entity_name)
            attr_names = []
            for attr_field in parts[1:]:
                for attr in attr_field.strip().split():
                    attr_clean = extract_entity_name(attr)
                    attr_names.append(attr_clean)
            attr_names.insert(0, entity_name)  # Add the entity name as the first attribute
            d[entity_name] = attr_names
    return d

# Function to build a dictionary of entity IDs to their attributes
def build_entity_attr_dict(entity_ids, entity_names, attr_name_dict):
    """
    Build a dictionary where entity IDs are the keys and the values are their respective attributes from attr_name_dict.
    """
    d = {}
    for eid, ename in zip(entity_ids, entity_names):
        attrs = attr_name_dict.get(ename)
        if attrs is None:
            attrs = [ename]  # Ensure there's at least the entity name
        d[eid] = attrs
    return d

# Function to load triples from a file
def load_triples(filepath):
    """
    Reads a triples file with format: head_entity \t relation \t tail_entity
    Returns a list of triples (tuples of entity pairs).
    """
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples

# Function to initialize an attribute embedding matrix for entities
def init_attr_embedding_matrix(num_entities, embedding_dim=1024):
    """
    Initializes an attribute embedding matrix with zeros for all entities.
    The matrix shape is (num_entities, embedding_dim).
    """
    return torch.zeros((num_entities, embedding_dim), dtype=torch.float)

# Function to build attribute embeddings for entities based on their attributes (BATCHED)
def build_entity_attr_embedding(entity_ids, attr_dict, batch_size=256):
    """
    Builds attribute embeddings for each entity using batched encoding for speed.
    Returns a dict: entity_id -> List[Tensor] (one tensor per attribute).
    """
    # 1) Flatten all attributes into a single list while keeping ownership mapping
    flat_texts = []
    owners = []  # owners[k] = entity_id that owns flat_texts[k]
    for eid in entity_ids:
        attrs = attr_dict.get(eid, [])
        if not attrs:
            continue
        flat_texts.extend(attrs)
        owners.extend([eid] * len(attrs))

    # Edge case: no attributes at all
    if len(flat_texts) == 0:
        return {eid: [] for eid in entity_ids}

    # 2) Encode in batches
    all_embeds = []
    for start in tqdm(range(0, len(flat_texts), batch_size), desc="Setting attribute embedding"):
        batch_texts = flat_texts[start:start + batch_size]
        batch_emb = model.encode(batch_texts, convert_to_tensor=True, device=device).cpu()
        all_embeds.append(batch_emb)
    all_embeds = torch.cat(all_embeds, dim=0)  # [num_texts, dim]

    # 3) Scatter embeddings back to each entity
    entity_attr_embeds = {eid: [] for eid in entity_ids}
    for idx, eid in enumerate(owners):
        entity_attr_embeds[eid].append(all_embeds[idx])

    return entity_attr_embeds

# Function to update the attribute embedding matrix with the calculated embeddings
def update_attr_embedding_matrix(attr_embed_matrix, entity_attr_embeds, entity_id_to_idx):
    """
    Updates the attribute embedding matrix by averaging the embeddings of an entity's attributes.
    If the entity's embedding is all zeros, it gets updated with the mean of its attribute embeddings.
    """
    for eid, embeds in entity_attr_embeds.items():
        if len(embeds) == 0:
            continue
        mean_embed = torch.stack(embeds).mean(dim=0)
        idx = entity_id_to_idx[eid]
        if torch.all(attr_embed_matrix[idx] == 0):  # Update only if it's a zero embedding
            attr_embed_matrix[idx] = mean_embed
    return attr_embed_matrix

# Function to calculate the cosine similarity between two matrices
def cosine_similarity_matrix(mat1, mat2):
    """
    Computes the cosine similarity matrix between two matrices.
    Each row in the matrices represents a vector, and the cosine similarity is computed between each pair of vectors.
    """
    mat1_norm = F.normalize(mat1, p=2, dim=1)
    mat2_norm = F.normalize(mat2, p=2, dim=1)
    return torch.mm(mat1_norm, mat2_norm.t())

# Function to build neighbors from the triples (adjacency list)
def build_neighbors(triples, entity_id_to_idx):
    """
    Build an adjacency list (neighbors) from the triples.
    Each entity will have a list of its neighbors (other entities it is connected to).
    """
    neighbors = {eid: [] for eid in entity_id_to_idx.keys()}
    for h, r, t in triples:
        if h in neighbors and t in entity_id_to_idx:
            neighbors[h].append(t)
        if t in neighbors and h in entity_id_to_idx:
            neighbors[t].append(h)
    return neighbors

# Function to update embeddings using neighbors' embeddings
def update_with_neighbor_embeds(attr_embed_matrix, neighbors, entity_id_to_idx):
    """
    Updates the attribute embedding matrix by averaging the embeddings of an entity's neighbors.
    Only entities with zero embeddings will be updated.
    """
    for eid, idx in entity_id_to_idx.items():
        if torch.all(attr_embed_matrix[idx] == 0):
            neighs = neighbors.get(eid, [])
            if len(neighs) == 0:
                continue
            neigh_embeds = [attr_embed_matrix[entity_id_to_idx[n]] for n in neighs if n in entity_id_to_idx]
            if len(neigh_embeds) == 0:
                continue
            mean_neigh_embed = torch.stack(neigh_embeds).mean(dim=0)
            attr_embed_matrix[idx] = mean_neigh_embed
    return attr_embed_matrix

# === Main Process ===

# File paths for entities, attributes, and triples
entity_id_file_1 = os.path.join(data_root, args.dataset, "ent_ids_1")
entity_id_file_2 = os.path.join(data_root, args.dataset, "ent_ids_2")
attr_file_1 = os.path.join(data_root, args.dataset, "training_attrs_1")
attr_file_2 = os.path.join(data_root, args.dataset, "training_attrs_2")
triple_file_1 = os.path.join(data_root, args.dataset, "triples_1")
triple_file_2 = os.path.join(data_root, args.dataset, "triples_2")

# Load entity IDs and names
entity_ids_1, entity_names_1 = load_entity_ids(entity_id_file_1)
entity_ids_2, entity_names_2 = load_entity_ids(entity_id_file_2)

# Load attribute name dictionaries for both datasets
attr_name_dict_1 = load_attr_dict_with_name_key(attr_file_1)
attr_name_dict_2 = load_attr_dict_with_name_key(attr_file_2)

# Build attribute dictionaries for both datasets
attr_dict_1 = build_entity_attr_dict(entity_ids_1, entity_names_1, attr_name_dict_1)
attr_dict_2 = build_entity_attr_dict(entity_ids_2, entity_names_2, attr_name_dict_2)

# Load triples for both datasets
triples_1 = load_triples(triple_file_1)
triples_2 = load_triples(triple_file_2)

# Mapping entity IDs to indexes for both datasets
entity_id_to_idx_1 = {eid: i for i, eid in enumerate(entity_ids_1)}
entity_id_to_idx_2 = {eid: i for i, eid in enumerate(entity_ids_2)}

# Initialize embedding matrices for attributes
attr_embed_1 = init_attr_embedding_matrix(len(entity_ids_1), 1024)
attr_embed_2 = init_attr_embedding_matrix(len(entity_ids_2), 1024)

# Build attribute embeddings for entities (batched)
entity_attr_embeds_1 = build_entity_attr_embedding(entity_ids_1, attr_dict_1)
entity_attr_embeds_2 = build_entity_attr_embedding(entity_ids_2, attr_dict_2)

# Update the attribute embedding matrices with computed embeddings
attr_embed_1 = update_attr_embedding_matrix(attr_embed_1, entity_attr_embeds_1, entity_id_to_idx_1)
attr_embed_2 = update_attr_embedding_matrix(attr_embed_2, entity_attr_embeds_2, entity_id_to_idx_2)

# Calculate cosine similarity between the attribute embedding matrices
attr_score_matrix = cosine_similarity_matrix(attr_embed_1, attr_embed_2)

# Combine the attribute embeddings from both datasets
attr_embed_combined = torch.cat([attr_embed_1, attr_embed_2], dim=0)

# Combine entity IDs from both datasets
entity_ids_combined = entity_ids_1 + entity_ids_2
entity_id_to_idx_combined = {eid: i for i, eid in enumerate(entity_ids_combined)}

# Build neighbors for both datasets and combine them
neighbors_1 = build_neighbors(triples_1, entity_id_to_idx_1)
neighbors_2 = build_neighbors(triples_2, entity_id_to_idx_2)
neighbors_combined = {}
neighbors_combined.update(neighbors_1)
neighbors_combined.update(neighbors_2)

# Update attribute embeddings with neighbors' embeddings
attr_embed_combined = update_with_neighbor_embeds(attr_embed_combined, neighbors_combined, entity_id_to_idx_combined)

# Save the final attribute embeddings and similarity matrix
np.save(os.path.join(data_root, args.dataset, 'Emb', 'attr_embedding.npy'), attr_embed_combined.numpy())
np.save(os.path.join(data_root, args.dataset, 'Score Matrix', 'Attr.npy'), attr_score_matrix.numpy())

# Print completion message and shapes of saved matrices
print("Saving complete.")
print("Combined attribute embedding matrix shape:", attr_embed_combined.shape)
print("Attribute similarity matrix shape:", attr_score_matrix.shape)
