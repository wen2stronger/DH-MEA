import argparse
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from sinkhorn import matrix_sinkhorn
import sparse_eval
from os.path import join as pjoin
import re
from config import *

parser = argparse.ArgumentParser(description='MSP process (Attr) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])

args = parser.parse_args()

source_dataset, target_dataset = args.dataset.split('-')



ent2id = {}
id2ent = {}
source2id = {}
id2source = {}
target2id = {}
id2target = {}

id2attrs = []

source_keys = set()
target_keys = set()

with open(pjoin(data_root, args.dataset, 'ent_ids_1'), 'r', encoding='UTF-8') as f:
    source_ids = []
    for line in f:
        line = line.strip()
        ent_id, ent_name = line.split('\t')
        source_ids.append(ent_id)
        ent2id[ent_name] = int(ent_id)
        id2ent[int(ent_id)] = ent_name
        source2id[ent_name] = int(ent_id)
        id2source[int(ent_id)] = ent_name

with open(pjoin(data_root, args.dataset, 'ent_ids_2'), 'r', encoding='UTF-8') as f:
    target_ids = []
    for line in f:
        line = line.strip()
        ent_id, ent_name = line.split('\t')
        target_ids.append(ent_id)
        ent2id[ent_name] = int(ent_id)
        id2ent[int(ent_id)] = ent_name
        target2id[ent_name] = int(ent_id) - len(source2id)
        id2target[int(ent_id) - len(source2id)] = ent_name


with open(pjoin(data_root, args.dataset, 'attr'), 'r', encoding='UTF-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        tmp = []
        if i < len(source2id):
            attr_value_schemas = line.split('^^^')
            for attr_value_schema in attr_value_schemas:
                if attr_value_schema == '':
                    continue
                attr, value, schema = attr_value_schema.split('|||')
                tmp.append((attr, value, schema))
            id2attrs.append(tmp)
        else:
            attr_values = line.split('^^^')
            for attr_value in attr_values:
                if attr_value == '':
                    continue
                attr, value = attr_value.split('|||')
                tmp.append((attr, value))
            id2attrs.append(tmp)

assert len(source2id) + len(target2id) == len(ent2id)

source_attr_value_set = set()
target_attr_value_set = set()

def date2float(date):
    if re.match(r'\d+-\d+-\d+', date):
        year = date.split('-')[0]
        mouth = date.split('-')[1]
        decimal_right = '0' if mouth == '12' else str(int(mouth) / 12)[2:]
        if mouth == '12':
            year = str(int(year) + 1)
        return year + '.' + decimal_right
    else:
        return date


for i, attrs in enumerate(id2attrs):
    if i < len(source2id):
        for attr, value, schema in attrs:
            value = date2float(value)
            source_attr_value_set.add(attr + ' ' + value)
            source_keys.add(attr)
    else:
        for attr, value in attrs:
            value = date2float(value)
            target_attr_value_set.add(attr + ' ' + value)
            target_keys.add(attr)

source_attr_value_2_id = {}
target_attr_value_2_id = {}
source_id2_attr_value = {}
target_id2_attr_value = {}


source2attr = np.zeros((len(source2id), len(source_attr_value_set)), dtype=np.float32)
target2attr = np.zeros((len(target2id), len(target_attr_value_set)), dtype=np.float32)

source_attr_value_list = sorted(list(source_attr_value_set))
target_attr_value_list = sorted(list(target_attr_value_set))

for i, attrs in enumerate(id2attrs):
    if i < len(source2id):
        for attr, value, schema in attrs:
            value = date2float(value)
            pos = source_attr_value_list.index(attr + ' ' + value)
            source2attr[i][pos] = 1
    else:
        for attr, value in attrs:
            value = date2float(value)
            pos = target_attr_value_list.index(attr + ' ' + value)
            target2attr[i - len(source2id)][pos] = 1


source_keyValue_sents = sorted(list(source_attr_value_set))
target_keyValue_sents = sorted(list(target_attr_value_set))


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = SentenceTransformer(model_att_root).to(device)



source_key_embeddings = []
target_key_embeddings = []
source_value_embeddings = []
target_value_embeddings = []

source_value = []
target_value = []
source_value_1 = []
target_value_1 = []

batch_size = 128
for i in tqdm(range(0, len(source_keyValue_sents), batch_size)):
    key_sents = source_keyValue_sents[i:i + batch_size]
    for j in range(len(key_sents)):
        try:
            source_value.append(float(key_sents[j].split(' ')[1]))
            source_value_1.append(str(key_sents[j].split(' ')[1]))
        except:
            source_value.append(0)
            source_value_1.append('0')
        key_sents[j] = key_sents[j].split(' ')[0]
    source_key_embeddings.append(model.encode(key_sents))
source_value_embeddings.append(model.encode(source_value_1))
source_value_embeddings = np.concatenate(source_value_embeddings, axis=0)
source_key_embeddings = np.concatenate(source_key_embeddings, axis=0)

for i in tqdm(range(0, len(target_keyValue_sents), batch_size)):
    key_sents = target_keyValue_sents[i:i + batch_size]
    for j in range(len(key_sents)):
        try:
            target_value.append(float(key_sents[j].split(' ')[1]))
            target_value_1.append(str(key_sents[j].split(' ')[1]))
        except:
            target_value.append(0)
            target_value_1.append('0')
        key_sents[j] = key_sents[j].split(' ')[0]
    target_key_embeddings.append(model.encode(target_keyValue_sents[i:i + batch_size]))
target_value_embeddings.append(model.encode(target_value_1))
target_value_embeddings = np.concatenate(target_value_embeddings, axis=0)
target_key_embeddings = np.concatenate(target_key_embeddings, axis=0)

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

# Update embeddings for entities whose embeddings are all zeros
def update_embeddings(ent_ids, adj_matrix, embeddings):
    """Update embeddings for entities with zero embeddings based on their neighbors."""
    for ent_id in ent_ids:
        ent_id_ = int(ent_id)
        # Update embeddings for source entities
        if ent_id_ < len(source_ids) and np.all(embeddings[ent_id_] == 0):
            if ent_id_ in adj_matrix:
                valid_neighbors = [embeddings[nbr] for nbr in adj_matrix[ent_id_] if np.any(embeddings[nbr] != 0)]
                if valid_neighbors:
                    embeddings[ent_id_] = np.mean(valid_neighbors, axis=0)
        # Update embeddings for target entities
        elif ent_id_ >= len(source_ids) and np.all(embeddings[ent_id_ - len(source_ids)] == 0):
            if ent_id_ in adj_matrix:
                valid_neighbors = [embeddings[nbr - len(source_ids)] for nbr in adj_matrix[ent_id_] if np.any(embeddings[nbr - len(source_ids)] != 0)]
                if valid_neighbors:
                    embeddings[ent_id_ - len(source_ids)] = np.mean(valid_neighbors, axis=0)
    return embeddings

source_key_emb = source2attr @ source_key_embeddings
source_value_emb = source2attr @ source_value_embeddings


target_key_emb = target2attr @ target_key_embeddings
target_value_emb = target2attr @ target_value_embeddings

# Update embeddings for entities with zero embeddings
source_key_final = update_embeddings(source_ids, adj_matrix1, source_key_emb)
source_value_final = update_embeddings(source_ids, adj_matrix1, source_value_emb)
target_key_final = update_embeddings(target_ids, adj_matrix2, target_key_emb)
target_value_final = update_embeddings(target_ids, adj_matrix2, target_value_emb)

# Stack the final embeddings of source and target entities

key_embedding = np.vstack([source_key_final, target_key_final])
value_embedding = np.vstack([source_value_final, target_value_final])

# save embedding
np.save(pjoin(data_root, args.dataset, 'Emb','key_embedding'), key_embedding )
np.save(pjoin(data_root, args.dataset, 'Emb','value_embedding'), value_embedding )

# get score
source_value = np.array(source_value)[:, np.newaxis]
target_value = np.array(target_value)[np.newaxis, :]
scores_key = np.matmul(source_key_embeddings, target_key_embeddings.T)
scores_value = 1 / (np.abs(source_value - target_value) + 1e-3)

attr2attr = scores_key * scores_value

source2target = source2attr @ attr2attr @ target2attr.T


source2target = (source2target - source2target.min()) / (source2target.max() - source2target.min())

# save the scores as .npy file
np.save(pjoin(data_root, args.dataset, 'Score Matrix','Attr.npy'), source2target)

