import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import *
from config import *
from data_loader import KGs
from model_nv import Encoder_Model
from seed_iterate import get_pair, SeedIterator
from sinkhorn import matrix_sinkhorn
import sparse_eval


# =============================================================================
# Feature loaders
# =============================================================================

def load_features_novalue(directory: str):
    """
    Load attribute and visual embeddings from {directory}/Emb/*.npy.
    Expected filenames:
      - attr_embedding.npy
      - vis_embedding.npy
    Returns:
      (attr_feature, vis_feature) as numpy arrays or None.
    """
    attr_feature = None
    vis_feature = None
    feature_path = os.path.join(directory, 'Emb')
    for filename in os.listdir(feature_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(feature_path, filename)
            if filename == 'attr_embedding.npy':
                attr_feature = np.load(file_path)
            elif filename == 'vis_embedding.npy':
                vis_feature = np.load(file_path)
            print(f"Loaded {filename} into {filename.split('_')[0]}_feature")
    return attr_feature, vis_feature


def load_side_modalities(directory: str):
    """
    Load similarity matrices from {directory}/Score Matrix/*.npy,
    normalize each, apply Sinkhorn (as distance = 1 - sim), and sum.
    Returns:
      side_modalities: dict[str -> np.ndarray]
      total_sum      : np.ndarray (sum of Sinkhorn-adjusted matrices)
      total_sum_1    : np.ndarray (sum of min-max normalized matrices)
    """
    side_modalities = {}
    matrix_path = os.path.join(directory, 'Score Matrix')

    for filename in os.listdir(matrix_path):
        if filename.endswith('.npy'):
            base_filename = filename.split('.')[0]
            side_modalities[base_filename] = np.load(os.path.join(matrix_path, filename))

    print(f'There are {len(side_modalities)} side modalities.')
    print(f'They are: {list(side_modalities.keys())}')

    total_sum = None
    total_sum_1 = None
    for _, array in side_modalities.items():
        # Min-max normalization in numpy
        array_1 = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)

        # Sinkhorn on (1 - sim) in torch
        t = torch.tensor(array_1, dtype=torch.float32, device=device)
        t = matrix_sinkhorn(1 - t).to(device)
        array_sink = t.detach().cpu().numpy()

        total_sum_1 = array_1 if total_sum_1 is None else (total_sum_1 + array_1)
        total_sum = array_sink if total_sum is None else (total_sum + array_sink)

    return side_modalities, total_sum, total_sum_1


# =============================================================================
# Utilities
# =============================================================================

def get_side_embedding(test_pair, key, value, vis):
    """
    Build per-pair side embeddings (not used in main flow; kept for compatibility).
    Returns:
      (side_embeddings_L, side_embeddings_R) as lists of torch tensors.
    """
    side_embeddings_L, side_embeddings_R = [], []
    for pair in test_pair:
        key_emb_0 = torch.from_numpy(key[pair[0]])
        value_emb_0 = torch.from_numpy(value[pair[0]])
        vis_emb_0 = torch.from_numpy(vis[pair[0]])

        key_emb_1 = torch.from_numpy(key[pair[1]])
        value_emb_1 = torch.from_numpy(value[pair[1]])
        vis_emb_1 = torch.from_numpy(vis[pair[1]])

        _ = torch.cat((key_emb_0, value_emb_0, vis_emb_0), dim=0)  # kept (unused)
        _ = torch.cat((key_emb_1, value_emb_1, vis_emb_1), dim=0)  # kept (unused)

        side_embeddings_L.append(key_emb_0)
        side_embeddings_R.append(key_emb_1)
    return side_embeddings_L, side_embeddings_R


def create_entity_to_scores_mapping(entity1, entity2):
    """
    Map entity IDs to row/column indices in the side-modality matrix.
    Returns:
      dicts for KG1 and KG2 entities.
    """
    entity_to_scores_1 = {eid: idx for idx, eid in enumerate(entity1)}
    entity_to_scores_2 = {eid: idx for idx, eid in enumerate(entity2)}
    return entity_to_scores_1, entity_to_scores_2


# =============================================================================
# Evaluation (tensor-only & vectorized)
# =============================================================================

def dual_evaluate(deal_pairs: np.ndarray):
    """
    Evaluate with model embeddings + side-modality boosts.
    - Keeps everything in torch
    - Side-modality KxK block is vectorized (no Python double-loop)
    Returns:
      (scores_full, stan)
      scores_full: torch.Tensor for full |E1| x |E2| grid after Sinkhorn + side mods
      stan       : eval metric from sparse_eval
    """
    # Embeddings for current eval pairs and full grids
    Lvec, Rvec, _ = model.get_embeddings(deal_pairs[:, 0], deal_pairs[:, 1])
    Lvec_all, Rvec_all, _ = model.get_embeddings(entity1, entity2)

    # Base similarities
    scores = torch.matmul(Lvec, Rvec.T)
    scores1 = torch.matmul(Lvec_all, Rvec_all.T)

    # Min-max normalize per matrix
    def _minmax_norm(x: torch.Tensor):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    scores = _minmax_norm(scores)
    scores1 = _minmax_norm(scores1)

    # Sinkhorn over distances
    scores = matrix_sinkhorn(1 - scores)
    scores1 = matrix_sinkhorn(1 - scores1)

    # Vectorized side-modality block for KxK submatrix
    left_rows = [entity_s1[int(e)] for e in deal_pairs[:, 0].tolist()]
    right_cols = [entity_s2[int(e)] for e in deal_pairs[:, 1].tolist()]
    side_block_np = side_modalities_sum[np.ix_(left_rows, right_cols)]
    side_block = torch.tensor(side_block_np, dtype=scores.dtype, device=scores.device)

    # Add side-modality signals
    scores = scores + side_block
    scores1 = scores1 + torch.tensor(side_modalities_sum1, dtype=scores1.dtype, device=scores1.device)

    # Normalize scores
    scores_min = scores.min()
    scores_max = scores.max()
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    # Normalize scores1
    scores1_min = scores1.min()
    scores1_max = scores1.max()
    scores1 = (scores1 - scores1_min) / (scores1_max - scores1_min + 1e-8)

    # Evaluate on KxK block (identity links)
    links = torch.stack([torch.arange(len(deal_pairs)), torch.arange(len(deal_pairs))], dim=0)
    _, stan = sparse_eval.evaluate_sim_matrix(link=links, sim_x2y=scores, no_csls=True)
    return scores1, stan


# =============================================================================
# Training loop
# =============================================================================

def train_base(args, train_pairs, model: Encoder_Model, entity1, ill_ent):
    """
    Training with periodic validation, early stopping, and iterative seed augmentation.
    """
    flag = 1
    total_train_time = 0.0
    best_stan = -np.inf
    no_improve_count = 0
    stop_step = args.stop_step

    for epoch in range(args.epoch):
        t0 = time.time()
        total_loss = 0.0
        np.random.shuffle(train_pairs)
        batch_num = len(train_pairs) // args.batch_size + 1

        model.train()
        for b in range(batch_num):
            pairs = train_pairs[b * args.batch_size:(b + 1) * args.batch_size]
            if len(pairs) == 0:
                continue
            pairs = torch.from_numpy(pairs).to(device)
            optimizer.zero_grad()
            loss = model(pairs, flag)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        t1 = time.time()
        total_train_time += (t1 - t0)
        print(f'[epoch {epoch + 1}/{args.epoch}] loss: {total_loss:.5f}, time: {(t1 - t0):.3f}s')
        flag = 1

        # ------ validation every 6 epochs after warmup ------
        if (epoch + 1) % 6 == 0 and epoch > 0:
            logging.info("---------Validation---------")
            model.eval()
            print("---------------- valid result ----------------")
            with torch.no_grad():
                scores1, stan = dual_evaluate(valid_pair)

                if stan > best_stan:
                    best_stan = stan
                    no_improve_count = 0
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved (stan={stan:.4f})")
                else:
                    no_improve_count += 1
                    print(f"No improvement for {no_improve_count}/{stop_step} steps")

                if no_improve_count >= stop_step:
                    print(f"Early stopping triggered at step {no_improve_count}!")
                    break

            # Update seed pairs
            train_pairs = get_pair(train_pairs, scores1, entity1, entity2, ill_ent, epoch)
            print(f" all {len(train_pairs)} pairs.")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment model')
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--dataset', type=str, default='OEA_D_W_15K_V2',
                        choices=['OEA_D_W_15K_V1', 'OEA_D_W_15K_V2', 'OEA_EN_DE_15K_V1', 'OEA_EN_FR_15K_V1',
                                 'fr_en', 'ja_en', 'zh_en'])
    parser.add_argument('--batch', default='base', type=str)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--ent_hidden', default=128, type=int)
    parser.add_argument('--rel_hidden', default=128, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--ind_dropout_rate', default=0.3, type=float)

    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--gamma', default=1, type=float)

    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--dev_interval', default=2, type=int)
    parser.add_argument('--stop_step', default=3, type=int)
    parser.add_argument('--sim_threshold', default=0.0, type=float)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--M', default=500, type=int)

    args = parser.parse_args()
    device = set_device(args)

    start_time = time.time()

    # ----------------------- load dataset -----------------------
    kgs = KGs()

    (train_pair, valid_pair, test_pair,
     ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,
     entity1, entity2, ill_ent, triples1, triples2) = kgs.load_data(data_root, args)

    # to torch (transpose per original)
    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)

    base_directory = os.path.join(data_root, args.dataset)

    # ----------------------- load embeddings & side mods -----------------------
    att, vis = load_features_novalue(base_directory)
    side_modalities, side_modalities_sum, side_modalities_sum1 = load_side_modalities(base_directory)

    # build entity -> (row/col) mappings for side modalities
    entity_s1, entity_s2 = create_entity_to_scores_mapping(entity1, entity2)

    # if no initial seeds, derive from side modalities
    if len(train_pair) == 0:
        scores1 = torch.tensor(side_modalities_sum, dtype=torch.float32, device=device)
        scores1 = matrix_sinkhorn(1 - scores1)
        train_pair = get_pair(train_pair, scores1, entity1, entity2, ill_ent, epoch=0)

    # save path (relative to this .py)
    save_path = os.path.join(current_dir, 'model_params.pth')

    print("Dataset loaded in:", time.time() - start_time, "s")

    # ----------------------------- build model -----------------------------
    model = Encoder_Model(
        node_hidden=args.ent_hidden,
        rel_hidden=args.rel_hidden,
        node_size=kgs.old_ent_num,
        rel_size=kgs.total_rel_num,
        triple_size=kgs.triple_num,
        device=device,
        adj_matrix=ent_adj,
        r_index=r_index,
        r_val=r_val,
        rel_matrix=ent_rel_adj,
        ent_matrix=ent_adj_with_loop,
        ill_ent=ill_ent,
        att_emb=att,
        vis_emb=vis,
        dropout_rate=args.dropout_rate,
        gamma=args.gamma,
        lr=args.lr,
        depth=args.depth
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    print("Train Pair Shape:", train_pair.shape)

    if 'base' in args.batch:
        train_base(args, train_pair, model, entity1, ill_ent)

    # ----------------------------- final test -----------------------------
    logging.info("---------test---------")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("---------------- test result ----------------")
    with torch.no_grad():
        _1, _2 = dual_evaluate(test_pair)
