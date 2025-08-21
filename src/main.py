import os
import time
import argparse
import logging
import numpy as np
import torch
from utils import *
from config import *
from data_loader import KGs
from model_add_all_double_1 import Encoder_Model
from seed_iterate import get_pair, SeedIterator
from sinkhorn import matrix_sinkhorn
import sparse_eval


# ======================================================================================
# IO: Feature & Side-Modality Loading
# ======================================================================================

def load_features(directory: str):
    """
    Load node features from {directory}/Emb/*.npy
    Expected files:
      - key_embedding.npy
      - value_embedding.npy
      - vis_embedding.npy
    Returns:
      (key_feature, value_feature, vis_feature) as np.ndarray or None
    """
    key_feature = value_feature = vis_feature = None
    feature_path = os.path.join(directory, 'Emb')
    if not os.path.isdir(feature_path):
        print(f"[load_features] Directory not found: {feature_path}")
        return key_feature, value_feature, vis_feature

    name_map = {
        'key_embedding.npy': 'key_feature',
        'value_embedding.npy': 'value_feature',
        'vis_embedding.npy': 'vis_feature',
    }

    for filename in os.listdir(feature_path):
        if not filename.endswith('.npy'):
            continue
        file_path = os.path.join(feature_path, filename)
        arr = np.load(file_path)
        if filename == 'key_embedding.npy':
            key_feature = arr
        elif filename == 'value_embedding.npy':
            value_feature = arr
        elif filename == 'vis_embedding.npy':
            vis_feature = arr
        print(f"[load_features] Loaded {filename} -> {name_map.get(filename, 'unknown')}")

    return key_feature, value_feature, vis_feature


def load_side_modalities(directory: str):
    """
    Load precomputed similarity matrices from {directory}/Score Matrix/*.npy
    Returns:
      side_modalities: dict[name -> np.ndarray]
      total_sum: np.ndarray sum of all modalities (same shape)
    """
    side_modalities = {}
    matrix_path = os.path.join(directory, 'Score Matrix')
    if not os.path.isdir(matrix_path):
        print(f"[load_side_modalities] Directory not found: {matrix_path}")
        return side_modalities, None

    for filename in os.listdir(matrix_path):
        if filename.endswith('.npy'):
            base = os.path.splitext(filename)[0]
            side_modalities[base] = np.load(os.path.join(matrix_path, filename))

    print(f"[load_side_modalities] Found {len(side_modalities)} side modalities.")
    if side_modalities:
        print(f"[load_side_modalities] Keys: {list(side_modalities.keys())}")

    # Sum all arrays (vectorized)
    total_sum = None
    for arr in side_modalities.values():
        total_sum = arr if total_sum is None else (total_sum + arr)
    return side_modalities, total_sum


# ======================================================================================
# Evaluation (vectorized)
# ======================================================================================

def dual_evaluate(
    deal_pairs: np.ndarray,
    model: Encoder_Model,
    entity1_ids: np.ndarray,
    entity2_ids: np.ndarray,
    side_modalities_sum: np.ndarray,
):
    """
    Evaluate with model embeddings + side-modality boost.
    Vectorized construction of the side-modality block removes O(n^2) Python loops.

    Args:
      deal_pairs: np.ndarray of shape [K, 2]
      model, entity arrays, side_modalities_sum: as prepared in main

    Returns:
      scores_full_sinkhorn: torch.Tensor  (for full entity grid)
      stan: evaluation metric from sparse_eval
    """
    # Model embeddings for current evaluation pairs
    Lvec, Rvec, _ = model.get_embeddings(deal_pairs[:, 0], deal_pairs[:, 1])
    # Model embeddings for the entire left-right entity sets
    Lvec_all, Rvec_all, _ = model.get_embeddings(entity1_ids, entity2_ids)

    # Base scores (dot product)
    scores = torch.matmul(Lvec, Rvec.T)              
    scores_full = torch.matmul(Lvec_all, Rvec_all.T) 
    
    # ---------------- Vectorized side-modality addition ----------------
    # For the KxK block, select corresponding rows/cols from side_modalities_sum
    left_idx = deal_pairs[:, 0]
    right_idx = deal_pairs[:, 1] - len(entity1_ids)
    # Use numpy.ix_ to build the KxK submatrix, then convert to torch
    side_block = torch.from_numpy(side_modalities_sum[np.ix_(left_idx, right_idx)]).float()

    scores = scores + side_block
    scores_full = scores_full + torch.from_numpy(side_modalities_sum).float()

    # Balanced matching via Sinkhorn on (1 - sim)
    scores_sinkhorn = matrix_sinkhorn(1 - scores)
    scores_full_sinkhorn = matrix_sinkhorn(1 - scores_full)

    # Metric on the KxK block — identity links
    links = torch.stack([torch.arange(len(deal_pairs)), torch.arange(len(deal_pairs))], dim=0)
    _, stan = sparse_eval.evaluate_sim_matrix(link=links, sim_x2y=scores_sinkhorn, no_csls=True)
    return scores_full_sinkhorn, stan


# ======================================================================================
# Training
# ======================================================================================

def train_base(
    args,
    train_pairs: np.ndarray,
    valid_pairs: np.ndarray,
    model: Encoder_Model,
    entity1_ids: np.ndarray,
    entity2_ids: np.ndarray,
    ill_ent,
    side_modalities_sum: np.ndarray,
    save_path: str,
):
    """
    Train with periodic validation, early stopping, and iterative seed augmentation.
    """
    flag = 1
    total_train_time = 0.0
    best_stan = -np.inf
    no_improve_count = 0
    stop_step = args.stop_step

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        t0 = time.time()
        total_loss = 0.0

        model.train()
        # mini-batches
        num = len(train_pairs)
        if num == 0:
            print("[train] WARNING: empty train_pairs, skipping epoch forward pass.")
        else:
            num_batches = (num + args.batch_size - 1) // args.batch_size
            for b in range(num_batches):
                batch_np = train_pairs[b * args.batch_size:(b + 1) * args.batch_size]
                if len(batch_np) == 0:
                    continue
                pairs = torch.from_numpy(batch_np).to(model.device)
                optimizer.zero_grad()
                loss = model(pairs, flag)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

        t1 = time.time()
        total_train_time += (t1 - t0)
        print(f"[epoch {epoch + 1}/{args.epoch}] loss: {total_loss:.5f} | time: {(t1 - t0):.3f}s")
        flag = 1

        # ---------------- Validation & Early Stopping (every 6 epochs after warmup) -----------
        if (epoch + 1) % 6 == 0 and epoch > 0:
            logging.info("--------- Validation ---------")
            print("---------------- valid result ----------------")
            model.eval()
            with torch.no_grad():
                scores_full, stan = dual_evaluate(
                    valid_pairs, model, entity1_ids, entity2_ids, side_modalities_sum
                )

                if stan > best_stan:
                    best_stan = stan
                    no_improve_count = 0
                    torch.save(model.state_dict(), save_path)
                    print(f"[valid] New best model saved (stan={stan:.4f})")
                else:
                    no_improve_count += 1
                    print(f"[valid] No improvement for {no_improve_count}/{stop_step}")

                if no_improve_count >= stop_step:
                    print(f"[early stop] Triggered at patience={no_improve_count}")
                    break

            # ---------------- Iterative seed augmentation ----------------
            train_pairs = get_pair(
                train_pairs, scores_full, entity1_ids, entity2_ids, ill_ent, epoch
            )
            print(f"[train] Total train pairs: {len(train_pairs)}")

    print(f"[train] Finished. Total training time: {total_train_time:.2f}s")

# ======================================================================================
# Main
# ======================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment model')
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--dataset', type=str, default='DB15K-FB15K',
                        choices=['DB15K-FB15K', 'YAGO15K-FB15K'])
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

    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--gamma', default=1.0, type=float)

    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--dev_interval', default=3, type=int)
    parser.add_argument('--stop_step', default=3, type=int)
    parser.add_argument('--sim_threshold', default=0.0, type=float)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--M', default=500, type=int)

    args = parser.parse_args()
    device = set_device(args)

    start_time = time.time()

    # ----------------------- Load KG data ------------------------------------------------
    kgs = KGs()
    
    (train_pair, valid_pair, test_pair,
     ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,
     entity1, entity2, ill_ent, triples1, triples2) = kgs.load_data(data_root, args)

    # Convert arrays to torch tensors (transpose per original)
    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)

    base_directory = os.path.join(data_root, args.dataset)

    # Embeddings and side modalities
    key, value, vis = load_features(base_directory)
    side_modalities, side_modalities_sum = load_side_modalities(base_directory)

    # If no initial seeds, derive from side modalities via Sinkhorn
    if len(train_pair) == 0:
        scores1 = torch.tensor(side_modalities_sum, dtype=torch.float32)
        scores1 = matrix_sinkhorn(1 - scores1)
        train_pair = get_pair(train_pair, scores1, entity1, entity2, ill_ent, epoch=0)

    # Save path (relative to this .py location)
    save_path = os.path.join(current_dir, 'model_params.pth')

    print(f"[load] Dataset init time: {time.time() - start_time:.2f}s")

    # ----------------------- Build Model -------------------------------------------------
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
        key_emb=key,
        value_emb=value,
        vis_emb=vis,
        dropout_rate=args.dropout_rate,
        gamma=args.gamma,
        lr=args.lr,
        depth=args.depth
    ).to(device)

    print("[info] Train Pair Shape:", getattr(train_pair, 'shape', None))

    if 'base' in args.batch:
        train_base(
            args=args,
            train_pairs=train_pair,
            valid_pairs=valid_pair,
            model=model,
            entity1_ids=entity1,
            entity2_ids=entity2,
            ill_ent=ill_ent,
            side_modalities_sum=side_modalities_sum,
            save_path=save_path,
        )

    # ------------------------ Final Test -------------------------------------------------
    logging.info("--------- test ---------")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print("---------------- test result ----------------")
    with torch.no_grad():
        _scores_full, _stan = dual_evaluate(
            test_pair, model, entity1, entity2, side_modalities_sum
        )
