# =============================================================================
# Encoder_Model (tidied: lean English comments + light safety guard)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_all_GraphAttention1_v2


class Encoder_Model(nn.Module):
    def __init__(
        self, node_hidden, rel_hidden, triple_size, node_size, rel_size, device,
        adj_matrix, r_index, r_val, rel_matrix, ent_matrix, ill_ent,
        att_emb, vis_emb, dropout_rate=0.3, gamma=3, lr=0.005, depth=2
    ):
        super(Encoder_Model, self).__init__()

        # ---------------------------- basic config ----------------------------
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        # graph structures / indices
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.ill_ent = ill_ent

        # ------------------------ trainable embeddings ------------------------
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        # ------------------- side embeddings -> 128 dims ----------------------
        d1 = len(att_emb[0])
        d2 = len(vis_emb[0])
        att_linear = nn.Linear(d1, 128)
        vis_linear = nn.Linear(d2, 128)

        # Optional extra linear (kept for compatibility)
        self.only_linear = nn.Linear(128, 128)

        # numpy -> torch
        att_emb = torch.from_numpy(att_emb).float()
        vis_emb = torch.from_numpy(vis_emb).float()

        # project to 128-d and make them trainable
        att_init = att_linear(att_emb)
        vis_init = vis_linear(vis_emb)
        self.att_embedding = nn.Embedding.from_pretrained(att_init, freeze=False)
        self.vis_embedding = nn.Embedding.from_pretrained(vis_init, freeze=False)
        torch.nn.init.xavier_uniform_(self.att_embedding.weight)
        torch.nn.init.xavier_uniform_(self.vis_embedding.weight)

        # ------------------------------ encoders ------------------------------
        self.e_encoder = NR_all_GraphAttention1_v2(
            node_size=self.node_size, rel_size=self.rel_size, triple_size=self.triple_size,
            node_dim=self.node_hidden, depth=self.depth, use_bias=True
        )
        self.r_encoder = NR_all_GraphAttention1_v2(
            node_size=self.node_size, rel_size=self.rel_size, triple_size=self.triple_size,
            node_dim=self.node_hidden, depth=self.depth, use_bias=True
        )
        self.a_encoder = NR_all_GraphAttention1_v2(
            node_size=self.node_size, rel_size=self.rel_size, triple_size=self.triple_size,
            node_dim=self.node_hidden, depth=self.depth, use_bias=True
        )
        self.v_encoder = NR_all_GraphAttention1_v2(
            node_size=self.node_size, rel_size=self.rel_size, triple_size=self.triple_size,
            node_dim=self.node_hidden, depth=self.depth, use_bias=True
        )

    # -----------------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------------
    def avg(self, adj, emb, size: int):
        """Row-normalized sparse aggregation via softmax(adj) @ emb."""
        adj = torch.sparse_coo_tensor(
            indices=adj,
            values=torch.ones_like(adj[0, :], dtype=torch.float),
            size=[self.node_size, size]
        )
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def only_forward(self, emb):
        """Kept for compatibility (simple extra linear)."""
        return self.only_linear(emb)

    # -----------------------------------------------------------------------------
    # GNN forward
    # -----------------------------------------------------------------------------
    def gcn_forward(self):
        # aggregate base embeddings
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        att_feature = self.avg(self.ent_adj, self.att_embedding.weight, self.node_size)
        vis_feature = self.avg(self.ent_adj, self.vis_embedding.weight, self.node_size)

        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]

        # two-branch encoding per modality
        ent_feature1, ent_feature2   = self.e_encoder([ent_feature] + opt)
        rel_feature1, rel_feature2   = self.r_encoder([rel_feature] + opt)
        att_feature1, att_feature2   = self.a_encoder([att_feature] + opt)
        vis_feature1, vis_feature2   = self.v_encoder([vis_feature] + opt)

        # concat for each branch
        out_feature_1 = torch.cat(
            [ent_feature1, rel_feature1, att_feature1, vis_feature1], dim=-1
        )
        out_feature_2 = torch.cat(
            [ent_feature2, rel_feature2, att_feature2, vis_feature2], dim=-1
        )

        # gated fusion (dynamic Linear to match dims)
        z = torch.cat([out_feature_1, out_feature_2], dim=-1)
        gate = torch.sigmoid(nn.Linear(z.size(-1), out_feature_1.size(-1)).to(z.device)(z))
        out_feature = torch.cat(
            [gate * out_feature_1, (1 - gate) * out_feature_2], dim=-1
        )

        out_feature = self.dropout(out_feature)
        return out_feature, out_feature_1, out_feature_2

    # -----------------------------------------------------------------------------
    # training forward / losses
    # -----------------------------------------------------------------------------
    def forward(self, train_paris: torch.Tensor, flag):
        """
        Compute loss over two branches; if flag is False, return 0 (safe guard).
        """
        if flag:
            _, f1, f2 = self.gcn_forward()
            loss_1 = self.align_loss(train_paris, f1)
            loss_2 = self.align_loss(train_paris, f2)
            return loss_1 + loss_2
        else:
            return torch.tensor(0.0, device=self.ent_embedding.weight.device)

    def align_loss(self, pairs, emb):
        """Margin-based alignment with all-entity negatives."""
        def squared_dist(A, B):
            a2 = torch.sum(A * A, dim=1).reshape(-1, 1)
            b2 = torch.sum(B * B, dim=1).reshape(1, -1)
            return a2 + b2 - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        # squared L2 distance for positives
        pos_dis = torch.sum((l_emb - r_emb) ** 2, dim=-1, keepdim=True)

        # negatives against all entities
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)
        del l_emb, r_emb

        # hinge-style scores with one-hot masking
        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        del r_neg_dis, l_neg_dis

        # per-row standardization (detach stats)
        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(
            r_loss, dim=-1, unbiased=False, keepdim=True
        ).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(
            l_loss, dim=-1, unbiased=False, keepdim=True
        ).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def loss_no_neg_samples(self, pairs, emb):
        """Pure L2 loss on positives only (kept for compatibility)."""
        if len(pairs) == 0:
            return 0.0
        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        diff = emb[l] - emb[r]
        return torch.sum(torch.sum(diff * diff, dim=-1))

    # -----------------------------------------------------------------------------
    # embedding export
    # -----------------------------------------------------------------------------
    def get_embeddings(self, index_a, index_b):
        """
        Return normalized embeddings for given index lists.
        """
        out_feature, _, _ = self.gcn_forward()
        out_feature = out_feature.cpu()

        index_a = torch.as_tensor(index_a, dtype=torch.long)
        index_b = torch.as_tensor(index_b, dtype=torch.long)

        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        return Lvec, Rvec, out_feature
