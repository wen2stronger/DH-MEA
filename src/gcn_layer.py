import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import math

class NR_all_GraphAttention1_v2(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=2,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False,
                 use_semantic_attn=True):  
        super(NR_all_GraphAttention1_v2, self).__init__()

        # ---------------------------- config ----------------------------
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth

        # Always-on semantic attention (gate + score over [src, nbr, diff])
        self.semantic_attn = nn.Linear(self.node_dim * 3, 1)
        self.semantic_gate = nn.Linear(self.node_dim * 3, 1)

        # Attention kernels per layer
        self.attn_kernels = nn.ParameterList()
        for _ in range(self.depth):
            k = torch.nn.Parameter(torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(k)
            self.attn_kernels.append(k)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, inputs):
        outputs_c, outputs_s = [], []

        # inputs: [features, rel_emb, adj, r_index, r_val]
        features = inputs[0]
        rel_emb = inputs[1]
        adj     = inputs[2]
        r_index = inputs[3]
        r_val   = inputs[4]

        # initial activation
        features = self.activation(features)
        outputs_c.append(features)
        outputs_s.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]

            # ----- relation-aware neighbor transform -----
            tri_rel = torch.sparse_coo_tensor(
                indices=r_index, values=r_val,
                size=[self.triple_size, self.rel_size], dtype=torch.float32
            )
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)          # [T, D]
            neighs  = features[adj[1, :].long()]                 # Nbr embeddings
            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs  = neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel

            # ----- structural attention over edges -----
            att_struct = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)  # [T]
            att_struct = torch.sparse_coo_tensor(indices=adj, values=att_struct,
                                                 size=[self.node_size, self.node_size])
            att_struct = torch.sparse.softmax(att_struct, dim=1)

            # aggregate (consistency branch)
            new_features = scatter_sum(
                src=neighs * torch.unsqueeze(att_struct.coalesce().values(), dim=-1),
                dim=0, index=adj[0, :].long()
            )
            features = self.activation(new_features)
            outputs_c.append(features)

            # ===================== difference branch =====================
            neighs     = features[adj[1, :].long()]
            source_emb = features[adj[0, :].long()]
            diff_raw   = source_emb - neighs               

            # semantic attention over [src, nbr, diff]
            att_input = torch.cat([source_emb, neighs, diff_raw], dim=-1)  # [T, 3D]
            att_score = torch.sigmoid(self.semantic_attn(att_input)).squeeze(-1)
            att_score = torch.clamp(att_score, min=1e-4)                    # avoid zeros on sparse graphs

            # fuse semantic and structural attentions via a learnable gate
            gate_score = torch.sigmoid(self.semantic_gate(att_input)).squeeze(-1)
            struct_vals = att_struct.coalesce().values()
            final_att = gate_score * att_score + (1 - gate_score) * struct_vals

            att = torch.sparse_coo_tensor(indices=adj, values=final_att,
                                          size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            # aggregate (difference branch) — no diff-proj
            new_diff_features = scatter_sum(
                src=diff_raw * torch.unsqueeze(att.coalesce().values(), dim=-1),
                dim=0, index=adj[0, :].long()
            )
            new_diff_features = self.activation(new_diff_features)
            outputs_s.append(new_diff_features)
            # =============================================================================

        outputs1 = torch.cat(outputs_c, dim=-1)
        outputs2 = torch.cat(outputs_s, dim=-1)
        return outputs1, outputs2
