import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_all_GraphAttention1_v2



class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden, triple_size, node_size, rel_size, device,
                 adj_matrix, r_index, r_val, rel_matrix, ent_matrix, ill_ent, key_emb, value_emb, vis_emb,
                 dropout_rate=0.3,
                 gamma=3, lr=0.005, depth=2):
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.ill_ent = ill_ent

        # 定义激活函数
        self.activate = nn.Tanh()

        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)

        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        # ------------------ 对已有嵌入做维度变换---------------------
        d1 = len(key_emb[0])
        d2 = len(value_emb[0])
        d3 = len(vis_emb[0])

        # 创建三个线性层，将各自维度变换到128维
        key_linear = nn.Linear(d1, 128)
        value_linear = nn.Linear(d2, 128)
        vis_linear = nn.Linear(d3, 128)

        # 将 numpy 数组转换为 PyTorch 张量
        key_emb = torch.from_numpy(key_emb).float()
        value_emb = torch.from_numpy(value_emb).float()
        vis_emb = torch.from_numpy(vis_emb).float()

        # 应用线性层
        key_embedding = key_linear(key_emb)
        value_embedding = value_linear(value_emb)
        vis_embedding = vis_linear(vis_emb)

        # 将参数变为可训练的数
        self.key_embedding = nn.Embedding.from_pretrained(key_embedding, freeze=False)
        self.value_embedding = nn.Embedding.from_pretrained(value_embedding, freeze=False)
        self.vis_embedding = nn.Embedding.from_pretrained(vis_embedding, freeze=False)

        torch.nn.init.xavier_uniform_(self.key_embedding.weight)
        torch.nn.init.xavier_uniform_(self.value_embedding.weight)
        torch.nn.init.xavier_uniform_(self.vis_embedding.weight)

        # 加权融合
        self.fusion_linear = nn.Linear(3840, 1920)  # 定义在模型 init 中，只定义一次

        # 一致性信息获取网络
        self.e_encoder = NR_all_GraphAttention1_v2(node_size=self.node_size,
                                                   rel_size=self.rel_size,
                                                   triple_size=self.triple_size,
                                                   node_dim=self.node_hidden,
                                                   depth=self.depth,
                                                   use_bias=True
                                                   )
        self.r_encoder = NR_all_GraphAttention1_v2(node_size=self.node_size,
                                                   rel_size=self.rel_size,
                                                   triple_size=self.triple_size,
                                                   node_dim=self.node_hidden,
                                                   depth=self.depth,
                                                   use_bias=True
                                                   )
        # 属性的
        self.k_encoder = NR_all_GraphAttention1_v2(node_size=self.node_size,
                                                   rel_size=self.rel_size,
                                                   triple_size=self.triple_size,
                                                   node_dim=self.node_hidden,
                                                   depth=self.depth,
                                                   use_bias=True
                                                   )

        self.a_encoder = NR_all_GraphAttention1_v2(node_size=self.node_size,
                                                   rel_size=self.rel_size,
                                                   triple_size=self.triple_size,
                                                   node_dim=self.node_hidden,
                                                   depth=self.depth,
                                                   use_bias=True
                                                   )

        # 视觉的
        self.v_encoder = NR_all_GraphAttention1_v2(node_size=self.node_size,
                                                   rel_size=self.rel_size,
                                                   triple_size=self.triple_size,
                                                   node_dim=self.node_hidden,
                                                   depth=self.depth,
                                                   use_bias=True
                                                   )

    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def gcn_forward(self):
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        key_feature = self.avg(self.ent_adj, self.key_embedding.weight, self.node_size)
        value_feature = self.avg(self.ent_adj, self.value_embedding.weight, self.node_size)
        vis_feature = self.avg(self.ent_adj, self.vis_embedding.weight, self.node_size)
        # print(self.ent_embedding.weight)
        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]

        # ----5类信息的处理 -----
        # ------------------------------------------一致性信息的获取\异质性信息的获取---------------------------
        ent_feature1, ent_feature2 = self.e_encoder([ent_feature] + opt)
        rel_feature1, rel_feature2 = self.r_encoder([rel_feature] + opt)
        key_feature1, key_feature2 = self.k_encoder([key_feature] + opt)
        value_feature1, value_feature2 = self.a_encoder([value_feature] + opt)
        vis_feature1, vis_feature2 = self.v_encoder([vis_feature] + opt)

        # -----------------------------------------------一致性信息和异质性信息的合并---------------------------------
        out_feature_1 = torch.cat([ent_feature1, rel_feature1, key_feature1, value_feature1, vis_feature1], dim=-1)
        out_feature_2 = torch.cat([ent_feature2, rel_feature2, key_feature2, value_feature2, vis_feature2], dim=-1)

        z = torch.cat([out_feature_1, out_feature_2], dim=-1)
        # 引入可学习门控控制不同模态信息强度（保留拼接信息）
        gate = torch.sigmoid(torch.nn.Linear(z.size(-1), out_feature_1.size(-1)).to(z.device)(z))
        out_feature = torch.cat([gate * self.activate(out_feature_1), (1 - gate) * self.activate(out_feature_2)],
                                dim=-1)


        out_feature = self.dropout(out_feature)

        return out_feature, out_feature_1, out_feature_2

    def forward(self, train_paris: torch.Tensor, flag):
        _, out_feature_1, out_feature_2 = self.gcn_forward()
        loss_1 = self.align_loss(train_paris, out_feature_1)
        loss_2 = self.align_loss(train_paris, out_feature_2)
        loss = loss_1 + loss_2
        return loss

    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        # 他这里正样本用的是 差值的平方 欧式距离
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def loss_no_neg_samples(self, pairs, emb):
        if len(pairs) == 0:
            return 0.0

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]
        loss = torch.sum(torch.square(l_emb - r_emb), dim=-1)
        loss = torch.sum(loss)

        return loss

    def get_embeddings(self, index_a, index_b):
        # forward
        out_feature, _, _ = self.gcn_forward()
        out_feature = out_feature.cpu()

        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        return Lvec, Rvec, out_feature




