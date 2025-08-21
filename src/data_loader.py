import numpy as np
import scipy.sparse as sp
import os
import random


# -----------------------------------------------------------------------------
# KGs: basic loader & graph builder
# -----------------------------------------------------------------------------
class KGs:
    def __init__(self):
        self.entity1, self.rel1, self.triples1 = None, None, None
        self.entity2, self.rel2, self.triples2 = None, None, None
        self.att_triples, self.att = None, None
        self.train_pairs, self.valid_pairs, self.test_pairs = None, None, None
        self.new_ent = set()
        self.anchors = None
        self.new_test_pairs = []
        self.total_ent_num, self.total_rel_num, self.total_att_num = 0, 0, 0
        self.triple_num = 0
        self.credible_pairs = None
        self.new_ent_nei = np.array([])

    # -----------------------------------------------------------------------------
    # IO: triples & alignment pairs
    # -----------------------------------------------------------------------------
    def load_triples(self, file_name):
        triples = []
        entity = set()
        rel = {0}  # self-loop marker
        for line in open(file_name, 'r'):
            head, r, tail = [int(item) for item in line.split()]
            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)  # shift relation ids by +1
            triples.append((head, r + 1, tail))
        return entity, rel, triples

    def load_alignment_pair(self, file_name):
        alignment_pair = []
        for line in open(file_name, 'r'):
            e1, e2 = line.split()
            alignment_pair.append((int(e1), int(e2)))
        # np.random.shuffle(alignment_pair)
        return alignment_pair

    # -----------------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------------
    def get_matrix(self, triples, entity, rel):
        ent_size = max(entity) + 1
        rel_size = (max(rel) + 1)
        print(f"Union graph: ent size={ent_size}, rel_size={rel_size}")

        # row-based list-of-lists sparse matrices
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))

        # add self-loops on diagonal
        for i in range(max(entity) + 1):
            adj_features[i, i] = 1

        # add undirected edges and relation counts
        for h, r, t in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1
            adj_features[t, h] = 1
            radj.append([h, t, r])
            radj.append([t, h, r + rel_size])  # inverse edge relation index
            rel_out[h][r] += 1
            rel_in[t][r] += 1

        # build (r_index, r_val) by grouping on (h, t)
        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
            if ' '.join([str(h), str(t)]) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h), str(t)]))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        # relation features: [in | out]
        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        rel_features = sp.lil_matrix(rel_features)
        # adj_matrix stores entries like: (row, col) -> 1.0
        return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features

    # -----------------------------------------------------------------------------
    # Dataset split
    # -----------------------------------------------------------------------------
    def load_base(self, path, rates):
        entity1, rel1, triples1 = self.load_triples(os.path.join(path, 'triples_1'))
        entity2, rel2, triples2 = self.load_triples(os.path.join(path, 'triples_2'))
        ill_ent = self.load_alignment_pair(os.path.join(path, 'ref_ent_ids'))

        # assume ill_ent is a list; shuffle in-place
        random.shuffle(ill_ent)
        # fixed split: train / valid / test
        train_pairs = ill_ent[:int(len(ill_ent) // 1 * rates)]
        valid_pairs = ill_ent[int(len(ill_ent) // 1 * rates): int(len(ill_ent) // 1 * (rates + 0.1))]
        test_pairs = ill_ent[int(len(ill_ent) // 1 * (rates + 0.1)):]

        return [entity1, rel1, triples1, entity2, rel2, triples2, train_pairs, valid_pairs, test_pairs, ill_ent]

    # -----------------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------------
    def load_data(self, dir, args):
        base_data = self.load_base(os.path.join(dir, args.dataset), 0.2)
        self.entity1, self.rel1, self.triples1 = base_data[0:3]
        self.entity2, self.rel2, self.triples2 = base_data[3:6]
        self.train_pairs, self.valid_pairs, self.test_pairs = base_data[6:9]
        self.ill_ent = base_data[9]
        self.anchors = set([p[0] for p in self.train_pairs]).union(set([p[1] for p in self.train_pairs]))
        self.old_ent_num = len(self.entity1) + len(self.entity2)  # stat
        self.old_ent_set = self.entity1.union(self.entity2)

        print(args.dataset + "的详细信息如下：")
        print(f"KG1 entity num={len(self.entity1)}, relation num={len(self.rel1)}, triple num={len(self.triples1)}")
        print(f"KG2 entity num={len(self.entity2)}, relation num={len(self.rel2)}, triple num={len(self.triples2)}")
        print(
            f"train pairs: {len(self.train_pairs)}, valid pairs: {len(self.valid_pairs)}, test pairs: {len(self.test_pairs)}, new test pairs: {len(self.new_test_pairs)}")

        adj_matrix, r_index, r_val, adj_features, rel_features = self.get_matrix(self.triples1 + self.triples2,
                                                                                 self.entity1.union(self.entity2),
                                                                                 self.rel1.union(self.rel2))
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        self.total_ent_num = adj_features.shape[0]
        self.total_rel_num = rel_features.shape[1]  # includes self-loop & inverse edges
        self.triple_num = ent_adj.shape[0]  # edge count

        self.train_pairs = np.array(self.train_pairs)
        self.valid_pairs = np.array(self.valid_pairs)
        self.test_pairs = np.array(self.test_pairs)

        return self.train_pairs, self.valid_pairs, self.test_pairs, \
            ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj, list(self.entity1), list(self.entity2), np.array(
            self.ill_ent), self.triples1, self.triples2
