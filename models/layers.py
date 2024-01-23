import math
import pickle

import numpy as np
import torch
from torch import nn

from models.utils import SingleHeadAttentionLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        center_codes = torch.unsqueeze(code_x, dim=-1)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)

        center_embeddings = center_codes * c_embeddings
        neighbor_embeddings = neighbor_codes * n_embeddings

        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)

        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))

        return co_embeddings, no_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        center_codes = torch.unsqueeze(code_x, dim=-1)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)

        center_embeddings = center_codes * c_embeddings
        neighbor_embeddings = neighbor_codes * n_embeddings

        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)

        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))

        return co_embeddings, no_embeddings


class GraphLayerAblation(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        code_x = torch.ones_like(code_x).cuda()
        neighbor = torch.ones_like(neighbor).cuda()

        center_codes = torch.unsqueeze(code_x, dim=-1)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)

        center_embeddings = center_codes * c_embeddings
        neighbor_embeddings = neighbor_codes * n_embeddings

        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)

        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))

        return co_embeddings, no_embeddings


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.time_layer = torch.nn.Linear(1, time_size)

    def forward(self, interval, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, is_last, hidden_state=None):
        time_features = self.activation(self.time_layer(1.0 / torch.log(interval.unsqueeze(-1) + torch.exp(torch.tensor(1.0)))))

        if not is_last:
            m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
            m1_index = torch.where(m1 > 0)[0]
            m2_index = torch.where(m2 > 0)[0]
            m3_index = torch.where(m3 > 0)[0]

            h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
            output_m1 = 0
            output_m23 = 0
            if len(m1_index) > 0:
                m1_embedding = co_embeddings[m1_index]
                h = hidden_state[m1_index] if hidden_state is not None else None
                h_m1 = self.gru(m1_embedding, h)
                h_new[m1_index] = h_m1
                output_m1, _ = torch.max(h_m1, dim=-2)
            if t > 0 and len(m2_index) + len(m3_index) > 0:
                q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
                v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
                h_m23 = self.activation(self.single_head_attention(q, v, v))
                h_new[m2_index] = h_m23[:len(m2_index)]
                h_new[m3_index] = h_m23[len(m2_index):]
                output_m23, _ = torch.max(h_m23, dim=-2)

            if len(m1_index) == 0:
                output = output_m23
            elif len(m2_index) + len(m3_index) == 0:
                output = output_m1
            else:
                output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)

        elif is_last:
            output = self.gru(co_embeddings, hidden_state)
            h_new = None

        return output+time_features, h_new


class TransitionLayerAblation(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.time_layer = torch.nn.Linear(1, time_size)

    def forward(self, interval, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, is_last, hidden_state=None):
        time_features = self.activation(self.time_layer(1.0 / torch.log(interval.unsqueeze(-1) + torch.exp(torch.tensor(1.0)))))

        if not is_last:
            m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
            m1_index = torch.where(m1 > 0)[0]
            m2_index = torch.where(m2 > 0)[0]
            m3_index = torch.where(m3 > 0)[0]
            m123_index = torch.tensor(list(set(m1_index.tolist() + m2_index.tolist() + m3_index.tolist())))

            h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
            m123_embedding = co_embeddings[m123_index]

            h = hidden_state[m123_index] if hidden_state is not None else None
            h_m1 = self.gru(m123_embedding, h)
            h_new[m123_index] = h_m1
            output_m123, _ = torch.max(h_m1, dim=-2)

            output, _ = torch.max(torch.vstack([output_m123]), dim=-2)

        elif is_last:
            output = self.gru(co_embeddings, hidden_state)
            h_new = None

        return output+time_features, h_new


class TransitionLayerNoTime(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()
        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, interval, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, is_last, hidden_state=None):

        if not is_last:
            m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
            m1_index = torch.where(m1 > 0)[0]
            m2_index = torch.where(m2 > 0)[0]
            m3_index = torch.where(m3 > 0)[0]

            h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
            output_m1 = 0
            output_m23 = 0
            if len(m1_index) > 0:
                m1_embedding = co_embeddings[m1_index]
                h = hidden_state[m1_index] if hidden_state is not None else None
                h_m1 = self.gru(m1_embedding, h)
                h_new[m1_index] = h_m1
                output_m1, _ = torch.max(h_m1, dim=-2)
            if t > 0 and len(m2_index) + len(m3_index) > 0:
                q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
                v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
                h_m23 = self.activation(self.single_head_attention(q, v, v))
                h_new[m2_index] = h_m23[:len(m2_index)]
                h_new[m3_index] = h_m23[len(m2_index):]
                output_m23, _ = torch.max(h_m23, dim=-2)

            if len(m1_index) == 0:
                output = output_m23
            elif len(m2_index) + len(m3_index) == 0:
                output = output_m1
            else:
                output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)

        elif is_last:
            output = self.gru(co_embeddings, hidden_state)
            h_new = None

        return output, h_new


class GlobalTimeLayer(nn.Module):
    def __init__(self, query_size):
        super(GlobalTimeLayer, self).__init__()
        self.query_size = query_size
        self.selection_layer = torch.nn.Linear(1, query_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(query_size, query_size)

    def forward(self, intervals, final_queries, mask):
        seq_time_step = (1.0 / torch.log(intervals.unsqueeze(2) + torch.exp(torch.tensor(1.0)).cuda()))
        selection_feature = self.tanh(self.selection_layer(seq_time_step))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / math.sqrt(self.query_size)
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        # time_weights = self.weight_layer(selection_feature)
        return torch.softmax(selection_feature, 1)
