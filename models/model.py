import numpy as np
import torch
from torch import nn

from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer, GlobalTimeLayer, GraphLayerAblation, TransitionLayerAblation, \
    TransitionLayerNoTime
from models.utils import DotProductAttention, DotProductAttentionOther


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, query_size, time_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_size = time_size
        self.graph_size = graph_size
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.graph_layer_ablation = GraphLayerAblation(adj, code_size, graph_size)
        self.transition_layer_ablation = TransitionLayerAblation(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.transition_layer_no_time = TransitionLayerNoTime(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 64)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

        self.attention_time = DotProductAttentionOther(hidden_size, time_size, 32)
        self.self_layer = torch.nn.Linear(hidden_size, 1)
        self.query_layer = torch.nn.Linear(hidden_size, query_size)
        self.time_encoder = GlobalTimeLayer(query_size)
        self.query_weight_layer = torch.nn.Linear(hidden_size, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, code_x, divided, neighbors, lens, intervals):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        sc_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()
        su_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()

        batch_size, maxlen = code_x.shape[0], code_x.shape[1]
        features = torch.zeros((batch_size, maxlen+1, self.hidden_size)).cuda()
        features_no_time = torch.zeros((batch_size, self.hidden_size)).cuda()
        mask = np.zeros((batch_size, maxlen+1), dtype=float)
        mask_final = np.zeros((batch_size, maxlen+1, 1), dtype=float)
        for i in range(batch_size):
            mask[i, :lens[i]] = 1
            mask_final[i, lens[i], 0] = 1
        mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
        mask_final = torch.Tensor(mask_final).cuda()
        output = []
        output_no_time = []
        for i, (code_x_i, divided_i, neighbor_i, len_i, interval_i) in enumerate(zip(code_x, divided, neighbors, lens, intervals)):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, interval_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), interval_i)):
                is_last = False
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                output_it, h_t = self.transition_layer(interval_it, t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, is_last, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                features[i][t] = output_it
                features_no_time[i] += output_it
                if t == len_i-1:
                    is_last = True
                    visit_feature, _ = self.transition_layer(interval_it, t+1, sc_embedding, d_it, no_embeddings_i_prev, su_embedding, is_last, output_it)
                    output_i.append(visit_feature)
                    features[i][t+1] = visit_feature
            tmp = torch.vstack(output_i)
            output_i = self.attention(tmp)
            output.append(output_i)
            output_i_no_time = torch.sum(tmp, dim=0) / len_i
            output_no_time.append(output_i_no_time)
        final_statues = features * mask_final
        queries = self.relu(self.query_layer(final_statues))
        self_weight = self.get_self_attention(features, mask_mult)
        new_intervals = torch.zeros((intervals.shape[0], intervals.shape[1]+1)).cuda()
        new_intervals[:, :intervals.shape[1]] = intervals
        time_weight = self.time_encoder(new_intervals, queries, mask_mult)
        attention_weight = torch.softmax(self.query_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)

        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)

        predictions = self.classifier(averaged_features)

        return predictions


class TaDCGNoD(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, query_size, time_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_size = time_size
        self.graph_size = graph_size
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.graph_layer_ablation = GraphLayerAblation(adj, code_size, graph_size)
        self.transition_layer_ablation = TransitionLayerAblation(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.transition_layer_no_time = TransitionLayerNoTime(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 64)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

        self.attention_time = DotProductAttentionOther(hidden_size, time_size, 32)
        self.self_layer = torch.nn.Linear(hidden_size, 1)
        self.query_layer = torch.nn.Linear(hidden_size, query_size)
        self.time_encoder = GlobalTimeLayer(query_size)
        self.query_weight_layer = torch.nn.Linear(hidden_size, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, code_x, divided, neighbors, lens, intervals):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        sc_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()
        su_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()

        batch_size, maxlen = code_x.shape[0], code_x.shape[1]
        features = torch.zeros((batch_size, maxlen+1, self.hidden_size)).cuda()
        features_no_time = torch.zeros((batch_size, self.hidden_size)).cuda()
        mask = np.zeros((batch_size, maxlen+1), dtype=float)
        mask_final = np.zeros((batch_size, maxlen+1, 1), dtype=float)
        for i in range(batch_size):
            mask[i, :lens[i]] = 1
            mask_final[i, lens[i], 0] = 1
        mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
        mask_final = torch.Tensor(mask_final).cuda()
        output = []
        output_no_time = []
        for i, (code_x_i, divided_i, neighbor_i, len_i, interval_i) in enumerate(zip(code_x, divided, neighbors, lens, intervals)):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, interval_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), interval_i)):
                is_last = False
                co_embeddings, no_embeddings = self.graph_layer_ablation(c_it, n_it, c_embeddings, n_embeddings)
                output_it, h_t = self.transition_layer(interval_it, t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, is_last, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                features[i][t] = output_it
                features_no_time[i] += output_it
                if t == len_i-1:
                    is_last = True
                    visit_feature, _ = self.transition_layer(interval_it, t+1, sc_embedding, d_it, no_embeddings_i_prev, su_embedding, is_last, output_it)
                    output_i.append(visit_feature)
                    features[i][t+1] = visit_feature
            tmp = torch.vstack(output_i)
            output_i_no_time = torch.sum(tmp, dim=0) / len_i
            output_no_time.append(output_i_no_time)
        final_statues = features * mask_final
        queries = self.relu(self.query_layer(final_statues))
        self_weight = self.get_self_attention(features, mask_mult)
        new_intervals = torch.zeros((intervals.shape[0], intervals.shape[1]+1)).cuda()
        new_intervals[:, :intervals.shape[1]] = intervals
        time_weight = self.time_encoder(new_intervals, queries, mask_mult)
        attention_weight = torch.softmax(self.query_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)

        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)

        predictions = self.classifier(averaged_features)

        return predictions


class TaDCGNoE(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, query_size, time_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_size = time_size
        self.graph_size = graph_size
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.graph_layer_ablation = GraphLayerAblation(adj, code_size, graph_size)
        self.transition_layer_ablation = TransitionLayerAblation(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.transition_layer_no_time = TransitionLayerNoTime(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 64)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

        self.attention_time = DotProductAttentionOther(hidden_size, time_size, 32)
        self.self_layer = torch.nn.Linear(hidden_size, 1)
        self.query_layer = torch.nn.Linear(hidden_size, query_size)
        self.time_encoder = GlobalTimeLayer(query_size)
        self.query_weight_layer = torch.nn.Linear(hidden_size, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, code_x, divided, neighbors, lens, intervals):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        sc_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()
        su_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()

        batch_size, maxlen = code_x.shape[0], code_x.shape[1]
        features = torch.zeros((batch_size, maxlen+1, self.hidden_size)).cuda()
        features_no_time = torch.zeros((batch_size, self.hidden_size)).cuda()
        mask = np.zeros((batch_size, maxlen+1), dtype=float)
        mask_final = np.zeros((batch_size, maxlen+1, 1), dtype=float)
        for i in range(batch_size):
            mask[i, :lens[i]] = 1
            mask_final[i, lens[i], 0] = 1
        mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
        mask_final = torch.Tensor(mask_final).cuda()
        output = []
        output_no_time = []
        for i, (code_x_i, divided_i, neighbor_i, len_i, interval_i) in enumerate(zip(code_x, divided, neighbors, lens, intervals)):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, interval_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), interval_i)):
                is_last = False
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                output_it, h_t = self.transition_layer_ablation(interval_it, t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, is_last, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                features[i][t] = output_it
                features_no_time[i] += output_it
                if t == len_i-1:
                    is_last = True
                    visit_feature, _ = self.transition_layer(interval_it, t+1, sc_embedding, d_it, no_embeddings_i_prev, su_embedding, is_last, output_it)
                    output_i.append(visit_feature)
                    features[i][t+1] = visit_feature
            tmp = torch.vstack(output_i)
            output_i = self.attention(tmp)
            output.append(output_i)
            output_i_no_time = torch.sum(tmp, dim=0) / len_i
            output_no_time.append(output_i_no_time)
        final_statues = features * mask_final
        queries = self.relu(self.query_layer(final_statues))
        self_weight = self.get_self_attention(features, mask_mult)
        new_intervals = torch.zeros((intervals.shape[0], intervals.shape[1]+1)).cuda()
        new_intervals[:, :intervals.shape[1]] = intervals
        time_weight = self.time_encoder(new_intervals, queries, mask_mult)
        attention_weight = torch.softmax(self.query_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)

        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)

        predictions = self.classifier(averaged_features)

        return predictions


class TaDCGNoTime(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, query_size, time_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_size = time_size
        self.graph_size = graph_size
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.graph_layer_ablation = GraphLayerAblation(adj, code_size, graph_size)
        self.transition_layer_ablation = TransitionLayerAblation(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.transition_layer_no_time = TransitionLayerNoTime(code_num, graph_size, hidden_size, time_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 64)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, code_x, divided, neighbors, lens, intervals):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        sc_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()
        su_embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(1, self.graph_size))).cuda().squeeze()

        batch_size, maxlen = code_x.shape[0], code_x.shape[1]
        features = torch.zeros((batch_size, maxlen+1, self.hidden_size)).cuda()
        output_no_time = []
        for i, (code_x_i, divided_i, neighbor_i, len_i, interval_i) in enumerate(zip(code_x, divided, neighbors, lens, intervals)):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, interval_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), interval_i)):
                is_last = False
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                output_it, h_t = self.transition_layer_no_time(interval_it, t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, is_last, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                features[i][t] = output_it
                if t == len_i-1:
                    is_last = True
                    visit_feature, _ = self.transition_layer(interval_it, t+1, sc_embedding, d_it, no_embeddings_i_prev, su_embedding, is_last, output_it)
                    output_i.append(visit_feature)
                    features[i][t+1] = visit_feature
            tmp = torch.vstack(output_i)
            output_i_no_time = torch.sum(tmp, dim=0) / len_i
            output_no_time.append(output_i_no_time)

        predictions = self.classifier(torch.vstack(output_no_time))

        return predictions
