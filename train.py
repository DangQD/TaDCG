import argparse
import csv
import os
import pickle
import random
import time

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from models.model import Model, TaDCGNoD, TaDCGNoE, TaDCGNoTime
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, load_procedure_map, load_medication_map
from metrics import evaluate_codes, evaluate_hf


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


def parse_arguments(parser):
    parser.add_argument('--seed', type=int, default=6669, help='Random seed (default value is 6669)')
    parser.add_argument('--dataset', type=str, default='mimic3', help='Dataset: mimic3 or mimic4 (default is mimic3)')
    parser.add_argument('--task', type=str, default='m', help='Dataset: h or m (default is h)')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Do use cuda? (default is Ture)')

    parser.add_argument('--code_size', type=int, default=48, help='code size (default is 48)')
    parser.add_argument('--graph_size', type=int, default=32, help='graph size (default is 32)')
    parser.add_argument('--query_size', type=int, default=64, help='query size (default is 32)')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size (default is 64 or 512)')
    parser.add_argument('--t_attention_size', type=int, default=32, help='t_attention_size(default is 32)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size (default is 32)')
    parser.add_argument('--epochs', type=int, default=60, help='The max epoch to train(default is 50)')
    parser.add_argument('--model_choice', type=str, default='TaDCG')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    seed = 6669
    dataset = 'mimic3'
    task = 'm'  # 'm' or 'h'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    code_size = 48
    graph_size = 32
    query_size = 64
    hidden_size = 512  # 64 or 512
    time_size = hidden_size
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 40
    model_choice = args.model_choice

    print("setting --dataset %s --task %s --code size %s --graph size %s --query size %s --hidden size %s --model_choice %s"
          % (dataset, task, code_size, graph_size, query_size, hidden_size, model_choice))


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('../Chet/data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)

    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    if model_choice == 'TaDCG':
        model = Model(code_num=code_num, code_size=code_size, adj=code_adj, graph_size=graph_size, hidden_size=hidden_size,
                      t_attention_size=t_attention_size, t_output_size=t_output_size, query_size=query_size, time_size=time_size,
                      output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    elif model_choice == 'TaDCGNoD':
        model = TaDCGNoD(code_num=code_num, code_size=code_size, adj=code_adj, graph_size=graph_size, hidden_size=hidden_size,
                      t_attention_size=t_attention_size, t_output_size=t_output_size, query_size=query_size, time_size=time_size,
                      output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    elif model_choice == 'TaDCGNoE':
        model = TaDCGNoE(code_num=code_num, code_size=code_size, adj=code_adj, graph_size=graph_size, hidden_size=hidden_size,
                      t_attention_size=t_attention_size, t_output_size=t_output_size, query_size=query_size, time_size=time_size,
                      output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    elif model_choice == 'TaDCGNoTime':
        model = TaDCGNoTime(code_num=code_num, code_size=code_size, adj=code_adj, graph_size=graph_size, hidden_size=hidden_size,
                      t_attention_size=t_attention_size, t_output_size=t_output_size, query_size=query_size, time_size=time_size,
                      output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # train
    results = []
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        apos_num = 0
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, procedure_x, medication_x, visit_lens, divided, y, neighbors, visit_intervals = train_data[step]
            visit_intervals[:, :-1] = visit_intervals[:, :-1] - visit_intervals[:, 1:]
            output = model(code_x, divided, neighbors, visit_lens, visit_intervals).squeeze()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        if task == 'h':
            valid_loss, valid_auc_score, valid_f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
            results.append([valid_auc_score, valid_f1_score])
        elif task == 'm':
            valid_loss, valid_f1_score, r_10, r_20, r_30, r_40 = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
            results.append([valid_f1_score, r_10, r_20, r_30, r_40])
        torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
    results = np.round(np.array(results), 4)
    max_result = np.max(results, axis=0)
    print(max_result)
