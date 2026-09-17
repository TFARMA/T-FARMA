import argparse
import os
import random
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.modules.rnn
from tqdm import tqdm

sys.path.append("..")
torch.set_printoptions(sci_mode=False)
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
import copy


def continual_test(model, history_list, data_list, num_rels, num_nodes, use_cuda, all_ans_list,
                   model_name, mode, static_graph):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """

    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    # load pretrained model which valid in the whole valid data
    start_idx = len(history_list)
    if not os.path.exists(model_name):
        print("Pretrain the model first before continual learning...")
        sys.exit()
    else:
        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        init_checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        print("Load pretrain model: {}. Using best epoch : {}".format(model_name,
                                                                      checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start continual learning" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        # save an init model for analysis
        model_initial = copy.deepcopy(model)
        model_initial.load_state_dict(init_checkpoint['state_dict'], strict=False)
        model_initial.eval()
        model.eval()
        epoch = checkpoint['epoch']
    previous_param = [param.detach().clone() for param in model.parameters()]
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    valid_input_list = [snap for snap in
                        history_list[-args.test_history_len - 2:-2]]  # history for ft training (, tc-2)
    valid_snap = history_list[-2]  # snapshot for ft training at snapshot at tc-2
    ft_input_list = [snap for snap in history_list[-args.test_history_len - 1:-1]]  # history for ft validation (,tc-1)
    ft_snap = history_list[-1]  # snapshot for ft validation snapshot at tc-1
    test_input_list = [snap for snap in history_list[-args.test_history_len:]]  # history for testing (, tc)
    # starting continual learning
    for time_idx, test_snap in enumerate(data_list):
        tc = start_idx + time_idx
        print("-----------------------{}-----------------------".format(tc))
        # step 1: get the history graphs for ft training : ft_input_list -> ft_snapshot
        ft_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in ft_input_list]
        ft_tensor = torch.LongTensor(ft_snap).cuda() if use_cuda else torch.LongTensor(ft_snap)
        # step 2: prepare inputs for test
        test_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in test_input_list]
        test_tensor = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_tensor = test_tensor.to(args.gpu)

        def create_variable(val_tensor, val_list):
            all_tail_seq = sp.load_npz(
                "../data/{}/history/tail_history_{}.npz".format(args.dataset, int(val_tensor[0, -1])))
            all_tail_decay_seq = sp.load_npz(
                "../data/{}/history/tail_decay_history_{}.npz".format(args.dataset, int(val_tensor[0, -1]))
            )
            histroy_data_npy = val_tensor.cpu().numpy()
            que_pair = utils.e2r(histroy_data_npy[:, :3])
            seq_idx = histroy_data_npy[:, 0] * num_rels + histroy_data_npy[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            history_g = build_sub_graph(num_nodes, num_rels, np.unique(np.vstack(val_list)[:, :3], axis=0), use_cuda,
                                        args.gpu)
            tail_seq_decay = torch.Tensor(all_tail_decay_seq[seq_idx].todense())
            one_hot_tail_seq = one_hot_tail_seq.cuda() if use_cuda else one_hot_tail_seq
            tail_seq_decay = tail_seq_decay.cuda() if use_cuda else tail_seq_decay
            return one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay

        """
        step 2: get the history graphs for ft validation : valid_input_list -> valid_snap
        # result of the pre-trained model on validation set (tc-1)
        valid_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in valid_input_list]
        valid_tensor = torch.LongTensor(valid_snap).cuda() if use_cuda else torch.LongTensor(valid_snap)

        one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(valid_tensor, 
        valid_input_list)
        _, final_score = model.predict(valid_history_glist, static_graph, valid_tensor,
                                       one_hot_tail_seq, use_cuda, history_g, que_pair, all_tail_seq, tail_seq_decay)
        mrr_filter_valid_snap, mrr_valid_snap, rank_raw_init, rank_filter_init = utils.get_total_rank(
            valid_tensor, final_score,
            all_ans_list[tc - 2],
            eval_bz=1000,
            rel_predict=0)
        """
        # result of the pre-trained model on test set (tc)
        one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(test_tensor,
                                                                                              test_input_list)
        _, final_score = model_initial.predict(test_history_glist, num_rels, static_graph, test_tensor,
                                               one_hot_tail_seq, use_cuda, history_g, que_pair, all_tail_seq,
                                               tail_seq_decay)
        mrr_filter_test_snap_init, mrr_test_snap_init, rank_raw_init, rank_filter_init = utils.get_total_rank(
            test_tensor, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0
        )
        print("Pretrained Model : test mrr ", mrr_filter_test_snap_init)
        # result of the last step fine-tuned model on test set (tc)
        _, final_score = model.predict(test_history_glist, static_graph, test_tensor, one_hot_tail_seq,
                                       use_cuda, history_g, que_pair, all_tail_seq, tail_seq_decay)
        mrr_filter_test_snap, mrr_test_snap, rank_raw, rank_filter = utils.get_total_rank(test_tensor, final_score,
                                                                                          all_ans_list[tc],
                                                                                          eval_bz=1000, rel_predict=0)
        print("Continual Model : test mrr before ft ", mrr_filter_test_snap)
        # init mrr for validation
        best_mrr = mrr_filter_test_snap  # use test set for validation
        # best_mrr = mrr_filter_valid_snap  use valid data for validation
        ft_epoch, losses = 0, []

        def temporal_regularization(params1, params2):
            regular = 0
            for (param1, param2) in zip(params1, params2):
                regular += torch.norm(param1 - param2, p=2)
            return regular

        while ft_epoch < 30:
            model.train()
            one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(ft_tensor,
                                                                                                  ft_input_list)
            loss_e, loss_static, loss_cl = model.get_loss(ft_history_glist, ft_tensor, static_graph,
                                                          one_hot_tail_seq,
                                                          use_cuda, history_g, tc, que_pair,
                                                          all_tail_seq)
            loss = loss_e + loss_static + loss_cl
            loss_norm = temporal_regularization(model.parameters(), previous_param)
            loss += loss_norm
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            model.eval()

            # validation on tc snapshot

            one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(test_tensor,
                                                                                                  test_input_list)
            _, final_score = model.predict(test_history_glist, static_graph, test_tensor, one_hot_tail_seq,
                                           use_cuda, history_g, que_pair, all_tail_seq, tail_seq_decay)

            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_tensor, final_score,
                                                                                    all_ans_list[tc], eval_bz=1000,
                                                                                    rel_predict=0)
            # validation on tc-1 snapshot
            # one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(valid_tensor,
            # valid_input_list)
            # _, final_score = model.predict(valid_history_glist,  static_graph, valid_tensor,
            #                                one_hot_tail_seq, use_cuda, history_g, que_pair, all_tail_seq,
            #                                tail_seq_decay)
            # mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            #     valid_tensor, final_score,
            #     all_ans_list[tc - 2],
            #     eval_bz=1000,
            #     rel_predict=0)

            ft_epoch += 1
            # update best_mrr
            if mrr_filter_snap > best_mrr:
                print(mrr_filter_snap, best_mrr, ft_epoch)
                best_mrr = mrr_filter_snap
                torch.save({"state_dict":model.state_dict(), "epoch":epoch}, model_name[:-4] + "_" + str(tc) + ".pth")
            else:
                if not os.path.exists(model_name[:-4] + str(tc) + ".pth"):
                    if time_idx == 0:
                        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
                    else:
                        checkpoint = torch.load(model_name[:-4] + str(tc - 1) + ".pth",
                                                map_location=torch.device(args.gpu))
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    torch.save({"state_dict":model.state_dict(), "epoch":epoch},
                               model_name[:-4] + "_" + str(tc) + ".pth")

        previous_param = [param.detach().clone() for param in model.parameters()]
        # ---------------start evaluate test snaoshot---------------

        # step 1: load current model
        checkpoint = torch.load(model_name[:-4] + "_" + str(tc) + ".pth", map_location=torch.device(args.gpu))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
        # step 2: start test
        one_hot_tail_seq, history_g, que_pair, all_tail_seq, tail_seq_decay = create_variable(test_tensor,
                                                                                              test_input_list)
        _, final_score = model.predict(test_history_glist, static_graph, test_tensor, one_hot_tail_seq,
                                       use_cuda, history_g, que_pair, all_tail_seq, tail_seq_decay)
        # step 3: evaluation
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            test_tensor, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0
        )
        print("Continual Model : ***test_mrr*** ", mrr_filter_snap)
        # step 4: update history glist and prepare inputs
        ft_input_list.pop(0)
        ft_input_list.append(ft_snap)
        valid_input_list.pop(0)
        valid_input_list.append(valid_snap)
        test_input_list.pop(0)
        test_input_list.append(test_snap)
        valid_snap = ft_snap.copy()
        ft_snap = test_snap.copy()
        # step 5: save results
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    return mrr_raw, mrr_filter


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, model_name, static_graph,
         time_list, history_time_nogt, mode, bid_num, ):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    # ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    idx = 0
    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device("cpu"))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint["epoch"]))
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        if args.multi_step:
            all_tail_seq = sp.load_npz(
                "../data/{}/history/tail_history_{}.npz".format(args.dataset, history_time_nogt)
            )
            all_tail_decay_seq = sp.load_npz(
                "../data/{}/history/tail_decay_history_{}.npz".format(args.dataset, history_time_nogt)
            )
        else:
            all_tail_seq = sp.load_npz(
                "../data/{}/history/tail_history_{}.npz".format(args.dataset, time_list[time_idx])
            )
            all_tail_decay_seq = sp.load_npz(
                "../data/{}/history/tail_decay_history_{}.npz".format(args.dataset, time_list[time_idx])
            )
        history_glist = [
            build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu, bid_num)
            for g in input_list
        ]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)

        histroy_data_npy = test_triples_input.cpu().numpy()
        que_pair = utils.e2r(histroy_data_npy[:, :3])
        seq_idx = histroy_data_npy[:, 0] * num_rels * bid_num + histroy_data_npy[:, 1]
        tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
        one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
        tail_seq_decay = torch.Tensor(all_tail_decay_seq[seq_idx].todense())
        one_hot_tail_seq = one_hot_tail_seq.cuda() if use_cuda else one_hot_tail_seq
        tail_seq_decay = tail_seq_decay.cuda() if use_cuda else tail_seq_decay

        '''
        # bid_num == 2
        inv_histroy_data = test_triples_input[:, [2, 1, 0, 3]]
        inv_histroy_data[:, 1] = inv_histroy_data[:, 1] + num_rels
        inv_histroy_data_npy = inv_histroy_data.cpu().numpy()
        que_pair_inv = utils.e2r(inv_histroy_data_npy[:, :3])
        seq_idx_inv = inv_histroy_data_npy[:, 0] * num_rels * bid_num + inv_histroy_data_npy[:, 1]
        tail_seq_inv = torch.Tensor(all_tail_seq[seq_idx_inv].todense())
        one_hot_tail_seq_inv = tail_seq_inv.masked_fill(tail_seq_inv != 0, 1)
        tail_seq_decay_inv = torch.Tensor(all_tail_decay_seq[seq_idx_inv].todense())
        one_hot_tail_seq_inv = one_hot_tail_seq_inv.cuda() if use_cuda else one_hot_tail_seq_inv
        tail_seq_decay_inv = tail_seq_decay_inv.cuda() if use_cuda else tail_seq_decay_inv
        '''

        # Concatenate historical data
        snap_global_history = np.unique(np.vstack(history_list)[:, :3], axis=0)
        history_g = build_sub_graph(num_nodes, num_rels, snap_global_history, use_cuda, args.gpu, bid_num)

        test_triples, final_score = model.predict(history_glist, static_graph, test_triples_input,
                                                  one_hot_tail_seq, use_cuda, history_g, que_pair, all_tail_seq,
                                                  tail_seq_decay)

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)


        '''
        test_triples_inv, final_score_inv = model.predict(history_glist, static_graph, inv_histroy_data,
                                                          one_hot_tail_seq_inv, use_cuda, history_g, que_pair_inv,
                                                          all_tail_seq, tail_seq_decay_inv)

        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(
            test_triples_inv, final_score_inv, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
        mrr_raw_list_inv.append(mrr_snap_inv)
        mrr_filter_list_inv.append(mrr_filter_snap_inv)
        '''
        if args.multi_step:
            predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
        # if [8, 16, 1, 336] in test_triples.cpu().tolist():
        #     index = test_triples.cpu().tolist().index([8, 16, 1, 336])
        #     score_tensor, indices = torch.sort(torch.exp(final_score)[index, :], descending=True)
        #     print(score_tensor[:5])
        #     print(indices[:5])
        #     break
        #
        # def get_score(score):
        #     score_tensor, indices = torch.sort(score, dim=1, descending=True)
        #     mask = (indices == (test_triples[:, 2].unsqueeze(-1)))
        #     target_score = score_tensor[torch.nonzero(mask)[:, 0], torch.nonzero(mask)[:, 1]]
        #     return target_score.unsqueeze(-1)
        #
        # t_score = get_score(torch.exp(final_score))
        # rank_filter_combine = torch.cat((test_triples, rank_filter.unsqueeze(-1), t_score,), dim=1)

    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")

    '''
    mrr_raw_inv, hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_ent_inv")
    mrr_filter_inv, hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_ent_inv")
    avg_mrr_raw = (mrr_raw + mrr_raw_inv) / 2
    avg_mrr_filter = (mrr_filter + mrr_filter_inv) / 2
    avg_hit_raw, avg_hit_filter, avg_hit_raw_r, avg_hit_filter_r = [], [], [], []
    for hit_id in range(len(hit_result_raw)):
        avg_hit_raw.append((hit_result_raw[hit_id] + hit_raw_inv[hit_id]) / 2)
        avg_hit_filter.append((hit_result_filter[hit_id] + hit_filter_inv[hit_id]) / 2)
    hits = [1, 3, 10]

    print("MRR ({}): {:.6f}".format("raw_ent_bid", avg_mrr_raw.item()))
    for index, hit in enumerate(hits):
        print("Hits ({}) @ {}: {:.6f}".format("raw_ent_bid", hit, avg_hit_raw[index]))

    print("MRR ({}): {:.6f}".format("filter_ent_bid", avg_mrr_filter.item()))
    for index, hit in enumerate(hits):
        print("Hits ({}) @ {}: {:.6f}".format("filter_ent_bid", hit, avg_hit_filter[index]))
    '''
    return mrr_raw, mrr_filter, hit_result_raw, hit_result_filter


def run_experiment(args, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None, ):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate
    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)  # 得到data类
    train_list, train_times = utils.split_by_time(data.train)  # 划分为snapshots，逐时间步的数据集
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)
    total_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    if args.dataset == "ICEWS14":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1] - train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    print(args)
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, False)
    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp_{}.pth".format(
        args.history_rate,
        args.dataset,
        args.encoder,
        args.decoder,
        args.n_layers,
        args.dilate_len,
        args.train_history_len,
        args.weight,
        args.discount,
        args.angle,
        args.dropout,
    )
    model_state_file = os.path.join("../models/", model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = (torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(
            args.gpu) if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long())
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    # create stat
    if args.dataset in ["YAGO", "WIKI"]:
        bid_num = 2
    else:
        bid_num = 1

    model = RecurrentRGCN(args.decoder, args.encoder, num_nodes, num_rels, num_static_rels, num_words, num_times,
                          time_interval, args.n_hidden, args.opn, args.history_rate,
                          sequence_len=args.train_history_len, num_bases=args.n_bases, num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers, dropout=args.dropout, self_loop=args.self_loop,
                          skip_connect=args.skip_connect, layer_norm=args.layer_norm, input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation, weight=args.weight, discount=args.discount, angle=args.angle,
                          use_static=args.add_static_graph, entity_prediction=args.entity_prediction,
                          relation_prediction=args.relation_prediction, use_cuda=use_cuda, gpu=args.gpu,
                          analysis=args.run_analysis, att_type=args.att_type, use_cl=args.use_cl, bid_num=bid_num,
                          rel_weight=args.rel_weight, decay_rate=args.tfa_weight)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        # checkpoint = torch.load(model_state_file, map_location=torch.device(args.gpu))
        # model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu,
                                       bid_num)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.test and os.path.exists(model_state_file):
        if args.continual_test:
            print("online trainging: train the models with timestamps in the valid set")
            mrr_raw, mrr_filter = continual_test(model, train_list, valid_list, num_rels, num_nodes,
                                                 use_cuda, all_ans_list, model_state_file, "test",
                                                 static_graph)  # online trainging -step 1

            print("online trainging: train the models with timestamps in the test set")
            mrr_raw, mrr_filter = continual_test(model, train_list + valid_list, test_list, num_rels, num_nodes,
                                                 use_cuda, all_ans_list, model_state_file, "test",
                                                 static_graph)  # online trainging -step 2

        else:
            mrr_raw, mrr_filter, hit_result_raw, hit_result_filter = test(model, train_list + valid_list, test_list,
                                                                          num_rels, num_nodes, use_cuda,
                                                                          all_ans_list_test, model_state_file,
                                                                          static_graph, test_times,
                                                                          history_test_time_nogt, mode="test",
                                                                          bid_num=bid_num, )
        return mrr_raw, mrr_filter, hit_result_raw, hit_result_filter
    elif args.test and not os.path.exists(model_state_file):
        print("----{} not exist, Change mode to train and generate stat for testing------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_static = []
            losses_cl = []
            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)
            for train_sample_num in tqdm(idx):
                if train_sample_num == 0:
                    continue
                output = train_list[train_sample_num: train_sample_num + 1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0:train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]
                # Generate the global graph before the current time.
                snap_global_history = np.unique(np.vstack(train_list[:train_sample_num])[:, :3], axis=0)
                """
                # Obtain the global graph nodes and one-hop neighboring nodes at the current time.
                one_hop_neigbhbor = np.union1d(np.unique(output[0][:, 0]), np.unique(output[0][:, 0]))
                snap_global_history = snap_global_history[np.isin(snap_global_history[:, 0], one_hop_neigbhbor)]
                """
                history_g = build_sub_graph(num_nodes, num_rels, snap_global_history, use_cuda, args.gpu, bid_num)
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu, bid_num) for snap in
                                 input_list]

                output = (
                    [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for
                                                                                          _ in output])
                histroy_data = output[0]
                if bid_num == 2:
                    inv_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                    inv_histroy_data[:, 1] = inv_histroy_data[:, 1] + num_rels
                    # histroy_data = torch.cat([histroy_data, inv_histroy_data])
                    histroy_data_npy = histroy_data.cpu().numpy()
                    inv_histroy_data_npy = inv_histroy_data.cpu().numpy()
                    que_pair = utils.e2r(histroy_data_npy[:, :3])
                    que_pair_inv = utils.e2r(inv_histroy_data_npy[:, :3])
                else:
                    histroy_data_npy = histroy_data.cpu().numpy()
                    que_pair = utils.e2r(histroy_data_npy[:, :3])
                # tail
                all_tail_seq = sp.load_npz(
                    "../data/{}/history/tail_history_{}.npz".format(args.dataset, train_times[train_sample_num]))

                if bid_num == 2:
                    seq_idx = histroy_data_npy[:, 0] * num_rels * bid_num + histroy_data_npy[:, 1]
                    tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
                    seq_idx_inv = inv_histroy_data_npy[:, 0] * num_rels * bid_num + inv_histroy_data_npy[:, 1]
                    tail_seq_inv = torch.Tensor(all_tail_seq[seq_idx_inv].todense())
                    one_hot_tail_seq_inv = tail_seq_inv.masked_fill(tail_seq_inv != 0, 1)

                else:
                    seq_idx = histroy_data_npy[:, 0] * num_rels * bid_num + histroy_data_npy[:, 1]
                    tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

                if use_cuda:
                    one_hot_tail_seq = one_hot_tail_seq.cuda()
                    if bid_num == 2:
                        one_hot_tail_seq_inv = one_hot_tail_seq_inv.cuda()

                for i in range(bid_num):
                    if not i:
                        loss_e, loss_static, loss_cl = model.get_loss(history_glist, histroy_data, static_graph,
                                                                      one_hot_tail_seq, use_cuda, history_g, que_pair,
                                                                      all_tail_seq)
                    else:
                        loss_e, loss_static, loss_cl = model.get_loss(history_glist, inv_histroy_data, static_graph,
                                                                      one_hot_tail_seq_inv, use_cuda, history_g,
                                                                      que_pair_inv, all_tail_seq)
                    loss = loss_e + loss_static + loss_cl
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_static.append(loss_static.item())
                    losses_cl.append(loss_cl.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-static-contrastive:{:.4f}-{:.4f}-{:.4f}| Model {} ".format(
                epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_static), np.mean(losses_cl), model_name, ))
            # validation
            if epoch + 1:
                (mrr_raw, mrr_filter, hit_result_raw, hit_result_filter) = test(model, train_list, valid_list, num_rels,
                                                                                num_nodes, use_cuda, all_ans_list_valid,
                                                                                model_state_file, static_graph,
                                                                                valid_times, history_val_time_nogt,
                                                                                mode="train", bid_num=bid_num, )
                if mrr_raw < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_raw
                    torch.save({"state_dict":model.state_dict(), "epoch":epoch}, model_state_file)
        mrr_raw, mrr_filter, hit_result_raw, hit_result_filter = test(model, train_list + valid_list, test_list,
                                                                      num_rels, num_nodes, use_cuda, all_ans_list_test,
                                                                      model_state_file, static_graph, test_times,
                                                                      history_test_time_nogt, mode="test",
                                                                      bid_num=bid_num, )
        return mrr_raw, mrr_filter, hit_result_raw, hit_result_filter


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="TIRGN")

    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--batch-size", type=int, default=128, help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--test", action="store_true", default=False, help="load stat from dir and directly test", )
    parser.add_argument("--run-analysis", action="store_true", default=False, help="print log info")
    parser.add_argument("--run-statistic", action="store_true", default=False, help="statistic the result", )
    parser.add_argument("--multi-step", action="store_true", default=False,
                        help="do multi-steps inference without ground truth", )
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth", )
    parser.add_argument("--add-static-graph", action="store_true", default=False, help="use the info of static graph", )
    parser.add_argument("--add-rel-word", action="store_true", default=False, help="use words in relaitons", )
    parser.add_argument("--relation-evaluation", action="store_true", default=False,
                        help="save model accordding to the relation evalution", )

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1, help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7, help="weight of entity prediction task", )
    parser.add_argument("--discount", type=float, default=1, help="discount of weight of static constraint", )
    parser.add_argument("--angle", type=int, default=10, help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none", help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--skip-connect", action="store_true", default=False,
                        help="whether to use skip connect in a RGCN Unit", )
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub", help="opn of compgcn")
    parser.add_argument("--n-bases", type=int, default=100, help="number of weight blocks for each relation", )
    parser.add_argument("--n-basis", type=int, default=100, help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--self-loop", action="store_true", default=True,
                        help="perform layer normalization in every layer of gcn ", )
    parser.add_argument("--layer-norm", action="store_true", default=False,
                        help="perform layer normalization in every layer of gcn ", )
    parser.add_argument("--relation-prediction", action="store_true", default=False,
                        help="add relation prediction loss", )
    parser.add_argument("--entity-prediction", action="store_true", default=False, help="add entity prediction loss", )
    parser.add_argument("--split_by_relation", action="store_true", default=False, help="do relation prediction", )

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of minimum training epochs on each time step", )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20, help="perform evaluation every n epochs", )

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse", help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10, help="history length")
    parser.add_argument("--test-history-len", type=int, default=20, help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1, help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="perform grid search for best configuration", )
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use", )
    parser.add_argument("--num-k", type=int, default=500, help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3, help="history rate")
    parser.add_argument("--rel-weight", type=float, default=0, help="rel weight")
    parser.add_argument("--save", type=str, default="one", help="number of save")
    parser.add_argument("--att-type", type=str, default="multi", help="attention type")
    parser.add_argument("--tfa-weight", type=float, default=0, help="tfa weight")
    parser.add_argument("--use-cl", action="store_true", default=False, help="contrastive learing")
    parser.add_argument("--continual-test", action="store_true", default=False, help="online trainning")
    args = parser.parse_args()
    mrr_raw, mrr_filter, hit_result_raw, hit_result_filter = run_experiment(args)
