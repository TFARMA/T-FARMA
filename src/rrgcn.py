import numpy as np
import torch.nn as nn

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.decoder import *
from src.model import BaseRGCN


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(
                self.h_dim,
                self.h_dim,
                self.num_rels,
                self.num_bases,
                activation=act,
                dropout=self.dropout,
                self_loop=self.self_loop,
                skip_connect=sc,
                rel_emb=self.rel_emb,
            )
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata["id"].squeeze()
            g.ndata["h"] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
                if i == 0:
                    first_output = g.ndata["h"]
            return g.ndata.pop("h"), first_output
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata["id"] = self.features
            node_id = g.ndata["id"].squeeze()
            g.ndata["h"] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop("h")


class RecurrentRGCN(nn.Module):
    def __init__(
            self,
            decoder_name,
            encoder_name,
            num_ents,
            num_rels,
            num_static_rels,
            num_words,
            num_times,
            time_interval,
            h_dim,
            opn,
            history_rate,
            sequence_len,
            num_bases=-1,
            num_basis=-1,
            temperature=0.07,
            num_hidden_layers=1,
            dropout=0,
            self_loop=False,
            skip_connect=False,
            layer_norm=False,
            input_dropout=0,
            hidden_dropout=0,
            feat_dropout=0,
            aggregation="cat",
            weight=1,
            discount=0,
            angle=0,
            use_static=False,
            entity_prediction=False,
            relation_prediction=False,
            use_cuda=False,
            gpu=0,
            analysis=False,
            att_type="",
            use_cl=False,
            bid_num=1,
            rel_weight=0.1,
            decay_rate = 0
    ):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.att_type = att_type
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.temp = temperature
        self.use_cl = use_cl
        self.bid_num = bid_num
        self.rel_weight = rel_weight
        self.decay_rate = decay_rate

        self.w1 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * bid_num, self.h_dim), requires_grad=True).float()
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(self.num_ents, self.h_dim), requires_grad=True).float()

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.weight_t3 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.weight_t4 = nn.parameter.Parameter(torch.randn(1, self.h_dim))

        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.bias_t3 = nn.parameter.Parameter(torch.randn(1, self.h_dim))
        self.bias_t4 = nn.parameter.Parameter(torch.randn(1, self.h_dim))

        torch.nn.init.xavier_normal_(self.emb_rel)
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                self.h_dim,
                self.h_dim,
                self.num_static_rels * bid_num,
                num_bases,
                activation=F.rrelu,
                dropout=dropout,
                self_loop=False,
                skip_connect=False,
            )
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(
            num_ents,
            h_dim,
            h_dim,
            num_rels * bid_num,
            num_bases,
            num_basis,
            num_hidden_layers,
            dropout,
            self_loop,
            skip_connect,
            encoder_name,
            self.opn,
            self.emb_rel,
            use_cuda,
            analysis,
        )

        self.rgcn_global = RGCNCell(
            num_ents,
            h_dim,
            h_dim,
            num_rels * bid_num,
            num_bases,
            num_basis,
            num_hidden_layers,
            dropout,
            self_loop,
            skip_connect,
            encoder_name,
            self.opn,
            self.emb_rel,
            use_cuda,
            analysis,
        )

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain("relu"))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain("relu"))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        self.projection_model = MLPLinear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim * 2, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            self.decoder_ob1 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)

    def forward(self, g_list, static_graph, use_cuda, sub_graph, query_mask):
        evolve_embs = []
        evolve_r_embs = []
        att_embs = []
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata["h"] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop("h")[: self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        his_r_emb = F.normalize(self.emb_rel)
        his_e_emb, _ = self.rgcn_global.forward(sub_graph.to(self.gpu), self.h, [self.emb_rel, self.emb_rel])
        his_e_emb = F.normalize(his_e_emb)

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = (
                torch.zeros(self.num_rels * self.bid_num, self.h_dim).float().cuda()
                if use_cuda
                else torch.zeros(self.num_rels * self.bid_num, self.h_dim).float()
            )
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]: span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            # using GRU for rel evolution 。
            x_input = torch.cat((self.emb_rel, x_input), dim=1)
            if i == 0:
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
            else:
                self.h_0 = self.relation_cell_1(x_input, self.h_0)

            # using timegate for rel evolution。
            # x_input = self.emb_rel + x_input
            # time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
            # self.h_0 = time_weight * x_input + (1 - time_weight) * self.emb_rel

            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h, first_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            first_h = F.normalize(first_h) if self.layer_norm else first_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h

            if self.att_type == "multi":  # multi head
                att_e_2 = F.softmax(self.w2(torch.tanh(query_mask + current_h)), dim=1)
                att_e_1 = F.softmax(self.w3(torch.tanh(query_mask + first_h)), dim=1)
                att_emb_1 = att_e_1 * self.h
                att_emb_2 = att_e_2 * self.h
                att_emb = self.w4(torch.concat([att_emb_1, att_emb_2], dim=1))
                # att_e = F.softmax(self.w2(query_mask + current_h), dim=1)
                # att_emb = F.softmax(self.w2(query_mask + current_h), dim=1)
            elif self.att_type == "ori":
                att_e = F.softmax(self.w2(query_mask + current_h), dim=1)
                att_emb = att_e * self.h
            else:  # dot attention
                att_e = F.softmax(torch.mm(query_mask, current_h.t()) / (self.h_dim ** 0.5), dim=-1)
                att_emb = torch.mm(att_e, self.h)
            att_embs.append(att_emb.unsqueeze(0))
            evolve_embs.append(self.h)
            evolve_r_embs.append(self.h_0)
        att_ent = torch.mean(torch.concat(att_embs, dim=0), dim=0)
        att_ent = F.normalize(att_ent)
        if self.att_type:
            pre_emb = att_ent + evolve_embs[-1]
        else:
            pre_emb = evolve_embs[-1]
        pre_emb = F.normalize(pre_emb) if self.layer_norm else pre_emb

        return (pre_emb, static_emb, self.h_0, evolve_embs, evolve_r_embs, his_e_emb, his_r_emb)

    def predict(
            self,
            test_graph,
            static_graph,
            test_triplets,
            entity_history_vocabulary,
            use_cuda,
            history_g,
            que_pair,
            all_tail_seq,
            tail_seq,
    ):
        self.use_cuda = use_cuda
        with torch.no_grad():
            all_triples = test_triplets
            uniq_e, r_len, r_idx = que_pair
            temp_r = self.emb_rel[r_idx]
            e_input = (
                torch.zeros(self.num_ents, self.h_dim).float().cuda()
                if use_cuda
                else torch.zeros(self.num_ents, self.h_dim).float()
            )
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]: span[1], :]
                if self.rel_weight:
                    counter_matrix = np.sum(
                        all_tail_seq[(r_idx[span[0]: span[1]] + self.num_rels * e_idx).cpu().numpy()], axis=1
                    )
                    counter_matrix = counter_matrix + 1
                    max_fre = np.max(counter_matrix)
                    min_fre = np.min(counter_matrix)
                    if max_fre != min_fre:
                        weight_matrix = 1 + (counter_matrix - max_fre) / (max_fre - min_fre) * self.rel_weight
                        weight_matrix = torch.tensor(weight_matrix).cuda() if use_cuda else torch.tensor(weight_matrix)
                        x = x * weight_matrix
                    x_mean = torch.sum(x, dim=0, keepdim=True)
                else:
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean
            query_mask = torch.zeros((self.num_ents, self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e]  # 实体所连的所有关系池化
            query_emb = self.w1(torch.concat([e1_emb, rel_emb], dim=1))
            query_mask[uniq_e] = query_emb
            embedding, static_emb, r_emb, evolve_embs, evolve_r_embs, his_e_emb, his_r_emb = self.forward(
                test_graph, static_graph, use_cuda, sub_graph=history_g, query_mask=query_mask
            )
            time_embs = self.get_init_time(all_triples)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)
            score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score_f = F.softmax(tail_seq, dim=1)
            return all_triples, torch.log(score + self.decay_rate * score_f)

    def get_loss(
            self,
            glist,
            triples,
            static_graph,
            entity_history_vocabulary,
            use_cuda,
            history_g,
            que_pair,
            all_tail_seq,
    ):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        all_triples = triples

        uniq_e, r_len, r_idx = que_pair
        temp_r = self.emb_rel[r_idx]
        e_input = (
            torch.zeros(self.num_ents, self.h_dim).float().cuda()
            if use_cuda
            else torch.zeros(self.num_ents, self.h_dim).float()
        )
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]: span[1], :]
            if self.rel_weight:
                counter_matrix = np.sum(
                    all_tail_seq[(r_idx[span[0]: span[1]] + self.num_rels * e_idx).cpu().numpy()], axis=1
                )
                max_fre = np.max(counter_matrix)
                min_fre = np.min(counter_matrix)
                if max_fre != min_fre:
                    weight_matrix = 1 + (counter_matrix - max_fre) / (max_fre - min_fre) * self.rel_weight
                    weight_matrix = torch.tensor(weight_matrix).cuda() if use_cuda else torch.tensor(weight_matrix)
                    x = x * weight_matrix
                x_mean = torch.sum(x, dim=0, keepdim=True)
            else:
                x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean
        query_mask = torch.zeros((self.num_ents, self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        e1_emb = self.dynamic_emb[uniq_e]
        rel_emb = e_input[uniq_e]  # 实体所连的所有关系池化
        query_emb = self.w1(torch.concat([e1_emb, rel_emb], dim=1))
        query_mask[uniq_e] = query_emb

        pre_emb, static_emb, r_emb, evolve_embs, evolve_r_embs, his_e_emb, his_r_emb = self.forward(
            glist, static_graph, use_cuda, sub_graph=history_g, query_mask=query_mask
        )

        time_embs = self.get_init_time(all_triples)
        score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
        score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
        score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
        scores_en = torch.log(score_en)
        loss_ent += F.nll_loss(scores_en, all_triples[:, 2])

        if self.att_type and self.use_cl:
            for index, evolve_emb in enumerate(evolve_embs):
                query = torch.concat([his_e_emb[all_triples[:, 0]], his_r_emb[all_triples[:, 1]]], dim=1)
                query2 = torch.concat([evolve_emb[all_triples[:, 0]], evolve_r_embs[index][all_triples[:, 1]]], dim=1)
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                loss_cl += self.get_loss_conv(x1, x2)

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = self.angle * math.pi / 180
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_static, loss_cl

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to("cuda")
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def get_loss_conv(self, ent1_emb, ent2_emb):
        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)
        # train_cl_loss = (loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)) / 2
        train_cl_loss = (
                                loss_fn(pred1 / self.temp, labels)
                                + loss_fn(pred2 / self.temp, labels)
                                + loss_fn(pred3 / self.temp, labels)
                                + loss_fn(pred4 / self.temp, labels)
                        ) / 4
        return train_cl_loss
