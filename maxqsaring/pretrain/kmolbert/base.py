import torch
from torch import nn
import numpy as np


def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to('cuda')
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        super(MultiHeadAttention, self).__init__()
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layernorm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm.cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class BERT_atom_embedding_generator(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim,
                 use_atom=False):
        super(BERT_atom_embedding_generator, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_atom = use_atom
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        if self.use_atom:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model + self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
            self.fc_weight = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, input_ids, atom_mask):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_global = output[:, 0]
        h_atom = output[:, atom_mask]
        return h_global, h_atom


class K_BERT_WCL(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim,
                 use_atom=False):
        super(K_BERT_WCL, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_atom = use_atom
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        if self.use_atom:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model + self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
            self.fc_weight = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_global = output[:, 0]
        if self.use_atom:
            h_atom = output[:, 1:]
            h_atom_weight = self.fc_weight(h_atom)
            h_atom_weight_expand = h_atom_weight.expand(h_atom.size())
            h_atom_mean = (h_atom * h_atom_weight_expand).mean(dim=1)
            h_mol = torch.cat([h_global, h_atom_mean], dim=1)
        else:
            h_mol = h_global
        h_embedding = self.fc(h_mol)
        logits_global = self.classifier_global(h_embedding)
        return logits_global


def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


class K_BERT(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim,
                 atom_label_dim):
        super(K_BERT, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        self.fc_global = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.fc_atom = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, canonical_input_ids, aug_input_ids_1, aug_input_ids_2, aug_input_ids_3, aug_input_ids_4):
        canonical_output = self.embedding(canonical_input_ids)
        aug_output_1 = self.embedding(aug_input_ids_1)
        aug_output_2 = self.embedding(aug_input_ids_2)
        aug_output_3 = self.embedding(aug_input_ids_3)
        aug_output_4 = self.embedding(aug_input_ids_4)

        canonical_enc_self_attn_mask = get_attn_pad_mask(canonical_input_ids)
        aug_enc_self_attn_mask_1 = get_attn_pad_mask(aug_input_ids_1)
        aug_enc_self_attn_mask_2 = get_attn_pad_mask(aug_input_ids_2)
        aug_enc_self_attn_mask_3 = get_attn_pad_mask(aug_input_ids_3)
        aug_enc_self_attn_mask_4 = get_attn_pad_mask(aug_input_ids_4)

        for layer in self.layers:
            canonical_output = layer(canonical_output, canonical_enc_self_attn_mask)
            aug_output_1 = layer(aug_output_1, aug_enc_self_attn_mask_1)
            aug_output_2 = layer(aug_output_2, aug_enc_self_attn_mask_2)
            aug_output_3 = layer(aug_output_3, aug_enc_self_attn_mask_3)
            aug_output_4 = layer(aug_output_4, aug_enc_self_attn_mask_4)

        h_canonical_global = self.fc_global(canonical_output[:, 0])
        h_aug_global_1 = self.fc_global(aug_output_1[:, 0])
        h_aug_global_2 = self.fc_global(aug_output_2[:, 0])
        h_aug_global_3 = self.fc_global(aug_output_3[:, 0])
        h_aug_global_4 = self.fc_global(aug_output_4[:, 0])

        h_cos_1 = torch.cosine_similarity(canonical_output[:, 0], aug_output_1[:, 0], dim=1)
        h_cos_2 = torch.cosine_similarity(canonical_output[:, 0], aug_output_2[:, 0], dim=1)
        h_cos_3 = torch.cosine_similarity(canonical_output[:, 0], aug_output_3[:, 0], dim=1)
        h_cos_4 = torch.cosine_similarity(canonical_output[:, 0], aug_output_4[:, 0], dim=1)
        consensus_score = (torch.ones_like(h_cos_1) * 4 - h_cos_1 - h_cos_2 - h_cos_3 - h_cos_4) / 8
        logits_canonical_global = self.classifier_global(h_canonical_global)
        logits_global_aug_1 = self.classifier_global(h_aug_global_1)
        logits_global_aug_2 = self.classifier_global(h_aug_global_2)
        logits_global_aug_3 = self.classifier_global(h_aug_global_3)
        logits_global_aug_4 = self.classifier_global(h_aug_global_4)
        canonical_cos_score_matric = torch.abs(cos_similar(canonical_output[:, 0], canonical_output[:, 0]))
        diagonal_cos_score_matric = torch.eye(canonical_cos_score_matric.size(0)).float().cuda()
        different_score = canonical_cos_score_matric - diagonal_cos_score_matric
        logits_global = torch.cat((logits_canonical_global, logits_global_aug_1, logits_global_aug_2,
                                   logits_global_aug_3, logits_global_aug_4), 1)

        h_atom = self.fc_atom(canonical_output[:, 1:])
        h_atom_emb = h_atom.reshape([len(canonical_output) * (self.maxlen - 1), self.d_model])
        logits_atom = self.classifier_atom(h_atom_emb)
        return logits_global, logits_atom, consensus_score, different_score


def load_model_weights(model, args):
    if args['pretrain_layer'] == 1:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 2:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 3:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 4:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight',
                                 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight',
                                 'layers.3.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 5:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight',
                                 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight',
                                 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight',
                                 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight',
                                 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight',
                                 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight',
                                 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight',
                                 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight',
                                 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight',
                                 'layers.4.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 6:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight',
                                 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight',
                                 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight',
                                 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight',
                                 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight',
                                 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight',
                                 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight',
                                 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight',
                                 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight',
                                 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight',
                                 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight',
                                 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight',
                                 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight',
                                 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight',
                                 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight',
                                 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight',
                                 'layers.5.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 7:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight',
                                 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight',
                                 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight',
                                 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight',
                                 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight',
                                 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight',
                                 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight',
                                 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight',
                                 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight',
                                 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight',
                                 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight',
                                 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight',
                                 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight',
                                 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight',
                                 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight',
                                 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight',
                                 'layers.5.pos_ffn.layernorm.bias', 'layers.6.enc_self_attn.linear.weight',
                                 'layers.6.enc_self_attn.linear.bias', 'layers.6.enc_self_attn.layernorm.weight',
                                 'layers.6.enc_self_attn.layernorm.bias', 'layers.6.enc_self_attn.W_Q.weight',
                                 'layers.6.enc_self_attn.W_Q.bias', 'layers.6.enc_self_attn.W_K.weight',
                                 'layers.6.enc_self_attn.W_K.bias', 'layers.6.enc_self_attn.W_V.weight',
                                 'layers.6.enc_self_attn.W_V.bias', 'layers.6.pos_ffn.fc.0.weight',
                                 'layers.6.pos_ffn.fc.2.weight', 'layers.6.pos_ffn.layernorm.weight',
                                 'layers.6.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 8:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 9:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                 'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                 'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                 'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                 'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                 'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                 'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                 'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 10:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                 'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                 'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                 'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                 'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                 'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                 'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                 'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                 'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                 'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                 'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                 'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                 'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                 'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                 'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 11:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                 'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                 'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                 'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                 'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                 'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                 'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                 'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                 'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                 'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                 'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                 'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                 'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                 'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                 'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                 'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                 'layers.10.enc_self_attn.layernorm.weight',
                                 'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                 'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                 'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                 'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                 'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                 'layers.10.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 12:
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                 'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                 'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                 'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                 'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                 'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                 'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                 'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                 'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                 'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                 'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                 'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                 'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                 'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                 'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                 'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                 'layers.10.enc_self_attn.layernorm.weight',
                                 'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                 'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                 'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                 'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                 'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                 'layers.10.pos_ffn.layernorm.bias', 'layers.11.enc_self_attn.linear.weight',
                                 'layers.11.enc_self_attn.linear.bias', 'layers.11.enc_self_attn.layernorm.weight',
                                 'layers.11.enc_self_attn.layernorm.bias', 'layers.11.enc_self_attn.W_Q.weight',
                                 'layers.11.enc_self_attn.W_Q.bias', 'layers.11.enc_self_attn.W_K.weight',
                                 'layers.11.enc_self_attn.W_K.bias', 'layers.11.enc_self_attn.W_V.weight',
                                 'layers.11.enc_self_attn.W_V.bias', 'layers.11.pos_ffn.fc.0.weight',
                                 'layers.11.pos_ffn.fc.2.weight', 'layers.11.pos_ffn.layernorm.weight',
                                 'layers.11.pos_ffn.layernorm.bias']
    elif args['pretrain_layer'] == 'all_6layer':
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight',
                                 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight',
                                 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight',
                                 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight',
                                 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight',
                                 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight',
                                 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight',
                                 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight',
                                 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight',
                                 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight',
                                 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight',
                                 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight',
                                 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight',
                                 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight',
                                 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight',
                                 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight',
                                 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight',
                                 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight',
                                 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight',
                                 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight',
                                 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight',
                                 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight',
                                 'layers.5.pos_ffn.layernorm.bias', 'fc.1.weight', 'fc.1.bias', 'fc.3.weight',
                                 'fc.3.bias', 'classifier_global.weight', 'classifier_global.bias',
                                 'classifier_atom.weight', 'classifier_atom.bias']
    elif args['pretrain_layer'] == 'all_12layer':
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                 'embedding.norm.weight', 'embedding.norm.bias',
                                 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',

                                 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                 'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                 'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                 'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                 'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                 'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                 'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                 'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                 'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                 'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                 'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                 'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                 'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                 'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                 'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                 'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                 'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                 'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                 'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                 'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                 'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                 'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                 'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                 'layers.10.enc_self_attn.layernorm.weight',
                                 'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                 'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                 'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                 'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                 'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                 'layers.10.pos_ffn.layernorm.bias'
                                 'fc.1.weight', 'fc.1.bias', 'fc.3.weight', 'fc.3.bias', 'classifier_global.weight',
                                 'classifier_global.bias', 'classifier_atom.weight', 'classifier_atom.bias']
    else:
        raise ValueError

    pretrained_model = torch.load(args['pretrain_model'], map_location=torch.device('cpu'))
    # pretrained_model = torch.load(self.pretrained_model)
    # model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
    # model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
