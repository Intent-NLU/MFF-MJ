import math

import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import numpy as np
from .module import IntentClassifier, SlotClassifier
import torch.nn.functional as F



class BBA_CNN(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(BBA_CNN, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.filter_sizes = [2, 3, 3]
        self.num_filters = 128
        self.textcnn = text_CNN(num_filters=self.num_filters, filter_sizes=self.filter_sizes)


        self.slot_classifier = SlotClassifier(config.hidden_size + len(self.filter_sizes) * self.num_filters,
                                              self.num_slot_labels, args.dropout_rate)


        # intent解码器修改部分
        self.intent_decoder = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size),
            nn.LeakyReLU(args.alpha),
            nn.Linear(config.hidden_size, self.num_intent_labels),
        )

        # intent解码器修改部分
        self.G_encoder = Encoder(args)

        self.intent_decoder = nn.Sequential(
            nn.Linear(config.hidden_size // 2,
                      config.hidden_size // 2),
            nn.LeakyReLU(args.alpha),
            nn.Linear(config.hidden_size // 2, self.num_intent_labels),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, train_or_val):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        g_c = self.G_encoder(sequence_output, 768)
        logits = self.intent_decoder(g_c)
        intent_logits = logits


        hidden_states = outputs[2]  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        hidden = self.textcnn(cls_embeddings, 12, self.filter_sizes)
        shape = hidden.shape[0]
        h = torch.reshape(hidden, (shape, 1, hidden.shape[1]))
        h1 = h.repeat(1, 50, 1)
        slot_encoder = torch.cat((sequence_output, h1), dim=2)



        intent_process = [[] for i in range(logits.shape[0])]


        slot_logits = self.slot_classifier(slot_encoder)
        intent_loss=0
        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                if train_or_val == 'val':
                    intent_pred_index = (torch.sigmoid(logits) > self.args.threshold).nonzero()
                    intent_pred_index = intent_pred_index.detach().cpu().numpy()
                    pred_index = [[0 for j in range(intent_label_ids.shape[1])] for i in range(intent_label_ids.shape[0])]
                    for i in intent_pred_index:
                        intent_process[i[0]].append(i[1])
                        pred_index[i[0]][i[1]] = 1

                    intent_pred_index = pred_index
                    intent_pred_index = torch.from_numpy(np.array(intent_pred_index))
                    intent_loss_fct = nn.BCEWithLogitsLoss()
                    intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids)
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:

            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs
        if train_or_val == 'val':
            return outputs, intent_pred_index, intent_process
        else:
            return outputs


hidden_size = 768




class text_CNN(nn.Module):
    def __init__(self, num_filters, filter_sizes):
        super(text_CNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)

        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])

    def forward(self, x, encode_layer, filter_sizes):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1)  # [bs, channel=1, seq, hidden]
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size=(encode_layer - filter_sizes[i] + 1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, 3)  # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])


        return h_pool_flat




class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # Initialize an LSTM Encoder object.
        self.encoder = LSTMEncoder(
            self.args.word_embedding_dim,
            self.args.encoder_hidden_dim,
            self.args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.attention = SelfAttention(
            self.args.word_embedding_dim,
            self.args.attention_hidden_dim,
            self.args.attention_output_dim,
            self.args.dropout_rate
        )



        self.sentattention = UnflatSelfAttention(
            self.args.encoder_hidden_dim + self.args.attention_output_dim,
            self.args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):

        lstm_hiddens = self.encoder(word_tensor, seq_lens)
        attention_hiddens = self.attention(word_tensor)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        c = self.sentattention(hiddens, seq_lens)
        return c


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        dropout_text = self.__dropout_layer(embedded_text)


        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(dropout_text)

        return lstm_hiddens


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        # max_len = max(lens)
        # for i, l in enumerate(lens):
        #     if l < max_len:
        #         scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x

class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """
        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor

