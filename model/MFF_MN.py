import math

import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel,AutoModelForPreTraining
from .module import IntentnumClassifier,IntentClassifier, SlotClassifier
import torch.nn.functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MFF_MN(BertPreTrainedModel):
    # class JointBERT(AutoModelForPreTraining):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(MFF_MN, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert



        self.G_encoder = Encoder(args)
        self.intent_decoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2,
                      config.hidden_size * 2),
            nn.LeakyReLU(args.alpha),
            nn.Linear(config.hidden_size * 2, self.num_intent_labels),
        )
        self.intentnum_classifier = IntentnumClassifier(config.hidden_size*2, self.num_intent_labels,
                                                        args.dropout_rate).to(device)
        self.slot_classifier = SlotClassifier(config.hidden_size * 2, self.num_slot_labels, args.dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, train_or_val,
                intent_num_batch):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        slot_encoder = sequence_output
        pooled_output = outputs[1]  # [CLS]

        g_hiddens, g_c = self.G_encoder(sequence_output, 768)
        s=g_c


        hidden_states = outputs[2]
        cls_embeddings = torch.tensor([])
        for i in range(1, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        avg_pooling = F.adaptive_avg_pool2d(cls_embeddings, (1, 768))
        max_pooling = F.adaptive_max_pool2d(cls_embeddings, (1, 768))
        pooling = 0.5 * avg_pooling + 0.5 * max_pooling

        pooling_zj = zhengjiao(pooling, sequence_output, pooling.shape[0])
        slot_encoder = torch.cat((sequence_output, pooling_zj), dim=2)

        poo = torch.reshape(pooling, (pooling.shape[0], -1))
        gc_zj = zhengjiao_gc(g_c, poo, g_c.shape[0])
        g_c = torch.cat((g_c, poo), dim=1)


        logits = self.intent_decoder(g_c)
        intent_logits = logits
        intent_num_logits = self.intentnum_classifier(g_c)
        slot_logits = self.slot_classifier(slot_encoder)
        intent_d = [[] for i in range(logits.shape[0])]


        pred_num = torch.argmax(intent_num_logits, dim=1)
        pred_num=pred_num+1

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                if self.args.order_type:
                    if train_or_val == 'val':
                        num_softmax = F.softmax(intent_num_logits)
                        num_argmax = torch.argmax(num_softmax, dim=1)
                        num_list = []

                        for i in range(logits.shape[0]):
                            _, indices = torch.topk(torch.sigmoid(logits[i]), k=num_argmax[i] + 1)
                            num_list.append(indices.cpu().numpy().tolist())

                        for i in range(logits.shape[0]):
                            _, indices = torch.topk(torch.sigmoid(logits[i]),k=num_argmax[i]+1)
                            num_list.append(indices.cpu().numpy().tolist())
                        pred_index = [[0 for j in range(intent_label_ids.shape[1])] for i in
                                      range(intent_label_ids.shape[0])]
                        for i in range(len(pred_index)):
                            for j in num_list[i]:
                                pred_index[i][j]=1
                        mid=pred_index
                        for pre in range(len(pred_index)):
                            intent_d[pre]=[i for i, e in enumerate(pred_index[pre]) if e != 0]

                        intent_pred_index = mid
                        intent_pred_index = torch.from_numpy(np.array(intent_pred_index))
                    intent_loss_fct = nn.BCEWithLogitsLoss()
                    intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids)

                    intent_num_loss = nn.CrossEntropyLoss()
                    num_loss = intent_num_loss(intent_num_logits, intent_num_batch.long().to(device))

                else:
                    intent_loss_fct = nn.CrossEntropyLoss()
                    intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels),
                                                  intent_label_ids.view(-1))
            total_loss += intent_loss
            total_loss += num_loss
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
        if self.args.order_type:
            if train_or_val == 'val':
                return outputs, intent_pred_index, intent_d,pred_num
            else:
                return outputs
        else:
            return outputs

hidden_size = 768




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
        # s=self.attention0(lstm_hiddens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        c = self.sentattention(hiddens, seq_lens)
        # return hiddens, c
        return hiddens, c,


class LSTMEncoder(nn.Module):


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

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        # packed_text = pack_padded_sequence(dropout_text, seq_lens.to('cpu'), batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(dropout_text)
        # padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

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


#mask_feature是不变的那个向量，input_feature是要变的向量
def zhengjiao(input_feature,mask_feature,batch_size):
    output=torch.tensor([])
    for feature,mask in zip(input_feature,mask_feature):
        output_temp=torch.tensor([])
        for m in mask:
            feature=feature.view(-1)
            mask_norm = torch.sqrt(torch.sum(m ** 2))
            project = (torch.dot(feature, m) / (mask_norm ** 2)) * m
            orthoganal = feature - project
            output_temp=torch.cat((output_temp,orthoganal.view(1,-1)))
        output=torch.cat((output,output_temp.view(1,50,-1)))
    #output=output.view(batch_size,-1)
    return output

def zhengjiao_gc(input_feature,mask_feature,batch_size):
    output=torch.tensor([])
    for feature,mask in zip(input_feature,mask_feature):
        output_temp=torch.tensor([])
        feature = feature.view(-1)
        mask_norm = torch.sqrt(torch.sum(mask ** 2))
        project = (torch.dot(feature, mask) / (mask_norm ** 2)) * mask
        orthoganal = feature - project
        output = torch.cat((output, orthoganal.view(1, -1)))
        #output=torch.cat((output,output_temp.view(1,50,-1)))
    #output=output.view(batch_size,-1)
    return output

def zhengjiao_simple(input_feature,mask_feature):
    mask_norm = torch.sqrt(torch.sum(mask_feature ** 2))
    project = (torch.dot(input_feature, mask_feature) / (mask_norm ** 2)) * mask_feature
    orthoganal = input_feature - project

    return orthoganal