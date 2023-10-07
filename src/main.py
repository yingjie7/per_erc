import collections
from os import path
import os
import pickle
from torch.nn import functional as F
import torch
import numpy as np
from torch import nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import random
import argparse
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import argparse
import math
# =====================

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class BatchPreprocessor(object): 
    def __init__(self, tokenizer, model_configs=None, data_type='valid') -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name  =  model_configs.data_name_pattern.split(".")[0]
        self.window_ct = model_configs.window_ct
        self.model_configs = model_configs
        
        if model_configs.llm_context:
            path_llm_context_vect=f'{model_configs.data_folder}/llm_vectors/{model_configs.llm_context_file_pattern.format(data_type)}'
            self.llm_context_vect = pickle.load(open(path_llm_context_vect, 'rb'))
            
            check_key = list(self.llm_context_vect.keys())[0]
            self.llm_context_vect_dim = self.llm_context_vect[check_key][0].shape[-1]
            
        if model_configs.speaker_description:
            path_file = f'{model_configs.data_folder}/llm_vectors/{model_configs.speaker_description_file_pattern.format(data_type)}'
            self.speaker_description = json.load(open(path_file, 'rt'))
            
    
    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data
              
    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                        "Ses01": {"F": "Mary", "M": "James"},
                        "Ses02": {"F": "Patricia", "M": "John"},
                        "Ses03": {"F": "Jennifer", "M": "Robert"},
                        "Ses04": {"F": "Linda", "M": "Michael"},
                        "Ses05": {"F": "Elizabeth", "M": "William"},
                    }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1) 
            return f"SPEAKER_{gender_idx}"
        elif data_name=='dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"
        
    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_sentences = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                if i == j:
                    tmp_s += " </s>"
                tmp_s +=  f" {self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    tmp_s += " </s>"
            new_sentences.append(tmp_s)
        return new_sentences
    
    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []
        llm_context_vectors = []

        # masked tensor  
        lengths = [len(sample['sentences']) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor([[False]*l_i+ [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation,max_len_conversation)
        for i, sample in enumerate(batch):
            sentences_mixed_arround = self.sentence_mixed_by_surrounding(sample['sentences'], 
                                                                        around_window=self.window_ct, 
                                                                        s_id=sample['s_id'], 
                                                                        genders=sample['genders'],
                                                                        data_name=self.dataset_name)
            # conversation padding 
            padded_conversation = sentences_mixed_arround + ["<pad_sentence>"]* (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            # label padding 
            labels += [int(label) for label in sample['labels']] + [-1]* (max_len_conversation - lengths[i])

            # speaker
            intra_speaker_masekd= torch.BoolTensor(len(padded_conversation),len(padded_conversation)).fill_(False)
            for j in  range(len( sample['genders'])):
                for k in  range(len( sample['genders'])):
                    gender_j = sample['genders'][j]
                    gender_k = sample['genders'][k]

                    if gender_j == gender_k:
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd

        if len(labels)!= len(raw_sentences_flatten):
            print('len(labels)!= len(raw_sentences_flatten)')
        
        raw_sentences_flatten_spdesc = raw_sentences_flatten + []
        n_spdesc = 0
        
        # ======== 
        # setting for using speaker description 
        sp_characteristic_word_ids = None
        if self.model_configs.speaker_description:    
            speaker_descriptions = []
            for i, sample in enumerate(batch):
                speaker_descriptions += self.speaker_description[sample['s_id']] 
            all_speaker_descriptions = list(set(speaker_descriptions))
            n_spdesc = len(all_speaker_descriptions)
            raw_sentences_flatten_spdesc = all_speaker_descriptions + raw_sentences_flatten
            
            all_speaker_descriptions_idx = [all_speaker_descriptions.index(e) for e in speaker_descriptions]
            sp_characteristic_word_ids = {'len_sp_desc': len(all_speaker_descriptions), 'all_speaker_descriptions': all_speaker_descriptions_idx}
        # ======== 
        
        # utterance vectorizer
        # v_single_sentences = self._encoding(sample['sentences'])
        contextual_sentences_ids = self.tokenizer(raw_sentences_flatten_spdesc,  padding='longest', max_length=512, truncation=True, return_tensors='pt')
        sent_indices, word_indices = torch.where(contextual_sentences_ids['input_ids'][n_spdesc:] == self.separate_token_id)
        gr_sent_indices = [[] for e in range(len(raw_sentences_flatten))]
        for sent_idx, w_idx in zip (sent_indices, word_indices):
            gr_sent_indices[sent_idx].append(w_idx.item())
            
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids['input_ids'][n_spdesc:].shape).fill_(False)
        for i in range(contextual_sentences_ids['input_ids'][n_spdesc:].shape[0]):
            if raw_sentences_flatten[i] =='<pad_sentence>':
                cur_sentence_indexes_masked[i][gr_sent_indices[i][0]] = True
                continue
            for j in range(contextual_sentences_ids['input_ids'][n_spdesc:].shape[1]):
                if  gr_sent_indices[i][0] <= j <= gr_sent_indices[i][1]:
                    cur_sentence_indexes_masked[i][j] = True
                    
        padding_word_masked = None
        llm_context_vectors = []
        if self.model_configs.llm_context:
            for i, sample in enumerate(batch):
                for s in range(len(sample['sentences'])):
                    llm_context_vectors.append(self.llm_context_vect[sample['s_id']][s])
            llm_context_vectors  = [e.to(torch.float) for e in llm_context_vectors]
            
            if self.model_configs.llm_aggregate_method == 'accwr_selfattn':
                max_sent_len  = max([e.shape[0] for e in llm_context_vectors])
                padding_word_masked = torch.BoolTensor([[False]*e.shape[0]+ [True]*(max_sent_len - e.shape[0]) for e in llm_context_vectors])
                llm_context_vectors  = torch.stack([F.pad(e, (0,0,0,max_sent_len - e.shape[0]), "constant", 0)   for e in llm_context_vectors], dim=0)
            elif self.model_configs.llm_aggregate_method == 'accwr_average':
                llm_context_vectors  = [torch.sum(e, dim=0) / e.shape[0] for e in llm_context_vectors]
                llm_context_vectors  = torch.stack(llm_context_vectors, dim=0)
            elif self.model_configs.llm_aggregate_method == 'cls':
                llm_context_vectors  = torch.stack(llm_context_vectors, dim=0)
        
            llm_context_vectors.requires_grad = False
        
        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, 
                llm_context_vectors, padding_word_masked, sp_characteristic_word_ids, raw_sentences) 

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, batch_first=True):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_input, k_input, v_input, attn_mask=None, key_padding_mask=None):
        attention_mask_mt = None
        if attn_mask is not None:
            attention_mask_mt = torch.zeros(attn_mask.shape, device=attn_mask.device, requires_grad=False)
            attention_mask_mt.masked_fill_(attn_mask, -9999)

        mixed_query_layer = self.query(q_input)
        mixed_key_layer = self.key(k_input)
        mixed_value_layer = self.value(v_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask_mt.unsqueeze(0).reshape(attention_scores.shape) if attention_mask_mt is not None else attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores
    


class PositionalEncoding(nn.Module):
    """positional encoding to encode the order of sequence in transformer architecture 
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        return self.pe[:x.size(1)].transpose(0,1)


class EmotionClassifier(pl.LightningModule):
    def __init__(
        self, 
        model_configs
    ):
        """Initialize."""
        super().__init__() 

        self.save_hyperparameters(model_configs)
        self.model_configs = model_configs

        # init pretrained language model - RoBERTa 
        # froze 10 layers, only train 2 last layer 
        self.model = AutoModel.from_pretrained(model_configs.pre_trained_model_name) #  a pretraied Roberta model
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        for i in range(self.model_configs.froze_bert_layer):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False

        d_model = self.model.config.hidden_size
        
        # global context modeling 
        nhead = 8
        num_layer = 1
        self.context_modeling = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layer)  # using transformer model in here 
        self.pos_encoding = PositionalEncoding(d_model)  # position encoding => for utterance positions (e.g., u_1, u_2, ...) in conversation. 
        

        if model_configs.intra_speaker_context:
            # intra speaker modeling 
            # self.intra_speaker_context  =nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layer)
            # self.output_layer_intra = nn.Linear(d_model, model_configs.num_labels)
            self.intra_speaker_modeling =  BertSelfAttention(int(d_model/2), num_heads=nhead, dropout=model_configs.dropout, batch_first=True)    
            self.iq =   nn.Linear(d_model, int(d_model/2))   
            self.output_layer_intra = nn.Linear(int(d_model/2), model_configs.num_labels)

            self.inter_speaker_modeling =  BertSelfAttention(int(d_model/2), num_heads=nhead, dropout=model_configs.dropout, batch_first=True)   
            self.aq =   nn.Linear(d_model, int(d_model/2))  
            self.output_layer_inter = nn.Linear(int(d_model/2), model_configs.num_labels)

        # reduce orverfitting by dropout 
        self.dropout_layer = nn.Dropout(model_configs.dropout)

        # output layer 
        if model_configs.llm_context:
            # llm modeling
            llm_model = model_configs.llm_context_vect_dim
            if model_configs.llm_aggregate_method == 'accwr_selfattn':
                self.llm_attention_modeling = nn.MultiheadAttention(llm_model, num_heads=8, dropout=0.2, batch_first=True )
                self.llm_query = nn.Linear(llm_model,int(llm_model))
                self.llm_key = nn.Linear(llm_model,int(llm_model))
                self.llm_value = nn.Linear(llm_model,int(llm_model))
            self.output_llm_context = nn.Linear(llm_model, model_configs.num_labels)
        
        if model_configs.speaker_description:
            self.output_layer_sp_description= nn.Linear(d_model, model_configs.num_labels)
            
            
        self.output_layer_context = nn.Linear(d_model, model_configs.num_labels)
        self.output_layer = nn.Linear(d_model, model_configs.num_labels)

        # softmax and loss 
        self.softmax_layer = nn.Softmax(dim=1)
        self.loss_computation = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1, weight=torch.Tensor(self.model_configs.class_weights)) # label_smoothing

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

        # aggregate_sentence_h_state
        if self.model_configs.sent_aggregate_method == 'mlp':
            self.sentence_modeling = nn.Linear(d_model,d_model)
            self.sentence_tanh = nn.Tanh()
        if self.model_configs.sent_aggregate_method == 'lstm': 
            self.sentence_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=2, 
                    dropout=0.2, batch_first=True, bidirectional=True)

    def aggregate_sentence_h_state(self, cls_hidden_state, h_words_hidden_states, cur_sentence_indexes_masked):
        if self.model_configs.sent_aggregate_method == 'average':  
            # using average word representations instead of CLS token for sentence/utterance representation. For example:
            # input:
            #   - bert_out[0]: is hidden vector of all words of current sentence and local context, arround sentences (2 context sentences befor + 2 next sentences)  
            #   - cur_sentence_indexes_masked: is a masked of the words is in current sentence. 
            #       - for example: [False, False, True, True, True, False, False, False, False] => current sentence contains words: 2 3 4, and other words (words 0, 1, 5, 6, 7, 8) is the context 
            # output requirements: 
            #   - sentence_vectors: is the average of word vectors in curent sentence, for example: = 1/3 * (w2 + w3 + w4)
            
            sentence_vectors = torch.sum(h_words_hidden_states*cur_sentence_indexes_masked.unsqueeze(-1), dim=1) * (1/ torch.sum(cur_sentence_indexes_masked, dim=1)).unsqueeze(-1) 
            return sentence_vectors
        elif self.model_configs.sent_aggregate_method == 'mlp':      
            # 
            # using average word representations + a linear layer + a tanh activation: 
            # h_words => average => linear => tanh
            # CODE HERE to compute the Linear layer over average of words.
            sentence_vectors_avg = torch.sum(h_words_hidden_states*cur_sentence_indexes_masked.unsqueeze(-1), dim=1) * (1/ torch.sum(cur_sentence_indexes_masked, dim=1)).unsqueeze(-1) 
            sentence_vectors_output = self.sentence_modeling(sentence_vectors_avg)
            sentence_vectors = self.sentence_tanh(sentence_vectors_output)
            return sentence_vectors
        elif self.model_configs.sent_aggregate_method == 'lstm':      
            # 
            # using LSTM for modeling sentence vector based on word vectors 
            h_words, (hn, cn) = self.sentence_lstm(h_words_hidden_states*cur_sentence_indexes_masked.unsqueeze(-1))
            sentence_vectors = torch.sum(h_words*cur_sentence_indexes_masked.unsqueeze(-1), dim=1) * (1/ torch.sum(cur_sentence_indexes_masked, dim=1)).unsqueeze(-1) 

            return sentence_vectors
        elif self.model_configs.sent_aggregate_method == 'cls':      
            # using CLS token hidden state for sentence vector representation 
            # h_cls
            sentence_vectors = cls_hidden_state
            return sentence_vectors

    def training_step(self, batch, batch_idx, return_y_hat=False):
        input_ids, labels, padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, llm_context_vectors,padding_word_masked, sp_description_word_ids, raw_sentences = batch
        n_conversation, len_longest_conversation = padding_utterance_masked.shape[0], padding_utterance_masked.shape[1]
        
        # 
        # Pretrained model forwarding to get hidden vector representation 
        # bert_out[0]: hidden states of `all words` for each sentence 
        # bert_out[1]: hidden states of `cls token` for each sentence
        bert_out = self.model(**input_ids)
        
        n_speaker_description = 0 
        if model_configs.speaker_description:
            all_speaker_descriptions = sp_description_word_ids['all_speaker_descriptions']
            n_speaker_description = sp_description_word_ids['len_sp_desc']
            sp_description_vector = bert_out.pooler_output [:n_speaker_description]

        # construct sentence vector based on word representations  
        sentence_vectors = self.aggregate_sentence_h_state(bert_out.pooler_output[n_speaker_description:], bert_out.last_hidden_state[n_speaker_description:], cur_sentence_indexes_masked)
        sentence_vectors_with_convers_shape = sentence_vectors.reshape(n_conversation, len_longest_conversation, -1)

        # learn global relationship 
        u_vector_fused_by_context = self.context_modeling(sentence_vectors_with_convers_shape +  self.pos_encoding(sentence_vectors_with_convers_shape), src_key_padding_mask=padding_utterance_masked)
        u_vector_fused_by_context = u_vector_fused_by_context.reshape(n_conversation*len_longest_conversation, -1)
        
        # combine global context (u_vector_fused_by_context) and local context (sentence_vectors)
        y_hat = self.output_layer_context(self.dropout_layer(u_vector_fused_by_context)) + self.output_layer(self.dropout_layer(sentence_vectors))

        #  intra speaker modeling 
        if self.model_configs.intra_speaker_context:

            # padding for masked utterance
            new_padding_utterance_masked = padding_utterance_masked.unsqueeze(-1).repeat(1,1,intra_speaker_masekd_all.shape[-1])
            new_padding_utterance_masked.fill_(False)
            for i_batch in range(padding_utterance_masked.shape[0]):
                count_masked_utterance = sum(padding_utterance_masked[i_batch])
                if count_masked_utterance > 0:
                    new_padding_utterance_masked[i_batch, -count_masked_utterance:, -count_masked_utterance:]= True
            
            inter_speaker_masekd_all = ((~intra_speaker_masekd_all) | new_padding_utterance_masked)
            new_intra_speaker_masekd_all = (intra_speaker_masekd_all | new_padding_utterance_masked)

            # intra and inter speaker dependencies  
            i_vector = self.iq(sentence_vectors_with_convers_shape)            
            u_vector_fused_by_intra_speaker, _ = self.intra_speaker_modeling(i_vector,  i_vector,  i_vector, 
                                                                        attn_mask=(inter_speaker_masekd_all).repeat(8,1,1), key_padding_mask=padding_utterance_masked) 
            u_vector_fused_by_intra_speaker = u_vector_fused_by_intra_speaker.reshape(n_conversation*len_longest_conversation, -1)

            a_vector = self.aq(sentence_vectors_with_convers_shape)
            u_vector_fused_by_inter_speaker, _ = self.inter_speaker_modeling(a_vector,  a_vector,  a_vector, 
                                                                        attn_mask=new_intra_speaker_masekd_all.repeat(8,1,1), key_padding_mask=padding_utterance_masked)
            u_vector_fused_by_inter_speaker = u_vector_fused_by_inter_speaker.reshape(n_conversation*len_longest_conversation, -1)

            y_hat = y_hat + self.output_layer_intra(self.dropout_layer(u_vector_fused_by_intra_speaker)) + self.output_layer_inter(self.dropout_layer(u_vector_fused_by_inter_speaker))

        if model_configs.llm_context:
            # llm self_attention modeling
            if model_configs.llm_aggregate_method == 'accwr_selfattn':
                llm_vectors_with_attention, attentions = self.llm_attention_modeling(self.llm_query(llm_context_vectors),
                                                                                    self.llm_key(llm_context_vectors),
                                                                                    self.llm_value(llm_context_vectors),
                                                                                    key_padding_mask=padding_word_masked
                                                                                    )
                llm_vectors_with_attention.masked_fill_(padding_word_masked.unsqueeze(-1), 0)
                llm_vectors_with_attention = torch.sum(llm_vectors_with_attention, dim=1) * (1/ torch.sum(~padding_word_masked, dim=1)).unsqueeze(-1)
                llm_context_vectors = llm_vectors_with_attention
                
            y_hat = y_hat + self.output_llm_context(self.dropout_layer(llm_context_vectors))
            
        if model_configs.speaker_description:
            sp_description_vector_extracted = torch.stack([sp_description_vector[idx] for idx in all_speaker_descriptions], dim=0)
            y_hat = y_hat + self.output_layer_sp_description(self.dropout_layer(sp_description_vector_extracted)) 

        # 
        # compute probabilities and loss 
        probabilities = self.softmax_layer(y_hat)

        loss = self.loss_computation(probabilities, labels)
        self.log('train/loss', loss)
        

        if return_y_hat:
            return loss, probabilities
        return loss

    def test_step(self, batch, batch_idx):
        out= self.validation_step( batch, batch_idx)
        self.test_step_outputs.append(out)
        return out
    
    def validation_step(self, batch, batch_idx):
        labels= batch[1]
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        _ = torch.argmax(y_hat, dim=1)

        loss, y_hat, labels = loss.detach().cpu(), y_hat.detach().cpu(), labels.detach().cpu()
        self.validation_step_outputs.append({'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels})
        self.log('valid/loss', loss)

        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels}
    
    def _eval_epoch_end(self, batch_parts):
        predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
        labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)

        # remove padding labels 
        labels, predictions = labels.cpu().tolist(), predictions.cpu().tolist()
        len_label = len(labels)
        for i in range(len_label):
            if labels[len_label-i-1] == -1:
                labels.pop(len_label-i-1)
                predictions.pop(len_label-i-1)
        
        if 'dailydialog' in self.model_configs.data_name_pattern:
            # 0 happy, 1 neutral, 2 anger, 3 sad, 4 fear, 5 surprise, 6 disgust 
            f1_weighted = f1_score(
                labels,
                predictions,
                average="micro",
                labels=[0,2,3,4,5,6] # do not count the neutral class in the dailydialog dataset
            )
        else:
            f1_weighted = f1_score(
                labels,
                predictions,
                average="weighted",
            )
            
        return f1_weighted*100
    
    def on_validation_epoch_end(self):
        f_val = self._eval_epoch_end(self.validation_step_outputs)
        self.log_dict({'valid/f1': f_val, 'hp_metric': f_val}, 
                    prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
    def on_test_epoch_end(self):
        f_val = self._eval_epoch_end(self.test_step_outputs)
        self.log_dict({'test/f1': f_val, 'hp_metric': f_val}, 
                    prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr = 1e-5, weight_decay=0.001)  
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                            betas=(0.9, 0.98),  # according to RoBERTa paper
                            lr=self.model_configs.lr,
                        eps=1e-06, weight_decay=0.001)
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        num_gpus = 1
        max_ep=self.model_configs.max_ep
        t_total = (len(train_loader) // (1 * num_gpus) + 1) * max_ep
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.model_configs.lr, pct_start=float(0/t_total),
            final_div_factor=10000, steps_per_epoch=len(train_loader),
            total_steps=t_total, anneal_strategy='linear'
        ) 
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "monitor": "train/loss"}]

    

#  
# main process 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", help="path to data folder", type=str, default='/home/s2220429/per_erc/data/all_raw_data') 
    parser.add_argument("--n_best_checkpoint", help="number of best checkpoints to save", type=int, default=1) 
    parser.add_argument("--accumulate_grad_batches", help="number of accumulate batch for each grandient update step", type=int, default=2) 
    parser.add_argument("--seed", help="seed value ", type=int, default=7) 
    parser.add_argument("--batch_size", help="batch ", type=int, default=1) 
    parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-5)
    parser.add_argument("--max_ep", help="max epoch", type=int, default=30)
    parser.add_argument("--sent_aggregate_method", help="sent_aggregate_method in  {average, cls, lstm, mlp}", type=str, default="average")
    parser.add_argument("--data_name_pattern", help="data_name_pattern", type=str, default="iemocap.{}window2.json")
    parser.add_argument("--log_dir", help="path of data log dir and trained model. This path is augmented by {dataset_name}", type=str, default="./trained_models")
    parser.add_argument("--pre_trained_model_name", help="pre_trained_model_name", type=str, default="roberta-large")
    parser.add_argument("--froze_bert_layer", help="froze_bert_layer", type=int, default=10)
    parser.add_argument("--intra_speaker_context", help="use information of intra speaker context", action="store_true", default=False)
    
    parser.add_argument("--llm_context", help="use llm context or not",action="store_true", default=False)
    parser.add_argument("--llm_aggregate_method", help="method to incoporate llm context vector in {cls, accwr_selfattn, accwr_average}", type=str, default="accwr_average")
    parser.add_argument("--llm_context_file_pattern", help="llm context vector path", type=str, default='iemocap.{}_v2_5')
    
    parser.add_argument("--speaker_description", help="use llm context or not",action="store_true", default=False)
    parser.add_argument("--speaker_description_file_pattern", help="speaker description path file pattern", type=str, default="iemocap.{}_speaker_descriptions.json")
    
    parser.add_argument("--window_ct", help="number of context window", type=int, default=5)
    options = parser.parse_args()


    # 
    #  init random seed
    set_random_seed(options.seed)

    # 
    # Label counting
    dataset_name = options.data_name_pattern.split(".")[0]
    data_name_pattern = options.data_name_pattern
    train_data = BatchPreprocessor.load_raw_data(f'{options.data_folder}/{data_name_pattern.format("train")}')
    all_labels = []
    for sample in train_data:
        all_labels += [e for e in sample['labels'] if e != -1]
    # count label 
    options.num_labels = len(set(all_labels))

    # compute class weight - for imbalance data 
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.array(all_labels)), y=np.array(all_labels))
    unique = list(np.unique(np.array(all_labels)))
    labels_dict = collections.Counter(all_labels)
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(total/labels_dict[key])
        weights.append(score)
    class_weights = weights
    
    # 
    # data loader
    # Load config from pretrained name or path 
    model_configs = options
    model_configs.class_weights = class_weights
    
    batch_size = options.batch_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_configs.pre_trained_model_name)

    data_loader_valid = BatchPreprocessor(bert_tokenizer, model_configs=model_configs, data_type='valid')
    valid_loader = DataLoader(BatchPreprocessor.load_raw_data(f"{options.data_folder}/{data_name_pattern.format('valid')}"), batch_size=model_configs.batch_size, collate_fn=data_loader_valid, shuffle=False)
    
    data_loader_train = BatchPreprocessor(bert_tokenizer, model_configs=model_configs, data_type='train')
    train_loader = DataLoader(BatchPreprocessor.load_raw_data(f"{options.data_folder}/{data_name_pattern.format('train')}"), batch_size=model_configs.batch_size, collate_fn=data_loader_train, shuffle=True)
    
    data_loader_test = BatchPreprocessor(bert_tokenizer, model_configs=model_configs, data_type='test')
    test_loader = DataLoader(BatchPreprocessor.load_raw_data(f"{options.data_folder}/{data_name_pattern.format('test')}"), batch_size=model_configs.batch_size, collate_fn=data_loader_test, shuffle=False)

    if model_configs.llm_context:
        model_configs.llm_context_vect_dim = data_loader_train.llm_context_vect_dim
    
    for e in test_loader:
        print('First epoch data:')
        print('input data\n', e[0])
        print('label data\n',e[1])
        # print('padding mask data\n',e[2])
        break  
    print('train size', len(train_loader))
    print('test size',  len(test_loader))

    # 
    # create folder save log data
    model_configs.log_dir = f"{model_configs.log_dir}/{dataset_name}"
    if not path.exists(model_configs.log_dir):
        os.makedirs(model_configs.log_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=f"{model_configs.log_dir}", save_top_k=model_configs.n_best_checkpoint, 
                                        auto_insert_metric_name=True, 
                                        mode="max", 
                                        monitor="valid/f1", 
                                        filename=model_configs.pre_trained_model_name+f"-{dataset_name}"+"-{valid/f1:.2f}",
                                    #   every_n_train_steps=opts.ckpt_steps
                                        )
    

    # init trainer 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(max_epochs=model_configs.max_ep, 
                        accelerator="gpu", devices=1,
                        callbacks=[checkpoint_callback, lr_monitor],
                        default_root_dir=f"{model_configs.log_dir}",
                        accumulate_grad_batches=model_configs.accumulate_grad_batches,
                        val_check_interval=0.5 if  'dailydialog' not in options.data_name_pattern else 0.1 # 50%/10% epoch - freq time to run evaluate 
                        )
    
    # init model 
    model = EmotionClassifier(model_configs)
    
    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # test
    trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)