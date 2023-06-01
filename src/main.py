from os import path
import os
import torch
import numpy as np
from torch import nn
 
from torch.utils.data import DataLoader
import yaml

from transformers import BertConfig, AutoTokenizer, AutoModel, RobertaModel
import json
import random
import argparse
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils import class_weight

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
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
    # def _encoding(self, sentences):
    #     input_ids = self.tokenizer(sentences,  padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    #     sentence_vectors = self.fine_tuned_model.model(**input_ids)[1] # get CLS of all sentences
    #     return sentence_vectors
    
    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []

        # masked tensor  
        lengths = [len(sample['sentences_mixed_around']) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor([[False]*l_i+ [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation,max_len_conversation)
        for i, sample in enumerate(batch):
            # conversation padding 
            padded_conversation = sample['sentences_mixed_around'] + ["<pad_sentence>"]* (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            # label padding 
            labels += [int(label) for label in sample['labels']] + [-1]* (max_len_conversation - lengths[i])

            # speaker
            intra_speaker_masekd= torch.BoolTensor(len(padded_conversation),len(padded_conversation))
            for j in  range(len(padded_conversation)):
                for k in  range(len(padded_conversation)):
                    gender_j = sample['genders'][j]
                    gender_k = sample['genders'][k]

                    if gender_j == gender_k:
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd

        if len(labels)!= len(raw_sentences_flatten):
            print('len(labels)!= len(raw_sentences_flatten)')

        # utterance vectorizer
        # v_single_sentences = self._encoding(sample['sentences'])
        contextual_sentences_ids = self.tokenizer(raw_sentences_flatten,  padding='longest', max_length=512, truncation=True, return_tensors='pt')
        cur_sentence_indexes = (contextual_sentences_ids['input_ids'] == self.separate_token_id).nonzero(as_tuple=True)[1].reshape(contextual_sentences_ids['input_ids'].shape[0], -1)[:, :2]
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids['input_ids'].shape).fill_(False)
        for i in range(contextual_sentences_ids['input_ids'].shape[0]):
            for j in range(contextual_sentences_ids['input_ids'].shape[1]):
                if  cur_sentence_indexes[i][0] <= j <= cur_sentence_indexes[i][1]:
                    cur_sentence_indexes_masked[i][j] = True

        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences) 


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
            self.intra_speaker_context  =nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layer)
            self.output_layer_intra = nn.Linear(d_model, model_configs.num_labels)

        # reduce orverfitting by dropout 
        self.dropout_layer = nn.Dropout(model_configs.dropout)

        # output layer 
        self.output_layer_context = nn.Linear(d_model, model_configs.num_labels)
        self.output_layer = nn.Linear(d_model, model_configs.num_labels)

        # softmax and loss 
        self.softmax_layer = nn.Softmax(dim=1)
        self.loss_computation = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1, weight=torch.Tensor(self.model_configs.class_weights)) # label_smoothing

        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    def training_step(self, batch, batch_idx, return_y_hat=False):
        input_ids, labels, padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences = batch
        n_conversation, len_longest_conversation = padding_utterance_masked.shape[0], padding_utterance_masked.shape[1]
        
        # 
        # Pretrained model forwarding to get hidden vector representation 
        # bert_out[0]: hidden states of `all words` for each sentence 
        # bert_out[1]: hidden states of `cls token` for each sentence
        bert_out = self.model(**input_ids)
        h_words_hidden_states = bert_out[0]

        # 
        # using average word representations instead of CLS token for sentence/utterance representation. For example:
        # input:
        #   - bert_out[0]: is hidden vector of all words of current sentence and local context, arround sentences (2 context sentences befor + 2 next sentences)  
        #   - cur_sentence_indexes_masked: is a masked of the words is in current sentence. 
        #       - for example: [False, False, True, True, True, False, False, False, False] => current sentence contains words: 2 3 4, and other words (words 0, 1, 5, 6, 7, 8) is the context 
        # output requirements: 
        #   - sentence_vectors: is the average of word vectors in curent sentence, for example: = 1/3 * (w2 + w3 + w4)
        sentence_vectors = torch.sum(h_words_hidden_states*cur_sentence_indexes_masked.unsqueeze(-1), dim=1) * (1/ torch.sum(cur_sentence_indexes_masked, dim=1)).unsqueeze(-1) 
        sentence_vectors_with_convers_shape = sentence_vectors.reshape(n_conversation, len_longest_conversation, -1)

        u_vector_fused_by_context = self.context_modeling(sentence_vectors_with_convers_shape +  self.pos_encoding(sentence_vectors_with_convers_shape), src_key_padding_mask=padding_utterance_masked)
        u_vector_fused_by_context = u_vector_fused_by_context.reshape(n_conversation*len_longest_conversation, -1)
        
        # 
        # combine global context (u_vector_fused_by_context) and local context (sentence_vectors)
        y_hat = self.output_layer_context(self.dropout_layer(u_vector_fused_by_context)) + self.output_layer(self.dropout_layer(sentence_vectors))

        # 
        #  intra speaker modeling 
        if model_configs.intra_speaker_context:

            utterance_vector_fused_by_speaker_history = sentence_vectors_with_convers_shape + 0 # for create a new tensor equal to output bert vector`fake_utterance_vector_from_bert
                                                                                                # original utterance vector list 
            for i_conversation in range(intra_speaker_masekd_all.shape[0]):
                # process for each conversation in batch data.

                # compute the intra-masked for each speaker
                intra_speaker_masked_all_users_one_conversation = torch.unique(intra_speaker_masekd_all[i_conversation], dim=0)
                n_speaker = intra_speaker_masked_all_users_one_conversation.shape[0]

                # get maximum the number of utterance for each speaker 
                n_utterance_each_speaker = []
                v_utterance_each_speaker = []
                for i_speaker in range(n_speaker):
                    i_speaker_mask = intra_speaker_masked_all_users_one_conversation[i_speaker]
                    v_i_speaker = sentence_vectors_with_convers_shape[i_conversation][i_speaker_mask]
                    n_utterance_each_speaker.append(v_i_speaker.shape[0])
                    v_utterance_each_speaker.append(v_i_speaker)
                max_n_utterance_each_speaker = max(n_utterance_each_speaker)

                # intra-padding for each speaker 
                for i_speaker in range(n_speaker):
                    if n_utterance_each_speaker[i_speaker] < max_n_utterance_each_speaker:
                        v_utterance_each_speaker[i_speaker] = F.pad(v_utterance_each_speaker[i_speaker], [0, 0, 0, max_n_utterance_each_speaker-n_utterance_each_speaker[i_speaker]])
                        
                tensor_all_speakers = torch.stack(v_utterance_each_speaker, dim=0)
                
                # learn intra speaker information based on sequence model 
                # h_words, (hn, cn) = self.speaker_history_model_by_lstm(v_all_speakers)                                        # for lstm architecture 
                h_words = self.speaker_history_model_by_lstm(tensor_all_speakers+self.pos_encoding(tensor_all_speakers))        # for transformer architecture 

                # put the intra information back to original utterance vector list 

                for i_speaker in range(n_speaker):
                    i_speaker_mask = intra_speaker_masked_all_users_one_conversation[i_speaker]
                    utterance_vector_fused_by_speaker_history[i_conversation][i_speaker_mask] += h_words[i_speaker][:n_utterance_each_speaker[i_speaker]]
    
            utterance_vector_fused_by_speaker_history = utterance_vector_fused_by_speaker_history.reshape(batch_size*len_longest_conversation, -1)

            # combine intra speaker context 
            y_hat = y_hat + self.output_layer_intra(self.dropout_layer(utterance_vector_fused_by_speaker_history))

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
        v_multi_conversation, labels, padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        predictions = torch.argmax(y_hat, dim=1)

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
        valid_f1 = self._eval_epoch_end(self.validation_step_outputs)
        self.log('valid/f1', valid_f1, prog_bar=True)
        self.log('hp_metric', valid_f1)
        self.validation_step_outputs.clear()
    def on_test_epoch_end(self):
        test_val = self._eval_epoch_end(self.test_step_outputs)
        self.log('test/f1', test_val, prog_bar=True)
        self.log('hp_metric', test_val)
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

    parser.add_argument("--batch_size", help="batch ", type=int, default=1) 
    parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-5)
    parser.add_argument("--max_ep", help="max epoch", type=int, default=30)
    parser.add_argument("--data_name_pattern", help="data_name_pattern", type=str, default="iemocap.{}window2.json")
    parser.add_argument("--log_dir", help="path of data log dir and trained model. This path is augmented by {dataset_name}", type=str, default="./trained_models")
    parser.add_argument("--pre_trained_model_name", help="pre_trained_model_name", type=str, default="roberta-large")
    parser.add_argument("--froze_bert_layer", help="froze_bert_layer", type=int, default=10)
    parser.add_argument("--intra_speaker_context", help="use information of intra speaker context", action="store_true", default=False)
    options = parser.parse_args()


    # 
    #  init random seed
    set_random_seed(7)
    data_folder= "/home/phuongnm/deeplearning_tutorial/src/SimpleNN/data/all_raw_data/"

    # 
    # Label counting
    dataset_name = options.data_name_pattern.split(".")[0]
    data_name_pattern = options.data_name_pattern
    train_data = json.load(open(f'{data_folder}/{data_name_pattern.format("train")}'))
    all_labels = []
    for sample in train_data:
        all_labels+=  sample['labels']
    # count label 
    options.num_labels = len(set(all_labels))

    # compute class weight - for imbalance data 
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.array(all_labels)), y=np.array(all_labels))
    unique = list(np.unique(np.array(all_labels)))
    labels_dict = dict([(i, e) for i, e in enumerate(list(np.bincount(np.array(all_labels))))])
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
    data_loader = BatchPreprocessor(bert_tokenizer)
    train_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('train')}")), batch_size=model_configs.batch_size, collate_fn=data_loader, shuffle=True)
    valid_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('valid')}")), batch_size=1, collate_fn=data_loader, shuffle=True)
    test_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('test')}")), batch_size=1, collate_fn=data_loader, shuffle=True)

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
    checkpoint_callback = ModelCheckpoint(dirpath=f"{model_configs.log_dir}", save_top_k=1, 
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
                        val_check_interval=0.5 if  'dailydialog' not in options.data_name_pattern else 0.1 # 50%/10% epoch - freq time to run evaluate 
                        )
    
    # init model 
    model = EmotionClassifier(model_configs)
    
    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # test
    trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)