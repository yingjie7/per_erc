from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import torch
import numpy as np
from torch import nn
 
from torch.utils.data import DataLoader
import torch
import yaml

from transformers import BertConfig, AutoTokenizer, AutoModel, RobertaModel
import json
import random
import argparse
import torch.nn.functional as F
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
        for i, sample in enumerate(batch):
            # conversation padding 
            padded_conversation = sample['sentences_mixed_around'] + ["<pad_sentence>"]* (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            # label padding 
            labels += [int(label) for label in sample['labels']] + [-1]* (max_len_conversation - lengths[i])
        if len(labels)!= len(raw_sentences_flatten):
            print('len(labels)!= len(raw_sentences_flatten)')

        # utterance vectorizer
        # v_single_sentences = self._encoding(sample['sentences'])
        contextual_sentences_ids = self.tokenizer(raw_sentences_flatten,  padding='longest', max_length=512, truncation=True, return_tensors='pt')


        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, raw_sentences) 


# init model = PE (position encoding) layer + 
class PositionalEncoding(nn.Module):

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

        self.model = AutoModel.from_pretrained(model_configs.pre_trained_model_name) #  a pretraied Roberta model
        for i in range(self.model_configs.froze_bert_layer):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False

        d_model = self.model.config.hidden_size
        
        # ===================================
        # PUSH YOUR CODE HERE 
        # this is model architecture init process 
        nhead = 8
        num_layer = 1
        self.context_modeling = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layer)  # using transformer model in here 
        self.pos_encoding = PositionalEncoding(d_model)  # position encoding => for utterance positions (e.g., u_1, u_2, ...) in conversation. 
        # ===================================

        self.dropout_layer = nn.Dropout(model_configs.dropout)
        self.output_layer_context = nn.Linear(d_model, model_configs.num_labels)
        self.output_layer = nn.Linear(d_model, model_configs.num_labels)
        self.softmax_layer = nn.Softmax(dim=1)
        self.loss_computation = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1) # label_smoothing

        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    def training_step(self, batch, batch_idx, return_y_hat=False):
        input_ids, labels, padding_utterance_masked, raw_sentences = batch
        n_conversation, len_longest_conversation = padding_utterance_masked.shape[0], padding_utterance_masked.shape[1]

        #
        # ===================================
        # PUSH YOUR CODE HERE 
        # output requirements: 
        # - sentence_vectors: bert encoding for all utterances in all conversations  => shape = (n_conversation*len_longest_conversation, hiden_size), where n_conversation is batch_size
        # - u_vector_fused_by_context: utterance vectors fused by context using self-attention mechanism (Transformer) => shape = (n_conversation*len_longest_conversation, hiden_size)
        # ===================================
        sentence_vectors = self.model(**input_ids)[1]

        # sentence_vectors = self.model(** contextual_sentences_ids)[1]

        sentence_vectors_with_convers_shape = sentence_vectors.reshape(n_conversation, len_longest_conversation, -1)

        u_vector_fused_by_context = self.context_modeling(sentence_vectors_with_convers_shape +  self.pos_encoding(sentence_vectors_with_convers_shape), src_key_padding_mask=padding_utterance_masked)

        u_vector_fused_by_context = u_vector_fused_by_context.reshape(n_conversation*len_longest_conversation, -1)

        # ===================================

        # 
        # combine global context (u_vector_fused_by_context) and local context (sentence_vectors)z
        y_hat = self.output_layer_context(self.dropout_layer(u_vector_fused_by_context)) + self.output_layer(self.dropout_layer(sentence_vectors))
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
        v_multi_conversation, labels, padding_utterance_masked, raw_sentences = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        predictions = torch.argmax(y_hat, dim=1)

        self.validation_step_outputs.append({'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels})
        self.log('valid/loss', loss)

        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels}
    
    def _eval_epoch_end(self, batch_parts):
        predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
        labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
        f1_weighted = f1_score(
            labels.cpu(),
            predictions.cpu(),
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
    parser.add_argument("--max_ep", help="max epoch", type=int, default=5)
    parser.add_argument("--data_name_pattern", help="data_name_pattern", type=str, default="iemocap.{}window2.json")
    parser.add_argument("--pre_trained_model_name", help="pre_trained_model_name", type=str, default="roberta-large")
    parser.add_argument("--froze_bert_layer", help="froze_bert_layer", type=int, default=0)
    options = parser.parse_args()


    # 
    #  init random seed
    set_random_seed(7)
    data_folder= "/home/s2220429/deeplearning_tutorial/data2/iemocap_raw/"

    # 
    # Label counting
    data_name_pattern = options.data_name_pattern
    train_data = json.load(open(f'{data_folder}/{data_name_pattern.format("train")}'))
    all_labels = []
    for sample in train_data:
        all_labels+=  sample['labels']
    # count label 
    options.num_labels = len(set(all_labels))
    
    # 
    # data loader
    # Load config from pretrained name or path 
    model_configs = options
    
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

    checkpoint_callback = ModelCheckpoint(dirpath="./", save_top_k=1, 
                                        auto_insert_metric_name=True, 
                                        mode="max", 
                                        monitor="valid/f1", 
                                        filename=model_configs.pre_trained_model_name+"-iemocap-{valid/f1:.2f}",
                                    #   every_n_train_steps=opts.ckpt_steps
                                        )
    

    # init trainer 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(max_epochs=model_configs.max_ep, 
                        accelerator="gpu", devices=1,
                        callbacks=[checkpoint_callback, lr_monitor],
                        default_root_dir="./", 
                        val_check_interval=0.5 # 10% epoch, run evaluate one time 
                        )
    
    # init model 
    model = EmotionClassifier(model_configs)
    
    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # test
    trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)
