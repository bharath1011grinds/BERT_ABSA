import torch
import torch.nn as nn
from transformers import BertModel
from pathlib import Path
import json

import numpy as np
from collections import Counter

#Below block of code handles weight imbalances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2id = {'negative' : 0, 'neutral' : 1, 'positive' : 2}

PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DATA_DIR/"train.json"


with open(TRAIN_FILE) as f:
    train_data = json.load(f)

labels = [label2id[x["polarity"]] for x in train_data]
counts = Counter(labels)
total = sum(counts.values())

class_weights = torch.tensor(
    [total / counts[i] for i in range(3)],
    dtype=torch.float
).to(device)   


class BERTABSA(nn.Module):
    
    def __init__(self, num_labels=3, model_name = "bert-base-uncased"):
        super().__init__()

        #using the baseline 12 layered BERT transformer
        self.bert = BertModel.from_pretrained(model_name) 

        hidden_size = self.bert.config.hidden_size #hidden dimension - 768 for BERT
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_labels) #the final classification layer

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        '''Note: input ids, tokentype_ids will have a shape [Batch_size, seqlen] consisting of the ids corresponding to the embedding vectors.'''
        outputs = self.bert(input_ids, token_type_ids, attention_mask)


        last_state = outputs.last_hidden_state
        #returns the op of size B*seq_len*hidden_dim

        #extract the CLS node, 0th index by convention
        '''CLS node approach       
        cls_node = last_state[ :, 0, : ] #pytorch autocollapses the middle dimension and the resultant vector is Batch_size * Hidden dim and NOT Batch*1*hidden_dim
        
        cls_node = self.dropout(cls_node)
        logits = self.classifier(cls_node) #returns batch_size*num_labels output
        '''

        sentence_mask = (token_type_ids == 0)
        true_sentence_mask = (sentence_mask) & (attention_mask==1) 
        
        seq_nodes = last_state*true_sentence_mask.unsqueeze(-1)
        seq_sum = seq_nodes.sum(dim=1)
        seq_len = true_sentence_mask.sum(dim=1)

        seq_repr = seq_sum/(seq_len.unsqueeze(-1))

        logits = self.classifier(seq_repr) 

        #extracting the aspect and the sentence mask, returns a boolean list with 1's on sentence and aspect positions respecitvely and zeros everywhere else
        '''
        sentence_mask = (token_type_ids == 0)
        aspect_mask = (token_type_ids == 1)

        #multiply the aspect mask with the hidden state to get only the aspect vectors, sentence vectors become zero
        aspect_embeds = last_state * aspect_mask.unsqueeze(-1) #B*L*H  *  B*L*1 = B*L*H

        #Add all the aspect vector per sequence.
        asp_sum = aspect_embeds.sum(dim=1) # Results in B*H vector

        #Find the asp_length per sequence and unsqueeze it to make it compatible with the next operation 
        asp_len = aspect_mask.sum(dim=1).unsqueeze(-1)  # B*1
        
        #Mean pooling of the aspect vectors
        asp_repr = asp_sum/asp_len #B*H resultant

        #B*L*H @ B*H*1 = B*L*1; B*L upon squeezing
        attn_scores = torch.bmm(last_state, asp_repr.unsqueeze(-1)).squeeze(-1)#B*L resultant
       
        #masking the padding tokens as well, they also have tokentype_id as zero but attention_mask is 0       
        true_sentence_mask = (sentence_mask) & (attention_mask==1) 
        true_sentence_mask[:, 0] = 0
        #make the padding and aspect token scores as -inf so that, softmax makes it zero
        attn_scores = attn_scores.masked_fill(~true_sentence_mask, float("-inf"))#B*L


        attn_weights = torch.softmax(attn_scores, dim=1) #B*L

        #element-wise multiplication of attention weights with the final hidden state, then collapsing the tokens in a sequence into 1 embedding
        sentence_repr = torch.sum(last_state * attn_weights.unsqueeze(-1), dim=1)
        '''


        #if the labels are also provided, for training
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss=loss_fn(logits, labels)
            return loss, logits
        
        #when we dont have the labels, eval and testing
        return logits


