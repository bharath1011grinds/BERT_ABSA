from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from lxml import etree
import re
from sklearn.model_selection import train_test_split

#Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

TRAIN_FILE = RAW_DATA_DIR / "Laptop_Train_v2.xml"
#TEST_FILE = RAW_DATA_DIR/"restaurants-trial.xml"

label2id = {'negative' : 0, 'neutral' : 1, 'positive' : 2}

id2label = {v:k for k,v in label2id.items()}

#using uncased bert because we dont need case-sensitivity for sentiment analysis tasks
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def parse_semeval_xml(file_path):

    #will return a list of (sentence, aspect, polarity) dictionaries

    tree = etree.parse(str(file_path))
    root=tree.getroot()

    samples = []

    for sentence in root.iter('sentence'):
        text_elem = sentence.find('text')
        if text_elem is None:
            continue

        sentence_text=text_elem.text.strip()

        aspect_terms=sentence.find('aspectTerms')
        if aspect_terms is None:
            continue

        for aspect in aspect_terms.iter('aspectTerm'):
            polarity = aspect.get('polarity')
            term = aspect.get('term')

            if term is None or polarity == 'conflict':
                continue

            samples.append({'sentence': sentence_text, 'polarity': polarity, 'aspect': term})

    return samples

def save_json(data, out_path):
    out_path.parent.mkdir(parents = True, exist_ok = True)

    with open(out_path, 'w', encoding = "utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii= False)



#handles the encoding part of our raw sentence
class ABSADataset(Dataset):

    def __init__(self, data, tokenizer, max_length = 128):
        '''data will be a list of dict, with keys - sentence, aspect, label'''

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        sentence = item['sentence']
        aspect = item['aspect']
        polarity = item['polarity']
        #we are returning a pt tensor instead of a list because, computation is faster with a pt tensor and datatype is standard at pt.long() but we have to squeeze the data before returning.
        #we are using trucate = 'only_first' to make sure the aspect[segment embedding =1] is not touched even if the length gets more than 128
        encoding = self.tokenizer(sentence, aspect, padding = "max_length",
                                  truncate = 'only_first', max_length = self.max_length,
                                  return_tensors = 'pt')
        
        return {'input_ids' : encoding['input_ids'].squeeze(0),
                'polarity' : torch.tensor(label2id[polarity], dtype=torch.long),
                'attention_mask' : encoding['attention_mask'].squeeze(0),
                'token_type_ids' : encoding['token_type_ids'].squeeze(0)
        }
    



if __name__ == "__main__":
    train_data = parse_semeval_xml(TRAIN_FILE)
    print(f"Train samples: {len(train_data)}")

    train_samples, val_samples = train_test_split(train_data, train_size=0.8, test_size=0.2, random_state=42,stratify=[x['polarity'] for x in train_data])#used stratify to make sure the sentiment propotions are same in the train and validation data


    save_json(train_samples, PROCESSED_DATA_DIR/"train.json")
    save_json(val_samples, PROCESSED_DATA_DIR/"validation.json")
    print(f"sample example: {train_data[0]}")

'''
    vocab = build_vocab(train_samples)
    save_json(vocab, PROCESSED_DATA_DIR/"vocab.json")
    print("vocab size:", len(vocab))
'''
