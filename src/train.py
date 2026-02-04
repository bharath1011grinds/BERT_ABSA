import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import json
from pathlib import Path


from data import ABSADataset, tokenizer
from model import BERTABSA
import os

os.makedirs("checkpoints", exist_ok=True)
best_f1 = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DATA_DIR = Path("data/processed")

TRAIN_FILE = PROCESSED_DATA_DIR/"train.json"
VAL_FILE = PROCESSED_DATA_DIR/"validation.json"
EPOCHS = 4
LR = 1e-5


with open(TRAIN_FILE) as f:
    train_data = json.load(f)

with open(VAL_FILE) as f:
    val_data = json.load(f)

#instantiating the dataset objects
train_dataset = ABSADataset(train_data, tokenizer=tokenizer, max_length=128)
val_dataset = ABSADataset(val_data, tokenizer=tokenizer, max_length=128)

#passing the Dataset objects to the dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

#instantiating the model
model = BERTABSA(num_labels = 3)
model.to(device)


#We are using AdamW instead of Adam because, AdamW decouples the weight decay from the gradient thereby preventing the LR from shrinking or blowing up too much. BERT tends to overfit and AdamW helps in better regularisation.
optimizer = AdamW(model.parameters(), lr = LR )

total_steps = len(train_loader)*EPOCHS

#the LR scheduler below schedules the LR such that there is a linear decay in the LR and when the parameters are updated for the num_training_steps'th time, it will reach zerp 
'''scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_training_steps=total_steps,
                                            num_warmup_steps=0)
'''

#NOTE: LR approaches zero when the parameters are updated for the num_training_stepsth time.


def train_epoch(model, dataloader):
    #set the model to training mode, dropout enabled
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad() #discard the accumulated gradients from the previous batch and start afresh 

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['polarity'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        #does the forward pass implicitly and computes the loss
        loss, _ = model(input_ids, token_type_ids, attention_mask, labels)

        #computing the gradients via backprop
        loss.backward()

        #updating the parameters after gradient computation
        optimizer.step()

        #reducing the LR as per the scheduler
        #scheduler.step()

        total_loss+=loss.item()#.item() is used to extract just the scalar value from the loss tensor

    return total_loss/len(dataloader)#returns the average loss per batch in that epoch




def eval_epoch(model, dataloader):

    model.eval() #model in evaluate mode, dropout is disabled.

    all_preds=[]
    all_labels =[]

    #torch.no_grad() is very different from opt.zero_grad()
    #NOTE: torch.no_grad() switches off the gradient tracking mechanism. Computations are not stored, leading to much faster results. Used only in validation and inference. 
    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['polarity'].to(device)
            logits = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)

            #we use argmax instead of a softmax, because, higher the value the more prob in softmax, and as a resultant, the prediction would be the same as the argmax index.
            preds = torch.argmax(logits, dim=1)#dim =1 because we want a prediction for all 32 items in the batch. Shape was 32*3, we make it 32*1.  

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro'), classification_report(all_labels, all_preds)


if __name__ == '__main__':

    for epoch in range(EPOCHS):

        train_loss = train_epoch(model=model, dataloader=train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} |")


        val_f1, class_report = eval_epoch(model, val_loader)
    
        if val_f1 > best_f1:
            best_f1 = val_f1

            checkpoint_path = f"checkpoints/best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1
            }, checkpoint_path)
            
            print(f"✅ Saved new best model (F1={best_f1:.4f})")


    #print(len(val_loader))
    print(f"Validation macro f1 {val_f1}")
    print(f"Conf matrix : {class_report}")


