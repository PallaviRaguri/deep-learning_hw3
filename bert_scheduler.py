import requests
import json
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
from transformers import BertModel, BertTokenizerFast, AdamW

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def loadData(path): 
    with open(path, 'rb') as f:
        raw_data = json.load(f)
    contexts = []
    questions = []
    answers = []
    num_q = 0
    num_pos = 0
    num_imp = 0

    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                num_q  = num_q  +1
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    return num_q, num_pos, num_imp, contexts, questions, answers


num_q, num_pos, num_imp, train_contexts, train_questions, train_answers = loadData('spoken_train-v1.1.json')
num_questions  = num_q
num_posible = num_pos
num_imposible  = num_imp

num_q, num_pos, num_imp, valid_contexts, valid_questions, valid_answers = loadData('spoken_test-v1.1.json')

def add_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

add_answer_end(train_answers, train_contexts)
add_answer_end(valid_answers, valid_contexts)


MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"
doc_stride = 128


tokenizerFast = BertTokenizerFast.from_pretrained(MODEL_PATH)
train_encodings_fast = tokenizerFast(train_questions, train_contexts,  max_length = MAX_LENGTH,truncation=True,stride=doc_stride,
        padding=True)
valid_encodings_fast = tokenizerFast(valid_questions,valid_contexts,  max_length = MAX_LENGTH, truncation=True,stride=doc_stride,
        padding=True)

def ret_Answer_start_and_end_train(idx):
    ret_start = 0
    ret_end = 0
    answer_encoding_fast = tokenizerFast(train_answers[idx]['text'],  max_length = MAX_LENGTH, truncation=True, padding=True)
    for a in range( len(train_encodings_fast['input_ids'][idx]) -  len(answer_encoding_fast['input_ids']) ):
        match = True
        for i in range(1,len(answer_encoding_fast['input_ids']) - 1):
            if (answer_encoding_fast['input_ids'][i] != train_encodings_fast['input_ids'][idx][a + i]):
                match = False
                break
            if match:
                ret_start = a+1
                ret_end = a+i+1
                break
    return(ret_start, ret_end)

start_positions = []
end_positions = []
ctr = 0
for h in range(len(train_encodings_fast['input_ids'])):
    s, e = ret_Answer_start_and_end_train(h)
    start_positions.append(s)
    end_positions.append(e)
    if s==0:
        ctr = ctr + 1

    
train_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})

def ret_Answer_start_and_end_valid(idx):
    ret_start = 0
    ret_end = 0
    answer_encoding_fast = tokenizerFast(valid_answers[idx]['text'],  max_length = MAX_LENGTH, truncation=True, padding=True)
    for a in range( len(valid_encodings_fast['input_ids'][idx])  -  len(answer_encoding_fast['input_ids'])   ):
        match = True
        for i in range(1,len(answer_encoding_fast['input_ids']) - 1):
            if (answer_encoding_fast['input_ids'][i] != valid_encodings_fast['input_ids'][idx][a + i]):
                match = False
                break
            if match:
                ret_start = a+1
                ret_end = a+i+1
                break
    return(ret_start, ret_end)

start_positions = []
end_positions = []
ctr = 0
for h in range(len(valid_encodings_fast['input_ids']) ):
    s, e = ret_Answer_start_and_end_valid(h)
    start_positions.append(s)
    end_positions.append(e)
    if s==0:
        ctr = ctr + 1

valid_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})

class InputDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][i]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][i]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][i]),
            'start_positions': torch.tensor(self.encodings['start_positions'][i]),
            'end_positions': torch.tensor(self.encodings['end_positions'][i])
        }
    def __len__(self):
        return len(self.encodings['input_ids'])
    
train_dataset = InputDataset(train_encodings_fast)
valid_dataset = InputDataset(valid_encodings_fast)

train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=1)

bert_model = BertModel.from_pretrained(MODEL_PATH)

class QAModel(nn.Module):
    def __init__(self):
        super(QAModel, self).__init__()
        self.bert = bert_model
        self.drop_out = nn.Dropout(0.1)
        self.l1 = nn.Linear(768 * 2, 768 * 2)
        self.l2 = nn.Linear(768 * 2, 2)
        self.linear_relu_stack = nn.Sequential(
            self.drop_out,
            self.l1,
            nn.LeakyReLU(),
            self.l2 
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = model_output[2]
        out = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        logits = self.linear_relu_stack(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

model = QAModel()

def focal_loss_fn(start_logits, end_logits, start_positions, end_positions, gamma):
    smax = nn.Softmax(dim=1)
    probs_start = smax(start_logits)
    inv_probs_start = 1 - probs_start
    probs_end = smax(end_logits)
    inv_probs_end = 1 - probs_end
    
    lsmax = nn.LogSoftmax(dim=1)
    log_probs_start = lsmax(start_logits)
    log_probs_end = lsmax(end_logits)
    
    nll = nn.NLLLoss()
    
    fl_start = nll(torch.pow(inv_probs_start, gamma)* log_probs_start, start_positions)
    fl_end = nll(torch.pow(inv_probs_end, gamma)*log_probs_end, end_positions)
    
    return ((fl_start + fl_end)/2)

optim = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)
scheduler = ExponentialLR(optim, gamma=0.9)
total_acc = []
total_loss = []

def train(model, dataloader, epoch):
    model = model.train()
    losses = []
    acc = []
    ctr = 0
    batch_tracker = 0
    for batch in tqdm(dataloader, desc = 'Running Epoch '):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        out_start, out_end = model(input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

        loss = focal_loss_fn(out_start, out_end, start_positions, end_positions,1)
        losses.append(loss.item())
        loss.backward()
        optim.step()
        
        start_pred = torch.argmax(out_start, dim=1)
        end_pred = torch.argmax(out_end, dim=1)
            
        acc.append(((start_pred == start_positions).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_positions).sum()/len(end_pred)).item())

        batch_tracker = batch_tracker + 1
        if batch_tracker==250 and epoch==1:
            total_acc.append(sum(acc)/len(acc))
            loss_avg = sum(losses)/len(losses)
            total_loss.append(loss_avg)
            batch_tracker = 0
    scheduler.step()
    ret_acc = sum(acc)/len(acc)
    ret_loss = sum(losses)/len(losses)
    return(ret_acc, ret_loss)


def eval_model(model, dataloader):
    model = model.eval()
    losses = []
    acc = []
    ctr = 0
    answer_list=[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = 'Running Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            out_start, out_end = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)

            start_pred = torch.argmax(out_start)
            end_pred = torch.argmax(out_end)
            answer = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
            tanswer = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(input_ids[0][start_true[0]:end_true[0]]))
            answer_list.append([answer,tanswer])

    return answer_list

from evaluate import load
wer = load("wer")
EPOCHS = 4
model.to(device)
wer_list=[]
for epoch in range(EPOCHS):
    train_acc, train_loss = train(model, train_data_loader, epoch+1)
    print('Epoch - {}'.format(epoch))
    print(f"Accuracy: {train_acc}")
    print(f"Loss: {train_loss}")
    answer_list = eval_model(model, valid_data_loader)
    pred_answers=[]
    true_answers=[]
    for i in range(len(answer_list)):
        if(len(answer_list[i][0])==0):
            answer_list[i][0]="$"
        if(len(answer_list[i][1])==0):
            answer_list[i][1]="$"
        pred_answers.append(answer_list[i][0])
        true_answers.append(answer_list[i][1])
    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_list.append(wer_score)

print('WER (after adding scheduler) - ',wer_list)