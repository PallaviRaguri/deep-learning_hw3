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
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import json

def get_data(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    
    questions, contexts, answers = zip(*[
        (qa['question'].lower(), paragraph['context'].lower(), answer)
        for group in raw_data['data']
        for paragraph in group['paragraphs']
        for qa in paragraph['qas']
        for answer in qa['answers']
    ])
    
    num_q = len(questions)
    num_pos = 0
    num_imp = 0

    return num_q, num_pos, num_imp, list(contexts), list(questions), list(answers)


num_q, num_pos, num_imp, train_contexts, train_questions, train_answers = get_data('spoken_train-v1.1.json')
num_questions  = num_q
num_posible = num_pos
num_imposible  = num_imp


num_q, num_pos, num_imp, valid_contexts, valid_questions, valid_answers = get_data('spoken_test-v1.1.json')


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
pad_on_right = tokenizerFast.padding_side == "right"
train_contexts_trunc=[]

for i, context in enumerate(train_contexts):
    if len(context) > MAX_LENGTH:
        answer = train_answers[i]
        mid = (answer['answer_start'] + answer['answer_end']) // 2
        para_start = max(0, min(mid - MAX_LENGTH // 2, len(context) - MAX_LENGTH))
        train_contexts_trunc.append(context[para_start:para_start + MAX_LENGTH])
        train_answers[i]['answer_start'] = MAX_LENGTH // 2 - len(answer['text']) // 2
    else:
        train_contexts_trunc.append(context)

train_encodings_fast = tokenizerFast(train_questions, train_contexts_trunc, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)
valid_encodings_fast = tokenizerFast(valid_questions, valid_contexts, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)


def ret_Answer_start_and_end_train(idx):
    ret_start, ret_end = 0, 0
    answer_enc = tokenizerFast(train_answers[idx]['text'], max_length=MAX_LENGTH, truncation=True, padding=True)
    enc_ids = train_encodings_fast['input_ids'][idx]
    ans_ids = answer_enc['input_ids']
    
    for a in range(len(enc_ids) - len(ans_ids)):
        if ans_ids[1:-1] == enc_ids[a + 1:a + len(ans_ids) - 1]:
            ret_start, ret_end = a + 1, a + len(ans_ids) - 1
            break

    return ret_start, ret_end


start_positions = []
end_positions = []
ctr = 0
for h, _ in enumerate(train_encodings_fast['input_ids']):
    s, e = ret_Answer_start_and_end_train(h)
    start_positions.append(s)
    end_positions.append(e)
    ctr += (s == 0)


train_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})

def ret_Answer_start_and_end_valid(idx):
    answer_encoding_fast = tokenizerFast(valid_answers[idx]['text'], max_length=MAX_LENGTH, truncation=True, padding=True)
    for a in range(len(valid_encodings_fast['input_ids'][idx]) - len(answer_encoding_fast['input_ids'])):
        if all(answer_encoding_fast['input_ids'][i] == valid_encodings_fast['input_ids'][idx][a + i] for i in range(1, len(answer_encoding_fast['input_ids']) - 1)):
            return a + 1, a + len(answer_encoding_fast['input_ids']) - 2
    return 0, 0

start_positions, end_positions, ctr = [], [], 0
for idx in range(len(valid_encodings_fast['input_ids'])):
    s, e = ret_Answer_start_and_end_valid(idx)
    start_positions.append(s)
    end_positions.append(e)
    ctr += s == 0

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
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),
            nn.LeakyReLU(),
            nn.Linear(768 * 2, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = model_output.hidden_states
        out = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        out = self.dropout(out)
        logits = self.classifier(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


model = QAModel()

def focal_loss_fn(start_logits, end_logits, start_positions, end_positions, gamma):
    smax = nn.Softmax(dim=1)
    lsmax = nn.LogSoftmax(dim=1)
    nll = nn.NLLLoss()

    probs_start, probs_end = smax(start_logits), smax(end_logits)
    inv_probs_start, inv_probs_end = 1 - probs_start, 1 - probs_end

    log_probs_start, log_probs_end = lsmax(start_logits), lsmax(end_logits)

    fl_start = nll(torch.pow(inv_probs_start, gamma) * log_probs_start, start_positions)
    fl_end = nll(torch.pow(inv_probs_end, gamma) * log_probs_end, end_positions)
    
    return (fl_start + fl_end) / 2


optim = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)
total_acc = []
total_loss = []

def train_epoch(model, dataloader, epoch):
    model.train()
    total_loss, total_acc = 0, 0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc=f'Running Epoch {epoch}'):
        optim.zero_grad()
        
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        del batch['start_positions']
        del batch['end_positions']
        
        # Move the rest of the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()} 
        
        # Forward pass
        out_start, out_end = model(**batch)

        # Compute loss
        loss = focal_loss_fn(out_start, out_end, start_positions, end_positions, gamma=1)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        start_pred, end_pred = torch.argmax(out_start, dim=1), torch.argmax(out_end, dim=1)
        total_acc += ((start_pred == start_positions).float().mean() + (end_pred == end_positions).float().mean()) / 2

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_acc, avg_loss



def eval_model(model, dataloader):
    model.eval()
    answer_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running Evaluation'):
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            del batch['start_positions']
            del batch['end_positions']
            
            batch = {k: v.to(device) for k, v in batch.items()}  # Efficient device assignment
            out_start, out_end = model(**batch)

            start_pred = torch.argmax(out_start, dim=1)
            end_pred = torch.argmax(out_end, dim=1)

            for idx in range(len(batch['input_ids'])):
                start_idx = int(start_pred[idx])
                end_idx = int(end_pred[idx]) + 1  # +1 because the end index is exclusive
                start_idx, end_idx = (start_idx, end_idx) if start_idx < end_idx else (end_idx, start_idx)  # Ensure correct ordering
                
                answer_pred = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(batch['input_ids'][idx][start_idx:end_idx]))
                true_start = int(start_positions[idx])
                true_end = int(end_positions[idx]) + 1  # +1 because the end index is exclusive
                answer_true = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(batch['input_ids'][idx][true_start:true_end]))
                answer_list.append([answer_pred, answer_true])

    return answer_list

from evaluate import load

wer = load("wer")
EPOCHS = 6
model.to(device)
wer_list = []
print('Starting training')

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_data_loader, epoch + 1)
    answer_list = eval_model(model, valid_data_loader)
    
    # Replace empty answers with a placeholder and collect predictions and true answers
    pred_answers, true_answers = zip(*[(ans[0] if ans[0] else "$", ans[1] if ans[1] else "$") for ans in answer_list])
    
    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_list.append(wer_score)

print('WER (base model) - ',wer_list)



##############################################################

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
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import json

def get_data(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    
    dataset_questions, dataset_contexts, dataset_answers = zip(*[
        (qa['question'].lower(), paragraph['context'].lower(), answer)
        for group in raw_data['data']
        for paragraph in group['paragraphs']
        for qa in paragraph['qas']
        for answer in qa['answers']
    ])
    
    n_questions = len(dataset_questions)
    num_pos = 0
    num_imp = 0

    return n_questions, num_pos, num_imp, list(dataset_contexts), list(dataset_questions), list(dataset_answers)


n_questions, num_pos, num_imp, train_contexts, train_questions, train_answers = get_data('spoken_train-v1.1.json')
num_questions  = n_questions
num_posible = num_pos
num_imposible  = num_imp


n_questions, num_pos, num_imp, valid_contexts, valid_questions, valid_answers = get_data('spoken_test-v1.1.json')


def append_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

append_answer_end(train_answers, train_contexts)
append_answer_end(valid_answers, valid_contexts)


MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"

doc_stride = 128
tokenizerFast = BertTokenizerFast.from_pretrained(MODEL_PATH)
pad_on_right = tokenizerFast.padding_side == "right"
train_contexts_trunc=[]

for i, context in enumerate(train_contexts):
    if len(context) > MAX_LENGTH:
        answer = train_answers[i]
        mid = (answer['answer_start'] + answer['answer_end']) // 2
        para_start = max(0, min(mid - MAX_LENGTH // 2, len(context) - MAX_LENGTH))
        train_contexts_trunc.append(context[para_start:para_start + MAX_LENGTH])
        train_answers[i]['answer_start'] = MAX_LENGTH // 2 - len(answer['text']) // 2
    else:
        train_contexts_trunc.append(context)

train_encodings_fast = tokenizerFast(train_questions, train_contexts_trunc, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)
valid_encodings_fast = tokenizerFast(valid_questions, valid_contexts, max_length=MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)


def get_answer_start_end_train(idx):
    ret_start, ret_end = 0, 0
    answer_enc = tokenizerFast(train_answers[idx]['text'], max_length=MAX_LENGTH, truncation=True, padding=True)
    enc_ids = train_encodings_fast['input_ids'][idx]
    ans_ids = answer_enc['input_ids']
    
    for a in range(len(enc_ids) - len(ans_ids)):
        if ans_ids[1:-1] == enc_ids[a + 1:a + len(ans_ids) - 1]:
            ret_start, ret_end = a + 1, a + len(ans_ids) - 1
            break

    return ret_start, ret_end


start_positions = []
end_positions = []
ctr = 0
for h, _ in enumerate(train_encodings_fast['input_ids']):
    s, e = get_answer_start_end_train(h)
    start_positions.append(s)
    end_positions.append(e)
    ctr += (s == 0)


train_encodings_fast.update({'start_positions': start_positions, 'end_positions': end_positions})

def get_answer_start_end_valid(idx):
    answer_encoding_fast = tokenizerFast(valid_answers[idx]['text'], max_length=MAX_LENGTH, truncation=True, padding=True)
    for a in range(len(valid_encodings_fast['input_ids'][idx]) - len(answer_encoding_fast['input_ids'])):
        if all(answer_encoding_fast['input_ids'][i] == valid_encodings_fast['input_ids'][idx][a + i] for i in range(1, len(answer_encoding_fast['input_ids']) - 1)):
            return a + 1, a + len(answer_encoding_fast['input_ids']) - 2
    return 0, 0

start_positions, end_positions, ctr = [], [], 0
for idx in range(len(valid_encodings_fast['input_ids'])):
    s, e = get_answer_start_end_valid(idx)
    start_positions.append(s)
    end_positions.append(e)
    ctr += s == 0
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
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),
            nn.LeakyReLU(),
            nn.Linear(768 * 2, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = model_output.hidden_states
        out = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        out = self.dropout(out)
        logits = self.classifier(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


model = QAModel()

def focal_loss_fn(start_logits, end_logits, start_positions, end_positions, gamma):
    smax = nn.Softmax(dim=1)
    lsmax = nn.LogSoftmax(dim=1)
    nll = nn.NLLLoss()

    probs_start, probs_end = smax(start_logits), smax(end_logits)
    inv_probs_start, inv_probs_end = 1 - probs_start, 1 - probs_end

    log_probs_start, log_probs_end = lsmax(start_logits), lsmax(end_logits)

    fl_start = nll(torch.pow(inv_probs_start, gamma) * log_probs_start, start_positions)
    fl_end = nll(torch.pow(inv_probs_end, gamma) * log_probs_end, end_positions)
    
    return (fl_start + fl_end) / 2


optim = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)
total_acc = []
total_loss = []

def train_epoch(model, dataloader, epoch):
    model.train()
    total_loss, total_acc = 0, 0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc=f'Running Epoch {epoch}'):
        optim.zero_grad()
        
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        del batch['start_positions']
        del batch['end_positions']
        
        batch = {k: v.to(device) for k, v in batch.items()} 
        
        # Forward pass
        out_start, out_end = model(**batch)

        # Compute loss
        loss = focal_loss_fn(out_start, out_end, start_positions, end_positions, gamma=1)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        start_pred, end_pred = torch.argmax(out_start, dim=1), torch.argmax(out_end, dim=1)
        total_acc += ((start_pred == start_positions).float().mean() + (end_pred == end_positions).float().mean()) / 2

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_acc, avg_loss



def eval_model(model, dataloader):
    model.eval()
    answer_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running Evaluation'):
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            del batch['start_positions']
            del batch['end_positions']
            
            batch = {k: v.to(device) for k, v in batch.items()}  # Efficient device assignment
            out_start, out_end = model(**batch)

            start_pred = torch.argmax(out_start, dim=1)
            end_pred = torch.argmax(out_end, dim=1)

            for idx in range(len(batch['input_ids'])):
                start_idx = int(start_pred[idx])
                end_idx = int(end_pred[idx]) + 1  # +1 because the end index is exclusive
                start_idx, end_idx = (start_idx, end_idx) if start_idx < end_idx else (end_idx, start_idx)  # Ensure correct ordering
                
                answer_pred = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(batch['input_ids'][idx][start_idx:end_idx]))
                true_start = int(start_positions[idx])
                true_end = int(end_positions[idx]) + 1  # +1 because the end index is exclusive
                answer_true = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(batch['input_ids'][idx][true_start:true_end]))
                answer_list.append([answer_pred, answer_true])

    return answer_list

from evaluate import load

wer = load("wer")
EPOCHS = 6
model.to(device)
wer_list = []
print('Starting training')

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_data_loader, epoch + 1)
    answer_list = eval_model(model, valid_data_loader)
    
    pred_answers, true_answers = zip(*[(ans[0] if ans[0] else "$", ans[1] if ans[1] else "$") for ans in answer_list])
    
    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_list.append(wer_score)

print('WER (base model) - ',wer_list)