from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

import torch.nn as nn
import random

import torch.optim as optim
import numpy as np
# import spacy
import tqdm
from torch.optim.lr_scheduler import LambdaLR
import re 

import torch.nn.functional as F
from transformers import BertModel, BertTokenizer



def tokenize_formula_lstm_lstm(formula):
    parts = re.split(r'([|(),])', formula)
    tokens = []
    for part in parts:
        if not part:
            continue
        elif part.startswith('#') and part[1:].isdigit():
            tokens.append(part)
        elif part.startswith('const_'):
            tokens.append(part)
        elif len(part) == 2 and part[0].isalpha() and part[1].isdigit():
            tokens.append(part)
        elif len(part) > 1 and all(char.isdigit() for char in part):
            continue
        else:
            tokens.append(part)
    return tokens



def tokenize_problems_nltk_lstm_lstm(problem_text):
    tokenized_problem = word_tokenize(problem_text)

    tokenized_problem = [token for token in tokenized_problem if not any(char.isdigit() for char in token)]

    return tokenized_problem

def problem_to_indices_lstm_lstm(problem_text):
    indices = [problem_vocab_lstm_lstm.get(token, problem_vocab_lstm_lstm['<unk>']) for token in tokenize_problems_nltk_lstm_lstm(problem_text)]
    indices = [problem_vocab_lstm_lstm['<sos>']] + indices + [problem_vocab_lstm_lstm['<eos>']]
    return indices

def formula_to_indices_lstm_lstm(formula_text):
    tokens= tokenize_formula_lstm_lstm(formula_text)
    indices = [formula_vocab_lstm_lstm.get(token, formula_vocab_lstm_lstm['<unk>']) for token in tokens]
    indices = [formula_vocab_lstm_lstm['<sos>']] + indices + [formula_vocab_lstm_lstm['<eos>']]
    return indices

formula_vocab_path_lstm_lstm= "formula_vocab_lstm_lstm.json"
problem_vocab_path_lstm_lstm= "problem_vocab_lstm_lstm.json"

with open(formula_vocab_path_lstm_lstm, "r") as json_file:
    formula_vocab_lstm_lstm = json.load(json_file)
    
    
with open(problem_vocab_path_lstm_lstm, "r") as json_file:
    problem_vocab_lstm_lstm = json.load(json_file)
    
    
embedding_matrix_json_lstm_lstm_path= 'embedding_matrix_lstm_lstm.json'

with open(embedding_matrix_json_lstm_lstm_path, "r") as json_file:
    embedding_matrix_list = json.load(json_file)


embedding_matrix_np = np.array(embedding_matrix_list)

embedding_matrix_lstm_lstm = torch.tensor(embedding_matrix_np,dtype=torch.float32)

formula_vocab_reverse_lstm_lstm = {value: key for key, value in formula_vocab_lstm_lstm.items()}



class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix_lstm_lstm, freeze= True)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers,bidirectional= True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc= nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, src):
        # print(f"\nIn ENCODER!!\n")
        # src = [src length, batch size]
        src_new = src.permute(1,0)
        # print(f"src shape: {src.shape}\n")
        embedded = self.dropout(self.embedding(src_new))
        # print(f"embedded shape: {embedded.shape}\n")
        # embedded = [src length, batch size, embedding dim]
        # print(f"embedded: {embedded.shape}\n")
        outputs, (hidden, cell) = self.rnn(embedded)
        # print(f"outputs shape: {outputs.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]

        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        # print(f"\nhidden shape: {hidden.shape}\n")
        hidden_final= self.fc(hidden)
        cell_final= self.fc(cell)
        return hidden_final.unsqueeze(0), cell_final.unsqueeze(0)

class BiLSTMDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, teacher_forcing_ratio):
        # print(f"\nIn DECODER!!\n")
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        # print(f"input shape before unsqueeze: {input.shape}\n")
        input = input.unsqueeze(0)
        # print(f"input shape after unsqueeze: {input.shape}\n")
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # print(f"embedded shape: {embedded.shape}\n")
        # embedded = [1, batch size, embedding dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print(f"output shape: {output.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        # print(f"output shape before squeeze: {output.shape}\n")
        prediction = self.fc_out(output.squeeze(0))
        # print(f"prediction shape: {prediction.shape}, and i/p to pred was: {output.squeeze(0)}\n")
        # prediction = [batch size, output dim]
        # print(f"prediction shape: {prediction.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        return prediction, hidden, cell

class Seq2SeqBiLSTMBiLSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = len(formula_vocab_lstm_lstm)
        # tensor to store decoder outputs
        # if is_val== True:
        #   print(trg_vocab_size)

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # print(f"From Encoder: hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        # print("here!!\n")
        input = trg[0, :]

        # input = [batch size]
        for t in range(1, trg_length):
            # print("here!!\n")
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # print(f"output shape to decoder: {output.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
            output, hidden, cell = self.decoder(input, hidden, cell,teacher_forcing_ratio)
            # print(f"output shape after decoder: {output.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token

            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
    

def beam_search_lstm_lstm(model, src, beam_width=2, max_length=50):

    model.eval()
    src_ = src.permute(1,0)

    with torch.no_grad():

        hidden, cell = model.encoder(src_)



        beam = [(torch.tensor([2]), 0, hidden, cell)]
        completed_beams = []


        for _ in range(max_length):
            new_beam = []

            for sequence, score, hidden, cell in beam:
                if sequence[-1] == 1:

                    completed_beams.append((sequence, score))
                    continue


                input = sequence[-1].unsqueeze(0).to(src.device)

                output, hidden, cell = model.decoder(input, hidden, cell, teacher_forcing_ratio=0)
                output = torch.log_softmax(output, dim=-1)


                topk_scores, topk_indices = output.topk(beam_width)
                topk_scores = topk_scores.squeeze() + score

                for i in range(beam_width):
                    new_token_id = topk_indices[0][i].item()
                    new_score = topk_scores[i].item()
                    new_sequence = torch.cat([sequence, torch.tensor([new_token_id])])
                    new_beam.append((new_sequence, new_score, hidden, cell))


            new_beam.sort(key=lambda x: x[1], reverse=True)


            beam = new_beam[:beam_width]


        completed_beams.extend([(beam[i][0], beam[i][1]) for i in range(len(beam))])


        completed_beams.sort(key=lambda x: x[1], reverse=True)

        # Return the best sequence
        return completed_beams[0][0][1:]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def tokenize_formula_lstm_lstm_attn(formula):
    parts = re.split(r'([|(),])', formula)
    tokens = []
    for part in parts:
        if not part:
            continue
        elif part.startswith('#') and part[1:].isdigit():
            tokens.append(part)
        elif part.startswith('const_'):
            tokens.append(part)
        elif len(part) == 2 and part[0].isalpha() and part[1].isdigit():
            tokens.append(part)
        elif len(part) > 1 and all(char.isdigit() for char in part):
            continue
        else:
            tokens.append(part)
    return tokens

def tokenize_problems_nltk_lstm_lstm_attn(problem_text):
    tokenized_problem = word_tokenize(problem_text)
    tokenized_problem = [token for token in tokenized_problem if not any(char.isdigit() for char in token)]
    return tokenized_problem

def problem_to_indices_lstm_lstm_attn(problem_text):
    indices = [problem_vocab_lstm_lstm.get(token, problem_vocab_lstm_lstm['<unk>']) for token in tokenize_problems_nltk_lstm_lstm_attn(problem_text)]
    indices = [problem_vocab_lstm_lstm['<sos>']] + indices + [problem_vocab_lstm_lstm['<eos>']]
    return indices

def formula_to_indices_lstm_lstm_attn(formula_text):
    tokens = tokenize_formula_lstm_lstm_attn(formula_text)
    indices = [formula_vocab_lstm_lstm.get(token, formula_vocab_lstm_lstm['<unk>']) for token in tokens]
    indices = [formula_vocab_lstm_lstm['<sos>']] + indices + [formula_vocab_lstm_lstm['<eos>']]
    return indices

formula_vocab_path_lstm_lstm_attn = "formula_vocab_lstm_lstm.json"
problem_vocab_path_lstm_lstm_attn = "problem_vocab_lstm_lstm.json"

with open(formula_vocab_path_lstm_lstm_attn, "r") as json_file:
    formula_vocab_lstm_lstm_attn = json.load(json_file)
    
with open(problem_vocab_path_lstm_lstm_attn, "r") as json_file:
    problem_vocab_lstm_lstm_attn = json.load(json_file)

embedding_matrix_json_lstm_lstm_attn_path = 'embedding_matrix_lstm_lstm.json'

with open(embedding_matrix_json_lstm_lstm_attn_path, "r") as json_file:
    embedding_matrix_list = json.load(json_file)

embedding_matrix_np = np.array(embedding_matrix_list)
embedding_matrix_lstm_lstm_attn = torch.tensor(embedding_matrix_np, dtype=torch.float32)

formula_vocab_reverse_lstm_lstm_attn = {value: key for key, value in formula_vocab_lstm_lstm.items()}
    

formula_vocab_path_bert_lstm_tuned= "formula_vocab_bert_lstm_attn_tuned.json"
problem_vocab_path_bert_lstm_tuned= "problem_vocab_bert_lstm_attn_tuned.json"



with open(formula_vocab_path_bert_lstm_tuned, "r") as json_file:
    formula_vocab_bert_lstm_attn_tuned = json.load(json_file)
    
with open(problem_vocab_path_bert_lstm_tuned, "r") as json_file:
    problem_vocab_bert_lstm_attn_tuned = json.load(json_file)
    
    
    
    
    
formula_vocab_path_bert_lstm_frozen= "formula_vocab_bert_lstm_attn_frozen.json"
problem_vocab_path_bert_lstm_frozen= "problem_vocab_bert_lstm_attn_frozen.json"



with open(formula_vocab_path_bert_lstm_frozen, "r") as json_file:
    formula_vocab_bert_lstm_attn_frozen = json.load(json_file)
    
with open(problem_vocab_path_bert_lstm_frozen, "r") as json_file:
    problem_vocab_bert_lstm_attn_frozen = json.load(json_file)
    
    






class BiLSTMEncoderAttn(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc= nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, src):
        src_new = src.permute(1, 0)
        embedded = self.dropout(self.embedding(src_new))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        hidden_final= self.fc(hidden)
        cell_final= self.fc(cell)
        return outputs, hidden_final.unsqueeze(0), cell_final.unsqueeze(0)
    

class BiLSTMAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class BiLSTMDecoderAttn(nn.Module):
    def __init__(self, output_dim, embedding_dim, decoder_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = decoder_hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM((embedding_dim * 2) + decoder_hidden_dim, decoder_hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((decoder_hidden_dim * 4), output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs, teacher_forcing_ratio):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
#         print(f"output_sq : {output.squeeze(0).shape}, weighted_sq: {weighted.squeeze(0).shape}, embedded: {embedded.squeeze(0).shape}")
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim = 1))
        return prediction, hidden, cell
    
    
class Seq2SeqBiLSTMLSTMAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = len(formula_vocab_lstm_lstm_attn)
        # tensor to store decoder outputs
        # if is_val== True:
        #   print(trg_vocab_size)

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden, cell = self.encoder(src)
        # print(f"From Encoder: hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        # print("here!!\n")
        input = trg[0, :]

        # input = [batch size]
        for t in range(1, trg_length):
            # print("here!!\n")
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # print(f"output shape to decoder: {output.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, teacher_forcing_ratio)
            # print(f"output shape after decoder: {output.shape}, hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token

            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
    
    
    

def beam_search_lstm_lstm_attn(model, src, beam_width=2, max_length=50):

    model.eval()
    src_ = src.permute(1,0)

    with torch.no_grad():

        encoder_output,hidden, cell = model.encoder(src_)



        beam = [(torch.tensor([2]), 0, hidden, cell)]
        completed_beams = []


        for _ in range(max_length):
            new_beam = []

            for sequence, score, hidden, cell in beam:
                if sequence[-1] == 1:

                    completed_beams.append((sequence, score))
                    continue


                input = sequence[-1].unsqueeze(0).to(src.device)
#                 input, hidden, cell, encoder_outputs
                output, hidden, cell = model.decoder(input, hidden, cell, encoder_output, teacher_forcing_ratio=0)
                output = torch.log_softmax(output, dim=-1)


                topk_scores, topk_indices = output.topk(beam_width)
                topk_scores = topk_scores.squeeze() + score

                for i in range(beam_width):
                    new_token_id = topk_indices[0][i].item()
                    new_score = topk_scores[i].item()
                    new_sequence = torch.cat([sequence, torch.tensor([new_token_id])])
                    new_beam.append((new_sequence, new_score, hidden, cell))


            new_beam.sort(key=lambda x: x[1], reverse=True)


            beam = new_beam[:beam_width]


        completed_beams.extend([(beam[i][0], beam[i][1]) for i in range(len(beam))])


        completed_beams.sort(key=lambda x: x[1], reverse=True)

        # Return the best sequence
        return completed_beams[0][0][1:]



device= 'cuda' if torch.cuda.is_available() else 'cuda'







# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Initialize BERT model
bert_model = BertModel.from_pretrained('bert-base-cased')


def tokenize_formula_bert(formula):
    # Split the expression based on pipe, parentheses, and commas
    parts = re.split(r'([|(),])', formula)
    tokens = []
    for part in parts:
        # Tag constants and placeholders accordingly
        if not part:
            # Skip empty parts
            continue
        elif part.startswith('#') and part[1:].isdigit():
            # If part starts with '#' followed by digits, treat it as a single token
            tokens.append(part)
        elif part.startswith('const_'):
            # Treat variables like const_100 as single tokens
            tokens.append(part)
        elif len(part) == 2 and part[0].isalpha() and part[1].isdigit():
            # If part is of type n1, treat it as a single token
            tokens.append(part)
        elif len(part) > 1 and all(char.isdigit() for char in part):
            # If part consists of all digits, skip it
            continue
        else:
            tokens.append(part)
    return tokens

def tokenize_problems_bert(problem_text):
    # Tokenize the problem text
    tokenized_problem = tokenizer.tokenize(problem_text)

    # Filter out tokens containing digits
#     tokenized_problem = [token for token in tokenized_problem if not any(char.isdigit() for char in token)]

    return tokenized_problem



# Text to index functions
def problem_to_indices_bert(problem_text):
    indices = [problem_vocab_bert_lstm_attn_tuned.get(token, problem_vocab_bert_lstm_attn_tuned['<unk>']) for token in tokenize_problems_bert(problem_text)]
    indices = [problem_vocab_bert_lstm_attn_tuned['<sos>']] + indices + [problem_vocab_bert_lstm_attn_tuned['<eos>']]
    return indices

def formula_to_indices_bert(formula_text):
    tokens= tokenize_formula_bert(formula_text)
    # print(f"tokens: {tokens}\n")
    indices = [formula_vocab_bert_lstm_attn_tuned.get(token, formula_vocab_path_bert_lstm_tuned['<unk>']) for token in tokens]
    indices = [formula_vocab_bert_lstm_attn_tuned['<sos>']] + indices + [formula_vocab_bert_lstm_attn_tuned['<eos>']]
    return indices



class BertEncoderFineTuned(nn.Module):
    def __init__(self, bert_model_name,decoder_hidden_dim):
        super(BertEncoderFineTuned, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.shape_params= nn.Linear(768,decoder_hidden_dim)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.hid  = nn.Linear(768, 128)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, problems, masks):
        input_ids = problems.to(device)
        # print(f"Input id shape: {input_ids.shape}\n")
        attention_mask = masks.to(device)
        # print(input_ids, attention_mask)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
#         print(bert_output.last_hidden_state)
#         print("______________")
#         print(bert_output[0])
#         bert_output = self.shape_params(bert_output)
        
        hidden = torch.tanh(self.hid(bert_output[:,0]))
        bert_output = self.shape_params(bert_output)
        bert_out=bert_output.permute(1,0,2)
#         print(hidden.shape)
        #print(f"bert output: {bert_output.shape} | {bert_output[:, 0, :].shape}\n")
#         bert_out = bert_output.permute(1,0,2)
#         bert_out, bert_output= self.shape_params(bert_out), self.shape_params(bert_output)
        #print(f"bert output: {bert_output.shape} | bert_out.shape: {bert_output[:, 0, :].shape}\n")
#         x= bert_output.permute(0,1,2)
        cls= hidden.unsqueeze(0)
#         print(f"x shape, cls.shape: {bert_out.shape},{cls.shape}\n")
        return bert_out, cls  # output, CLS token of the last hidden state



class BertAttentionFineTuned(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim *1) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #print(f"hidden = {hidden.shape}, enc_out ={encoder_outputs.shape}\n")
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
    
    
    
    
formula_vocab_reverse_bert_lstm_attn_tuned = {value: key for key, value in formula_vocab_bert_lstm_attn_tuned.items()}

decoder_embedding_dim_bert_ft=128
class BertDecoderFineTuned(nn.Module):
    def __init__(self, output_dim, embedding_dim, decoder_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = decoder_hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM((decoder_embedding_dim_bert_ft * 1) + decoder_hidden_dim, decoder_hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((decoder_hidden_dim * 3), output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs, teacher_forcing_ratio):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #print(f"rnn_ip: {rnn_input.shape} | hidden:  {hidden.shape}  cell | {cell.shape}\n")
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.contiguous(), cell.contiguous()))
#         print(f"output_sq : {output.squeeze(0).shape}, weighted_sq: {weighted.squeeze(0).shape}, embedded: {embedded.squeeze(0).shape}")
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim = 1))
        return prediction, hidden, cell


class Seq2SeqBertSTMAttnFineTuned(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, src, trg, mask,teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = len(formula_vocab_bert_lstm_attn_tuned)
        # print(trg_vocab_size)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out, hidden= self.encoder(src,mask)
        hidden= torch.tanh(hidden)
        cell = hidden
        # print(f"From Encoder: hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        # print("here!!\n")
        input = trg[0, :]

        # input = [batch size]
        for t in range(1, trg_length):
            # print("here!!\n")
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # print(f" hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")

            # input, hidden, cell, encoder_outputs, teacher_fo hiddercing_ratio

            output, hidden, cell = self.decoder(input, hidden, cell, enc_out,teacher_forcing_ratio)
            # print(f"output shape after decoder: {output.shape},n shape: {hidden.shape}, cell shape: {cell.shape}\n")
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token

            outputs[t] = output
            # print("here")
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs










def beam_search_bert_ft(model, src,attn_mask, beam_width=2, max_length=50):

    model.eval()
    # print(src.shape)
    # src_ = src.permute(1,0)
    # attn_mask= attn_mask.permute(1,0)
    
    # print(src_.shape, attn_mask.shape)
    src_= src 

    with torch.no_grad():

        encoder_output,hidden = model.encoder(src_, attn_mask)
        cell= hidden


        beam = [(torch.tensor([2]), 0, hidden, cell)]
        completed_beams = []


        for _ in range(max_length):
            new_beam = []

            for sequence, score, hidden, cell in beam:
                if sequence[-1] == 1:

                    completed_beams.append((sequence, score))
                    continue


                input = sequence[-1].unsqueeze(0).to(src.device)
#                 input, hidden, cell, encoder_outputs
                output, hidden, cell = model.decoder(input, hidden, cell, encoder_output, teacher_forcing_ratio=0)
                output = torch.log_softmax(output, dim=-1)


                topk_scores, topk_indices = output.topk(beam_width)
                topk_scores = topk_scores.squeeze() + score

                for i in range(beam_width):
                    new_token_id = topk_indices[0][i].item()
                    new_score = topk_scores[i].item()
                    new_sequence = torch.cat([sequence, torch.tensor([new_token_id])])
                    new_beam.append((new_sequence, new_score, hidden, cell))


            new_beam.sort(key=lambda x: x[1], reverse=True)


            beam = new_beam[:beam_width]


        completed_beams.extend([(beam[i][0], beam[i][1]) for i in range(len(beam))])


        completed_beams.sort(key=lambda x: x[1], reverse=True)

        # Return the best sequence
        return completed_beams[0][0][1:]
    
    

def beam_search_bert_fr(model, src,attn_mask, beam_width=2, max_length=50):

    model.eval()
    # print(src.shape)
    # src_ = src.permute(1,0)
    # attn_mask= attn_mask.permute(1,0)
    
    # print(src_.shape, attn_mask.shape)
    src_= src 

    with torch.no_grad():

        encoder_output,hidden = model.encoder(src_, attn_mask)
        cell= hidden


        beam = [(torch.tensor([2]), 0, hidden, cell)]
        completed_beams = []


        for _ in range(max_length):
            new_beam = []

            for sequence, score, hidden, cell in beam:
                if sequence[-1] == 1:

                    completed_beams.append((sequence, score))
                    continue


                input = sequence[-1].unsqueeze(0).to(src.device)
#                 input, hidden, cell, encoder_outputs
                output, hidden, cell = model.decoder(input, hidden, cell, encoder_output, teacher_forcing_ratio=0)
                output = torch.log_softmax(output, dim=-1)


                topk_scores, topk_indices = output.topk(beam_width)
                topk_scores = topk_scores.squeeze() + score

                for i in range(beam_width):
                    new_token_id = topk_indices[0][i].item()
                    new_score = topk_scores[i].item()
                    new_sequence = torch.cat([sequence, torch.tensor([new_token_id])])
                    new_beam.append((new_sequence, new_score, hidden, cell))


            new_beam.sort(key=lambda x: x[1], reverse=True)


            beam = new_beam[:beam_width]


        completed_beams.extend([(beam[i][0], beam[i][1]) for i in range(len(beam))])


        completed_beams.sort(key=lambda x: x[1], reverse=True)

        # Return the best sequence
        return completed_beams[0][0][1:]
    
    
    
    
    
decoder_embedding_dim_bert_fr=128
class BertEncoderFrozen(nn.Module):
    def __init__(self, bert_model_name,decoder_hidden_dim):
        super(BertEncoderFrozen, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.shape_params= nn.Linear(768,decoder_hidden_dim)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.hid  = nn.Linear(768, 128)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, problems, masks):
        input_ids = problems.to(device)
        # print(f"Input id shape: {input_ids.shape}\n")
        attention_mask = masks.to(device)
        # print(input_ids, attention_mask)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
#         print(bert_output.last_hidden_state)
#         print("______________")
#         print(bert_output[0])
#         bert_output = self.shape_params(bert_output)
        
        hidden = torch.tanh(self.hid(bert_output[:,0]))
        bert_output = self.shape_params(bert_output)
        bert_out=bert_output.permute(1,0,2)
#         print(hidden.shape)
        #print(f"bert output: {bert_output.shape} | {bert_output[:, 0, :].shape}\n")
#         bert_out = bert_output.permute(1,0,2)
#         bert_out, bert_output= self.shape_params(bert_out), self.shape_params(bert_output)
        #print(f"bert output: {bert_output.shape} | bert_out.shape: {bert_output[:, 0, :].shape}\n")
#         x= bert_output.permute(0,1,2)
        cls= hidden.unsqueeze(0)
#         print(f"x shape, cls.shape: {bert_out.shape},{cls.shape}\n")
        return bert_out, cls  # output, CLS token of the last hidden state



class BertAttentionFrozen(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim *1) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #print(f"hidden = {hidden.shape}, enc_out ={encoder_outputs.shape}\n")
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
    
    
    
    
formula_vocab_reverse_bert_lstm_attn_frozen = {value: key for key, value in formula_vocab_bert_lstm_attn_frozen.items()}

decoder_embedding_dim_bert_ft=128
class BertDecoderFrozen(nn.Module):
    def __init__(self, output_dim, embedding_dim, decoder_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = decoder_hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM((decoder_embedding_dim_bert_fr * 1) + decoder_hidden_dim, decoder_hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((decoder_hidden_dim * 3), output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs, teacher_forcing_ratio):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #print(f"rnn_ip: {rnn_input.shape} | hidden:  {hidden.shape}  cell | {cell.shape}\n")
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.contiguous(), cell.contiguous()))
#         print(f"output_sq : {output.squeeze(0).shape}, weighted_sq: {weighted.squeeze(0).shape}, embedded: {embedded.squeeze(0).shape}")
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim = 1))
        return prediction, hidden, cell


class Seq2SeqBertSTMAttnFrozen(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, src, trg, mask,teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = len(formula_vocab_bert_lstm_attn_tuned)
        # print(trg_vocab_size)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out, hidden= self.encoder(src,mask)
        hidden= torch.tanh(hidden)
        cell = hidden
        # print(f"From Encoder: hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        # print("here!!\n")
        input = trg[0, :]

        # input = [batch size]
        for t in range(1, trg_length):
            # print("here!!\n")
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # print(f" hidden shape: {hidden.shape}, cell shape: {cell.shape}\n")

            # input, hidden, cell, encoder_outputs, teacher_fo hiddercing_ratio

            output, hidden, cell = self.decoder(input, hidden, cell, enc_out,teacher_forcing_ratio)
            # print(f"output shape after decoder: {output.shape},n shape: {hidden.shape}, cell shape: {cell.shape}\n")
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token

            outputs[t] = output
            # print("here")
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs    
    
    
    
    
    