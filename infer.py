import argparse
import json
import torch
import tqdm
from models import * 
import os 


def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def infer(model, file_path, formula_reverse_vocab, device, beam_size):
    
    # print(f"{file_path}\n")
    
    with open(file_path, 'r') as f_in:
        serialized_data_dev = json.load(f_in)    
    
    # print(serialized_data_dev)
    for entry in tqdm.tqdm(serialized_data_dev):
        problem_indices = problem_to_indices_lstm_lstm(entry["Problem"])
        problem_indices = torch.LongTensor(problem_indices)

        src = problem_indices.unsqueeze(1).to(device)
        if isinstance(model, Seq2SeqBiLSTMBiLSTM):
            
            predicted_tokens = beam_search_lstm_lstm(model, src, beam_size)
        # elif isinstance(model, LSTMEncoderDecoderWithAttention):
        #     predicted_tokens = beam_search_lstm_lstm_attn(model, src, beam_size)
        # elif isinstance(model, BERTEncoderLSTMDecoderWithAttention):
        #     predicted_tokens = beam_search_bert_lstm_attn_frozen(model, src, beam_size)
        # else:
        #     raise ValueError("Invalid model type")

        predicted_formula = ''
        for tok in predicted_tokens:
            if formula_reverse_vocab[tok.item()] == '<eos>':
                break
            else:
                predicted_formula += formula_reverse_vocab[tok.item()]

        entry["predicted"] = predicted_formula
        
        with open(file_path, 'w') as f_out:
            json.dump(serialized_data_dev, f_out, indent=4)
            
            
            
            
            
            
def infer_attn(model, file_path, formula_reverse_vocab, device, beam_size):
    
    # print(f"{file_path}\n")
    
    with open(file_path, 'r') as f_in:
        serialized_data_dev = json.load(f_in)    
    
    # print(serialized_data_dev)
    for entry in tqdm.tqdm(serialized_data_dev):
        problem_indices = problem_to_indices_lstm_lstm_attn(entry["Problem"])
        problem_indices = torch.LongTensor(problem_indices)

        src = problem_indices.unsqueeze(1).to(device)
        if isinstance(model, Seq2SeqBiLSTMLSTMAttn):
            predicted_tokens = beam_search_lstm_lstm_attn(model, src, beam_size)

        predicted_formula = ''
        for tok in predicted_tokens:
            if formula_reverse_vocab[tok.item()] == '<eos>':
                break
            else:
                predicted_formula += formula_reverse_vocab[tok.item()]

        entry["predicted"] = predicted_formula
        
        with open(file_path, 'w') as f_out:
            json.dump(serialized_data_dev, f_out, indent=4)
            
            
            
            
            
def infer_bert_ft(model, file_path, formula_reverse_vocab, device, beam_size):
    
    # print(f"{file_path}\n")
    
    with open(file_path, 'r') as f_in:
        serialized_data_dev = json.load(f_in)    
        
    
    # print(serialized_data_dev)
    for entry in tqdm.tqdm(serialized_data_dev):
        
        problem= entry["Problem"]
        # formula= entry["linear-formula"]
        
        tokenized_problem = tokenizer(problem, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized_problem["input_ids"]
        attention_masks = tokenized_problem["attention_mask"]
        
        # formula_indices = formula_to_indices_bert(formula)

        problem_indices = torch.LongTensor(input_ids)

        src = problem_indices.to(device)
        attention_masks = attention_masks.to(device)
        
        if isinstance(model, Seq2SeqBertSTMAttnFineTuned):
            predicted_tokens = beam_search_bert_ft(model, src, attention_masks, beam_size)

        predicted_formula = ''
        for tok in predicted_tokens:
            if formula_reverse_vocab[tok.item()] == '<eos>':
                break
            else:
                predicted_formula += formula_reverse_vocab[tok.item()]

        entry["predicted"] = predicted_formula
        
        with open(file_path, 'w') as f_out:
            json.dump(serialized_data_dev, f_out, indent=4)
            
            
            
            
            
def infer_bert_fr(model, file_path, formula_reverse_vocab, device, beam_size):
    
    # print(f"{file_path}\n")
    
    with open(file_path, 'r') as f_in:
        serialized_data_dev = json.load(f_in)    
        
    
    # print(serialized_data_dev)
    for entry in tqdm.tqdm(serialized_data_dev):
        
        problem= entry["Problem"]
        # formula= entry["linear-formula"]
        
        tokenized_problem = tokenizer(problem, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized_problem["input_ids"]
        attention_masks = tokenized_problem["attention_mask"]
        
        # formula_indices = formula_to_indices_bert(formula)

        problem_indices = torch.LongTensor(input_ids)

        src = problem_indices.to(device)
        attention_masks = attention_masks.to(device)
        
        if isinstance(model, Seq2SeqBertSTMAttnFrozen):
            predicted_tokens = beam_search_bert_ft(model, src, attention_masks, beam_size)

        predicted_formula = ''
        for tok in predicted_tokens:
            if formula_reverse_vocab[tok.item()] == '<eos>':
                break
            else:
                predicted_formula += formula_reverse_vocab[tok.item()]

        entry["predicted"] = predicted_formula
        
        with open(file_path, 'w') as f_out:
            json.dump(serialized_data_dev, f_out, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model_file', type=str, help='Path to the trained model')
    parser.add_argument('--beam_size', type=int, choices=[1, 10, 20], help='Beam size for inference')
    parser.add_argument('--model_type', type=str, choices=['lstm_lstm', 'lstm_lstm_attn', 'bert_lstm_attn_frozen', 'bert_lstm_attn_tuned'], help='Type of model')
    parser.add_argument('--test_data_file', type=str, help='Path to the test data JSON file')

    args = parser.parse_args()

    # Load the test JSON file
    test_data = load_data(args.test_data_file)
    
    formula_reverse_vocab_lstm_lstm={}

    # Load the model
    if args.model_type == 'lstm_lstm':
        input_dim = len(problem_vocab_lstm_lstm)
        output_dim = len(formula_vocab_lstm_lstm)
        encoder_embedding_dim = 300
        decoder_embedding_dim = 512
        hidden_dim = 1024
        n_layers = 1
        encoder_dropout = 0
        decoder_dropout = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = BiLSTMEncoder(input_dim,encoder_embedding_dim,hidden_dim,n_layers,encoder_dropout)

        decoder = BiLSTMDecoder(output_dim,decoder_embedding_dim,hidden_dim,n_layers,decoder_dropout)

        model = Seq2SeqBiLSTMBiLSTM(encoder, decoder, device).to(device)
        print(model.load_state_dict(torch.load(args.model_file)))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        formula_reverse_vocab_lstm_lstm= formula_vocab_reverse_lstm_lstm
        beam_size= args.beam_size 
        # Perform inference
        if args.beam_size==1:
            beam_size= 2
        infer(model, args.test_data_file, formula_reverse_vocab_lstm_lstm, device, beam_size)

    elif args.model_type== 'lstm_lstm_attn':
        
        input_dim = len(problem_vocab_lstm_lstm_attn)
        output_dim = len(formula_vocab_lstm_lstm_attn)
        encoder_embedding_dim = 100
        decoder_embedding_dim = 512
        encoder_hidden_dim= 512
        decoder_hidden_dim= 512

        n_layers = 1
        encoder_dropout = 0
        decoder_dropout = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_dim, embedding_dim, hidden_dim, n_layers, dropout
        encoder = BiLSTMEncoderAttn(input_dim, encoder_embedding_dim, encoder_hidden_dim, n_layers, encoder_dropout)
        # enc_hid_dim, dec_hid_dim
        attention = BiLSTMAttention(encoder_hidden_dim, decoder_hidden_dim)
        # output_dim, embedding_dim, hidden_dim, n_layers, dropout, attention
        decoder = BiLSTMDecoderAttn(output_dim, decoder_embedding_dim, decoder_hidden_dim, n_layers, decoder_dropout, attention)

        model_attn =Seq2SeqBiLSTMLSTMAttn(encoder, decoder, device).to(device)
        
        print(model_attn.load_state_dict(torch.load(args.model_file)))
        model_attn.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_attn.to(device)
        
        formula_reverse_vocab_lstm_lstm_attn= formula_vocab_reverse_lstm_lstm_attn
        beam_size= args.beam_size 
        # Perform inference
        if args.beam_size==1:
            beam_size= 2
        infer_attn(model_attn, args.test_data_file, formula_reverse_vocab_lstm_lstm_attn, device, beam_size)
        
        
    elif args.model_type== 'bert_lstm_attn_tuned':
        input_dim = len(problem_vocab_bert_lstm_attn_tuned)
        output_dim = len(formula_vocab_bert_lstm_attn_tuned)
        encoder_embedding_dim = 768
        decoder_embedding_dim = 128
        encoder_hidden_dim= 128
        decoder_hidden_dim= 128

        n_layers = 1
        encoder_dropout = 0
        decoder_dropout = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_dim, embedding_dim, hidden_dim, n_layers, dropout
        encoder = BertEncoderFineTuned("bert-base-cased", decoder_hidden_dim)
        # enc_hid_dim, dec_hid_dim
        attention = BertAttentionFineTuned(encoder_hidden_dim, decoder_hidden_dim)
        # output_dim, embedding_dim, hidden_dim, n_layers, dropout, attention
        decoder = BertDecoderFineTuned(output_dim, decoder_embedding_dim, decoder_hidden_dim, n_layers, decoder_dropout, attention)

        model_bert_ft = Seq2SeqBertSTMAttnFineTuned(encoder, decoder, device).to(device)
        print("I am inside here!!")

        print(model_bert_ft.load_state_dict(torch.load(args.model_file)))
        model_bert_ft.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_bert_ft.to(device)
        
        
        beam_size= args.beam_size 
        # Perform inference
        if args.beam_size==1:
            beam_size= 2
        infer_bert_ft(model_bert_ft, args.test_data_file, formula_vocab_reverse_bert_lstm_attn_tuned, device, beam_size)
        
    elif args.model_type== 'bert_lstm_attn_frozen':
        input_dim = len(problem_vocab_bert_lstm_attn_tuned)
        output_dim = len(formula_vocab_bert_lstm_attn_tuned)
        encoder_embedding_dim = 768
        decoder_embedding_dim = 128
        encoder_hidden_dim= 128
        decoder_hidden_dim= 128

        n_layers = 1
        encoder_dropout = 0
        decoder_dropout = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_dim, embedding_dim, hidden_dim, n_layers, dropout
        encoder = BertEncoderFrozen("bert-base-cased", decoder_hidden_dim)
        # enc_hid_dim, dec_hid_dim
        attention = BertAttentionFineTuned(encoder_hidden_dim, decoder_hidden_dim)
        # output_dim, embedding_dim, hidden_dim, n_layers, dropout, attention
        decoder = BertDecoderFrozen(output_dim, decoder_embedding_dim, decoder_hidden_dim, n_layers, decoder_dropout, attention)

        model_bert_fr = Seq2SeqBertSTMAttnFrozen(encoder, decoder, device).to(device)
        print("I am inside here!!")

        print(model_bert_fr.load_state_dict(torch.load(args.model_file)))
        model_bert_fr.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_bert_fr.to(device)
        
        
        beam_size= args.beam_size 
        # Perform inference
        if args.beam_size==1:
            beam_size= 2
        infer_bert_fr(model_bert_fr, args.test_data_file, formula_vocab_reverse_bert_lstm_attn_tuned, device, beam_size)
            
        
    
    