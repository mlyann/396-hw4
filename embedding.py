import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

model_name = 'FacebookAI/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token, use_fast=True)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token).to(device)

def load_dataset(file_path, sample_size=100):
    with open(file_path, 'r') as file:
        texts = file.readlines()
    texts = [text.strip() for text in texts]
    return np.random.choice(texts, sample_size, replace=False).tolist()

def tokenize_texts(texts, batch_size=16):
    tokenized_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing Texts"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        tokenized_texts.append(inputs)
    return tokenized_texts

def generate_average_embeddings(tokenized_texts):
    token_embeddings = defaultdict(list)
    
    with torch.no_grad():
        for inputs in tqdm(tokenized_texts, desc="Generating Embeddings"):
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state 
            
            for i in range(inputs['input_ids'].shape[0]):
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][i].cpu().numpy())
                hidden_state = hidden_states[i].cpu().numpy()
                
                for j, token in enumerate(tokens):
                    token_embeddings[token].append(hidden_state[j])
    
    average_embeddings = {token: np.mean(embeds, axis=0) for token, embeds in token_embeddings.items()}
    return average_embeddings

def main():
    dataset_path = 'assignment4-dataset.txt'
    texts = load_dataset(dataset_path, sample_size=100000000)
    tokenized_texts = tokenize_texts(texts)
    average_embeddings = generate_average_embeddings(tokenized_texts)

    embedding_data = {'Token': list(average_embeddings.keys()), 
                      'Embedding': [embedding.tolist() for embedding in average_embeddings.values()]}
    df = pd.DataFrame(embedding_data)
    df.to_csv('result.csv', index=False)

if __name__ == "__main__":
    main()
