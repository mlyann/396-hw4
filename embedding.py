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


def generate_average_embeddings(tokenized_texts, output_file="result_partial.csv"):
    token_embeddings = defaultdict(list)
    batch_results = []

    with torch.no_grad():
        for batch_inputs in tqdm(tokenized_texts, desc="Generating Embeddings"):
            outputs = model(**batch_inputs)
            hidden_states = outputs.last_hidden_state.cpu().numpy()

            for input_ids, hidden_state in zip(batch_inputs['input_ids'], hidden_states):
                tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())
                for token, embedding in zip(tokens, hidden_state):
                    token_embeddings[token].append(embedding)

        # Write intermediate results to disk
        batch_results.append({token: np.mean(embeds, axis=0).tolist() for token, embeds in token_embeddings.items()})
        pd.DataFrame(batch_results).to_csv(output_file, mode='a', index=False, header=False)
        token_embeddings.clear()  # Free memory


def main():
    dataset_path = 'assignment4-dataset.txt'
    texts = load_dataset(dataset_path, sample_size=1000000)
    tokenized_texts = tokenize_texts(texts)
    average_embeddings = generate_average_embeddings(tokenized_texts)

    embedding_data = {'Token': list(average_embeddings.keys()), 
                      'Embedding': [embedding.tolist() for embedding in average_embeddings.values()]}
    df = pd.DataFrame(embedding_data)
    df.to_csv('result.csv', index=False)

if __name__ == "__main__":
    main()
