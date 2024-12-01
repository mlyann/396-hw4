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



# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import numpy as np
# import csv

# # Step 1: Load Dataset and Vocabulary
# def load_files(dataset_file, vocab_file, max_lines=10000):
#     with open(dataset_file, "r", encoding="utf-8") as file:
#         dataset = [next(file).strip() for _ in range(max_lines)]

#     with open(vocab_file, "r", encoding="utf-8") as file:
#         vocabulary = file.read().splitlines()
    
#     return dataset, vocabulary

# # Step 2: Tokenize Dataset
# def tokenize_dataset(dataset, tokenizer):
#     tokenized_dataset = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt")
#     return tokenized_dataset

# # Step 3: Generate Contextualized Embeddings
# def generate_contextualized_embeddings(tokenized_dataset, model):
#     with torch.no_grad():
#         outputs = model(**tokenized_dataset)
#         embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
#     return embeddings

# # Step 4: Compute Static Embeddings (with tqdm)
# def compute_static_embeddings(tokenized_dataset, embeddings, tokenizer):
#     vocab_embeddings = {}
#     token_counts = {}

#     for idx, tokens in tqdm(enumerate(tokenized_dataset["input_ids"]), total=len(tokenized_dataset["input_ids"]), desc="Processing tokens"):
#         for i, token_id in enumerate(tokens):
#             token = tokenizer.convert_ids_to_tokens(token_id)
#             if token not in vocab_embeddings:
#                 vocab_embeddings[token] = embeddings[idx][i].numpy()
#                 token_counts[token] = 1
#             else:
#                 vocab_embeddings[token] += embeddings[idx][i].numpy()
#                 token_counts[token] += 1

#     for token in tqdm(vocab_embeddings, desc="Averaging embeddings"):
#         vocab_embeddings[token] /= token_counts[token]

#     return vocab_embeddings

# # Step 5: Save Embeddings to CSV
# def save_embeddings_to_csv(vocab_embeddings, output_file="result.csv"):
#     with open(output_file, "w", encoding="utf-8", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Token", "Embedding"])
#         for token, embedding in vocab_embeddings.items():
#             writer.writerow([token, embedding.tolist()])

# # Step 6: Implement Most Similar Words
# def most_similar_words(word, vectors, index_to_key, key_to_index, topn=10):
#     word_id = key_to_index.get(word)
#     if word_id is None:
#         return f"'{word}' not in vocabulary."

#     emb = vectors[word_id]
#     similarities = vectors @ emb
#     ids_descending = similarities.argsort()[::-1]
#     ids_descending = ids_descending[ids_descending != word_id]
#     top_ids = ids_descending[:topn]
#     return [(index_to_key[i], similarities[i]) for i in top_ids]

# # Main Script
# if __name__ == "__main__":
#     dataset_file = "assignment4-dataset.txt"
#     vocab_file = "glove.6B.300d-vocabulary.txt"

#     # Load files (limit to 1 million lines)
#     dataset, vocabulary = load_files(dataset_file, vocab_file)

#     # Choose model
#     model_name = "FacebookAI/roberta-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     # Tokenize dataset
#     tokenized_dataset = tokenize_dataset(dataset, tokenizer)

#     # Generate embeddings
#     contextualized_embeddings = generate_contextualized_embeddings(tokenized_dataset, model)

#     # Compute static embeddings
#     vocab_embeddings = compute_static_embeddings(tokenized_dataset, contextualized_embeddings, tokenizer)

#     # Save embeddings to result.csv
#     save_embeddings_to_csv(vocab_embeddings, output_file="result.csv")

#     # Prepare vectors and indices for most_similar_words
#     index_to_key = list(vocab_embeddings.keys())
#     key_to_index = {token: idx for idx, token in enumerate(index_to_key)}
#     vectors = np.array(list(vocab_embeddings.values()))

#     # Test words for most similar words
#     test_words = ["cactus", "cake", "angry", "quickly", "between", "the"]
#     for word in test_words:
#         similar_words = most_similar_words(word, vectors, index_to_key, key_to_index)
#         print(f"Most similar to '{word}': {similar_words}")
