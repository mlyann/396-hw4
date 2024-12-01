import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import csv

# Step 1: Load Dataset and Vocabulary
def load_files(dataset_file, vocab_file, max_lines=10000):
    with open(dataset_file, "r", encoding="utf-8") as file:
        dataset = [next(file).strip() for _ in range(max_lines)]

    with open(vocab_file, "r", encoding="utf-8") as file:
        vocabulary = file.read().splitlines()
    
    return dataset, vocabulary

# Step 2: Tokenize Dataset
def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt")
    return tokenized_dataset

# Step 3: Generate Contextualized Embeddings
def generate_contextualized_embeddings(tokenized_dataset, model):
    with torch.no_grad():
        outputs = model(**tokenized_dataset)
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    return embeddings

# Step 4: Compute Static Embeddings (with tqdm)
def compute_static_embeddings(tokenized_dataset, embeddings, tokenizer):
    vocab_embeddings = {}
    token_counts = {}

    for idx, tokens in tqdm(enumerate(tokenized_dataset["input_ids"]), total=len(tokenized_dataset["input_ids"]), desc="Processing tokens"):
        for i, token_id in enumerate(tokens):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token not in vocab_embeddings:
                vocab_embeddings[token] = embeddings[idx][i].numpy()
                token_counts[token] = 1
            else:
                vocab_embeddings[token] += embeddings[idx][i].numpy()
                token_counts[token] += 1

    for token in tqdm(vocab_embeddings, desc="Averaging embeddings"):
        vocab_embeddings[token] /= token_counts[token]

    return vocab_embeddings

# Step 5: Save Embeddings to CSV
def save_embeddings_to_csv(vocab_embeddings, output_file="result.csv"):
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "Embedding"])
        for token, embedding in vocab_embeddings.items():
            writer.writerow([token, embedding.tolist()])

# Step 6: Implement Most Similar Words
def most_similar_words(word, vectors, index_to_key, key_to_index, topn=10):
    word_id = key_to_index.get(word)
    if word_id is None:
        return f"'{word}' not in vocabulary."

    emb = vectors[word_id]
    similarities = vectors @ emb
    ids_descending = similarities.argsort()[::-1]
    ids_descending = ids_descending[ids_descending != word_id]
    top_ids = ids_descending[:topn]
    return [(index_to_key[i], similarities[i]) for i in top_ids]

# Main Script
if __name__ == "__main__":
    dataset_file = "assignment4-dataset.txt"
    vocab_file = "glove.6B.300d-vocabulary.txt"

    # Load files (limit to 1 million lines)
    dataset, vocabulary = load_files(dataset_file, vocab_file)

    # Choose model
    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Generate embeddings
    contextualized_embeddings = generate_contextualized_embeddings(tokenized_dataset, model)

    # Compute static embeddings
    vocab_embeddings = compute_static_embeddings(tokenized_dataset, contextualized_embeddings, tokenizer)

    # Save embeddings to result.csv
    save_embeddings_to_csv(vocab_embeddings, output_file="result.csv")

    # Prepare vectors and indices for most_similar_words
    index_to_key = list(vocab_embeddings.keys())
    key_to_index = {token: idx for idx, token in enumerate(index_to_key)}
    vectors = np.array(list(vocab_embeddings.values()))

    # Test words for most similar words
    test_words = ["cactus", "cake", "angry", "quickly", "between", "the"]
    for word in test_words:
        similar_words = most_similar_words(word, vectors, index_to_key, key_to_index)
        print(f"Most similar to '{word}': {similar_words}")
