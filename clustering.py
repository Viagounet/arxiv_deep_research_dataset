import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE # For visualization
import matplotlib.pyplot as plt # For visualization
import os # To check file existence

# --- Configuration ---
JSON_FILE_PATH = "export/initial_questions.json"
# Choose a pre-trained model (other options: 'paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# Define the desired number of clusters (k)
# Finding the 'optimal' k often requires experimentation (Elbow method, Silhouette score)
NUM_CLUSTERS = 300 # <<-- ADJUST THIS VALUE AS NEEDED
RANDOM_STATE = 42 # For reproducibility

# --- 1. Load the data ---
print(f"Loading data from {JSON_FILE_PATH}...")
if not os.path.exists(JSON_FILE_PATH):
    print(f"Error: File not found at {JSON_FILE_PATH}")
    exit()

try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        arxiv_data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# --- 2. Extract all questions ---
print("Extracting questions...")
all_questions = []
original_indices = [] # Keep track of which paper each question came from (optional)
questions_mapping = {}
for i, entry in enumerate(arxiv_data):
    if 'questions' in entry and isinstance(entry['questions'], list):
        for question in entry['questions']:
            if isinstance(question, str) and question.strip(): # Basic validation
                 all_questions.append(question.strip())
                 original_indices.append(i) # Store index of the paper
                 questions_mapping[question] = entry["id"]
    # else:
    #     print(f"Warning: Entry {i} has no 'questions' list or it's not a list.")


if not all_questions:
    print("Error: No valid questions found in the data.")
    exit()

print(f"Found {len(all_questions)} questions.")

# --- 3. Load Sentence Transformer model ---
print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    exit()

# --- 4. Encode the questions ---
print("Encoding questions into embeddings (this may take a while)...")
# The encode function returns a NumPy array of embeddings
embeddings = model.encode(all_questions, show_progress_bar=True)
print(f"Embeddings generated with shape: {embeddings.shape}") # Shape: (num_questions, embedding_dimension)

# --- 5. Apply K-Means ---
print(f"Applying K-Means clustering with k={NUM_CLUSTERS}...")
# n_init='auto' is recommended in recent scikit-learn versions
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init='auto')

# Fit K-Means to the embeddings
kmeans.fit(embeddings)

# Get the cluster label assigned to each question
cluster_labels = kmeans.labels_

# --- 6. Analyze and display results ---
print("\n--- Clustering Results ---")
clustered_questions = {i: [] for i in range(NUM_CLUSTERS)}
for question, label in zip(all_questions, cluster_labels):
    clustered_questions[label].append(question)

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

clustered_dataset = []
for cluster_num, questions_in_cluster in clustered_questions.items():
    print(f"{cluster_num} / {NUM_CLUSTERS}")
    questions_in_cluster_string = "\n".join(questions_in_cluster)
    completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "developer", "content": "Your role will be to generate a more high-level question that encompasses all the other questions provided by the user. You will answer ONLY with the question and nothing else. Your question will be phrase in an easy to understand and rather short format"},
                        {"role": "user", "content": f"Fine-grained questions: \n\n{questions_in_cluster_string}"}
                    ]
                    )
    cluster_data = {"id": cluster_num, "main_question": completion.choices[0].message.content, "finegrained_questions": questions_in_cluster, "relevant_arxiv_ids": []}
    relevant_ids = []
    for i, question in enumerate(questions_in_cluster):
        relevant_ids.append(questions_mapping[question])
    relevant_ids = list(set(relevant_ids))
    cluster_data["relevant_arxiv_ids"] = relevant_ids
    clustered_dataset.append(cluster_data)
    with open("export/cluster_data.json", "w", encoding="utf-8") as f:
        json.dump(clustered_dataset, f, ensure_ascii=False, indent=4)
# --- 7. (Optional) Visualize using t-SNE ---
print("\nGenerating t-SNE visualization...")

# Reduce dimensionality using t-SNE
# Note: t-SNE can be slow on large datasets. PCA is faster but might capture less nuance.
# Adjust perplexity based on dataset size, typically 5-50. Must be less than N.
perplexity_value = min(30, len(all_questions) - 1)
if perplexity_value > 0:
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity_value, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title(f't-SNE visualization of Question Clusters (k={NUM_CLUSTERS})')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    # Create a legend - might be crowded if NUM_CLUSTERS is large
    if NUM_CLUSTERS <= 20: # Only show legend for a reasonable number of clusters
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(NUM_CLUSTERS)], title="Clusters")
    plt.grid(True)
    print("Showing plot...")
    plt.show()
else:
    print("Skipping t-SNE visualization: Not enough data points.")

print("\nClustering process complete.")