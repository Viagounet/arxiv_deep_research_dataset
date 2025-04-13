import json, os, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI

# --- Config ---
IN_FILE = "export/initial_questions.json"
OUT_FILE = "export/clustered_questions_shortest_v2.json"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
NUM_CLUSTERS = 300 # Adjust if needed
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Must be set

print("Starting clustering...")

# --- Load & Extract ---
try:
    with open(IN_FILE, "r", encoding="utf-8") as f: data = json.load(f)
    questions = [q.strip() for e in data if isinstance(e.get('questions'), list) for q in e['questions'] if isinstance(q, str) and q.strip()]
    q_to_id = {q: e['id'] for e in data if isinstance(e.get('questions'), list) and 'id' in e for q in e['questions'] if isinstance(q, str) and q.strip()}
    if not questions: raise ValueError("No questions found")
except Exception as e: print(f"Error loading/extracting: {e}"); exit()

# --- Encode & Cluster ---
try:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(questions)
    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(questions)), random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
except Exception as e: print(f"Error encoding/clustering: {e}"); exit()

# --- Summarize & Structure Results (Requires OpenAI API Key) ---
results = []
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    questions_by_cluster = [[] for _ in range(kmeans.n_clusters)]
    for q, lbl in zip(questions, labels): questions_by_cluster[lbl].append(q)

    for i, cluster_qs in enumerate(questions_by_cluster):
        if not cluster_qs: continue
        summary = f"Cluster {i} Summary Placeholder"
        try:
            res = client.chat.completions.create(
                model="gpt-4o", # Or other suitable model
                messages=[{"role": "system", "content": "Summarize these questions into one short question:"},
                          {"role": "user", "content": "\n".join(f"- {q}" for q in cluster_qs)}]
            )
            summary = res.choices[0].message.content.strip()
        except Exception as e: print(f"Warn: OpenAI call failed cluster {i}: {e}")
        ids = list(set(q_to_id.get(q) for q in cluster_qs if q_to_id.get(q)))
        results.append({"id": i, "summary": summary, "questions": cluster_qs, "arxiv_ids": ids})
else:
    print("Warning: OPENAI_API_KEY not set. Skipping summaries.")
    # Basic structure without summaries
    questions_by_cluster = [[] for _ in range(kmeans.n_clusters)]
    for q, lbl in zip(questions, labels): questions_by_cluster[lbl].append(q)
    for i, cluster_qs in enumerate(questions_by_cluster):
         if not cluster_qs: continue
         ids = list(set(q_to_id.get(q) for q in cluster_qs if q_to_id.get(q)))
         results.append({"id": i, "summary": "N/A (API key missing)", "questions": cluster_qs, "arxiv_ids": ids})

# --- Save ---
try:
    with open(OUT_FILE, "w", encoding="utf-8") as f: json.dump(results, f, indent=4)
    print(f"Clustering complete. Results saved to {OUT_FILE}")
except Exception as e: print(f"Error saving results: {e}")