import os
import sys
import json
import random
from pathlib import Path

# Add root to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from qdrant_client import QdrantClient
from app.core.config import settings
from app.generation.llm import LLMClient
from app.retrieval.retriever import COLLECTION_NAME



def get_random_chunks(limit=50):
    client = QdrantClient(url=settings.qdrant_url)
    try:
        # Scroll points to get a sample of data
        result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [r.payload["content"] for r in result if "content" in r.payload]
    except Exception as e:
        print(f"Error fetching from Qdrant: {e}")
        return []

def generate_questions(chunks: list[str], llm: LLMClient) -> list[dict]:
    dataset = []
    
    # 1. Factual Questions
    print("Generating Factual Questions...")
    for chunk in random.sample(chunks, min(10, len(chunks))):
        prompt = f"Given the following text, generate 1 factual question and its ground truth answer based strictly on the text.\n\nText: {chunk}\n\nFormat your output as a pure JSON object like this: {{\"question\": \"...\", \"ground_truth\": \"...\"}}. Do not add any markdown formatting."
        res = llm.generate(prompt=prompt, system_prompt="You are an expert dataset generator. Extract precise facts. Return ONLY JSON.")
        try:
            pair = json.loads(res.strip('` \n'))
            dataset.append({"question": pair["question"], "ground_truth": pair["ground_truth"], "type": "factual"})
        except Exception:
            pass

    # 2. Unanswerable / Adversarial Questions
    print("Generating Unanswerable Questions...")
    for chunk in random.sample(chunks, min(10, len(chunks))):
        prompt = f"Given the following text, generate 1 'unanswerable' or tricky question. The question should sound like it COULD be answered by the text, but the text actually does NOT contain the answer. The ground truth MUST state that the information is not provided.\n\nText: {chunk}\n\nFormat your output as a pure JSON object like this: {{\"question\": \"...\", \"ground_truth\": \"...\"}}. Do not add any markdown formatting."
        res = llm.generate(prompt=prompt, system_prompt="You are a tricky dataset generator. Ask questions that are out of scope. Return ONLY JSON.")
        try:
            pair = json.loads(res.strip('` \n'))
            dataset.append({"question": pair["question"], "ground_truth": pair["ground_truth"], "type": "unanswerable"})
        except Exception:
            pass

    # 3. Multi-hop Questions (Combine 2 chunks)
    print("Generating Multi-hop Questions...")
    for i in range(10):
        if len(chunks) < 2: break
        c1, c2 = random.sample(chunks, 2)
        prompt = f"Given the following TWO disconnected texts, generate 1 complex question that requires combining information from BOTH texts to answer. Provide the ground truth answer.\n\nText 1: {c1}\n\nText 2: {c2}\n\nFormat your output as a pure JSON object like this: {{\"question\": \"...\", \"ground_truth\": \"...\"}}. Do not add any markdown formatting."
        res = llm.generate(prompt=prompt, system_prompt="You generate complex multi-hop reasoning questions. Return ONLY JSON.")
        try:
            pair = json.loads(res.strip('` \n'))
            dataset.append({"question": pair["question"], "ground_truth": pair["ground_truth"], "type": "multi_hop"})
        except Exception:
            pass

    return dataset

if __name__ == "__main__":
    print("Fetching chunks from Qdrant...")
    chunks = get_random_chunks(100)
    if not chunks:
        print("No chunks found. Exiting.")
        sys.exit(1)
        
    llm = LLMClient()
    
    print(f"Loaded {len(chunks)} chunks. Generating dataset...")
    dataset = generate_questions(chunks, llm)
    
    out_path = root / "data" / "eval" / "eval_dataset_extended.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully generated {len(dataset)} questions!")
    print(f"Saved to {out_path}")
