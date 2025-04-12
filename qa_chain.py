import numpy as np
from llama_cpp import Llama

llm = Llama(model_path="/Users/windsofoctober/Downloads/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",n_ctx=2048)

def ask_question(question, chunks, index, embed_model, top_k=2):
    q_embed = embed_model.encode([question])
    distances, indices = index.search(np.array(q_embed), top_k)
    context = "\n".join([chunks[i] for i in indices[0]])

    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = llm(prompt, max_tokens=200, stop=["\n", "Question:"])
    return response['choices'][0]['text'].strip()

