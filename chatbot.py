
from file_loader import load_pdf_text
from embedder import embed_and_index
from qa_chain import ask_question
import os
import sys

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "uploads/sample.pdf"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        print("Loading and processing file...")
        text = load_pdf_text(file_path)
        chunks, index, embed_model = embed_and_index(text)
    except Exception as e:
        print("Error during setup:", e)
        return

    print("Chatbot is ready. Type your question or 'exit' to quit.")
    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        try:
            answer = ask_question(q, chunks, index, embed_model)
            print("Answer:", answer)
        except Exception as e:
            print("Error processing question:", e)

if __name__ == "__main__":
    main()
