from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

def retrieve(query, k=5):

    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient('./chroma_db')
    collection = chroma_client.get_collection('ml-book')

    query_embedding = transformer_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    retrieved = []

    for i in range(k):
        retrieved.append({
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "page_start": results["metadatas"][0][i]["page_start"],
            "page_end": results["metadatas"][0][i]["page_end"]
        })

    return retrieved

def create_context(results):
    context = ""

    for i, result in enumerate(results):
        passage = f"""
        ----Passage {i+1}----
        Pages: {result['page_start']} -- {result['page_end']}
        {result['text']}
        """

        context += passage

    return context

def build_prompt(query, context):

    prompt = f"""
You are answering questions about a machine learning textbook.

Use only the information provided in the retrieved passages.

Answer the question clearly and concisely.

Cite the page or page range where the information came from.
If multiple retrieved passages overlap in their page ranges,
combine them into a single citation rather than repeating them.

If the passages do not contain enough information to answer the
question, say that the answer cannot be determined from the
provided passages.

Retrieved passages:

{context}

Question:
{query}

Answer:
"""

    return prompt

def ask_rag(query, k=5):

    load_dotenv()
    client = genai.Client()

    # 1. Retrieve relevant chunks
    results = retrieve(query, k)

    # 2. Build context
    context = create_context(results)

    # 3. Build prompt
    prompt = build_prompt(query, context)

    # 4. Ask Gemini
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return response.output_text

#Answer the query using the retrieved chunks and query using an LLM
query = input('Enter a query: ')
answer = ask_rag(query)
print(answer)