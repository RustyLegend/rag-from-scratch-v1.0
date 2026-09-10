from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('all-MiniLM-L6-v2')

text = """
Machine learning models can suffer from overfitting. Overfitting
occurs when a model learns the training data too closely and
performs poorly on unseen data.

Regularization is a technique used to reduce overfitting. It
adds a penalty to the model's objective function, discouraging
excessive model complexity.

There are two common types of regularization: L1 and L2.
L1 regularization can produce sparse models by driving some
feature coefficients to zero, while L2 regularization tends
to shrink coefficients without necessarily making them zero.
"""

chunks = []
chunks_size = 100
for i in range(0, len(text), chunks_size):
    chunks.append(text[i : i + chunks_size])

para_chunks = text.strip().split('\n\n')

query = "How does L1 regularization affect feature coefficients?"

embeddings = []
for chunk in chunks:
    embeddings.append(model.encode(chunk))

para_embeddings = []
for chunk in para_chunks:
    para_embeddings.append(model.encode(chunk))

encoded_query = model.encode(query)

similarity = []
for i, embedding in enumerate(embeddings, start = 1):
    sim = cos_sim(encoded_query, embedding).item()
    similarity.append((sim, i))

para_similarity = []
for i, embedding in enumerate(para_embeddings, start = 1):
    sim = cos_sim(encoded_query, embedding).item()
    para_similarity.append((sim, i))

similarity.sort(reverse=True)
para_similarity.sort(reverse=True)

print('----------100 Characters-----------')
for sim, i in similarity:
    print('Chunk: ', chunks[i-1])
    print(f'Chunk {i} | Similarity {sim}')
    print('-----------------------------')
print('------------------------------')

print('---------Paragraphs-----------')
for sim, i in para_similarity:
    print('Chunk: ', para_chunks[i-1])
    print(f'Chunk {i} | Similarity {sim}')
    print('-----------------------------')
print('------------------------------')