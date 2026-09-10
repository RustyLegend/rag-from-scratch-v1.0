from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "Regularization helps prevent overfitting."

s2 = "Regularization reduces the risk of a model overfitting to its training data."

s3 = "The Python interpreter executes code line by line and manages program execution."

s4 = "Increasing the complexity of a machine learning model can sometimes improve its performance."

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)
e4 = model.encode(s4)

print("Embedding dimensions:", len(e1))

print("e1 vs e2:", cos_sim(e1, e2))
print("e1 vs e3:", cos_sim(e1, e3))
print("e1 vs e4:", cos_sim(e1, e4))