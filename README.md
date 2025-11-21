# Word Similarity Analyzer  
A simple mini-project that calculates similarity between words using **NumPy** and **cosine similarity**.  
This project demonstrates how word embeddings can be compared using vector math — a core idea in AI and Natural Language Processing (NLP).

---

## 📌 Objective  
To analyze how similar two words are by comparing their vector representations (embeddings) using the dot product and cosine similarity.

---

## 🚀 Features  
- Creates a small dictionary of simulated word embeddings  
- Uses cosine similarity for word comparison  
- Compares multiple word pairs  
- Prints similarity scores  
- Beginner-friendly and easy to understand  

---

## 🧠 Concept Summary  

| Concept | Use in AI |
|--------|------------|
| Vectors | Represent features or word embeddings |
| Dot Product | Measures similarity |
| Cosine Similarity | Compares direction of vectors |
| Matrices | Store datasets or model weights |
| Vector Norm | Measures vector magnitude |

---

## 📝 Code  
```python
import numpy as np

embeddings = {
    "king": np.array([0.8, 0.65, 0.1]),
    "queen": np.array([0.82, 0.6, 0.12]),
    "man": np.array([0.9, 0.2, 0.1]),
    "woman": np.array([0.88, 0.18, 0.12]),
    "apple": np.array([0.1, 0.9, 0.3]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [
    ("king", "queen"),
    ("king", "man"),
    ("man", "woman"),
    ("king", "apple"),
    ("queen", "apple")
]

for w1, w2 in pairs:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(w1, "-", w2, ":", round(sim, 3))
