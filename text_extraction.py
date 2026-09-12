import pymupdf
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

doc = pymupdf.open('Introduction to Machine Learning.pdf')

def extract_page(page, page_num):

    page_height = page.rect.height

    blocks = page.get_text("dict")["blocks"]

    clean_blocks = []

    for block in blocks:

        if "lines" not in block:
            continue

        x0, y0, x1, y1 = block["bbox"]

        # Ignore blocks at the bottom of the page
        if y0 > page_height - 51.3:
            continue

        # Get text and font size
        block_text = ""
        font_sizes = []

        for line in block["lines"]:
            for span in line["spans"]:
                block_text += span["text"]
                font_sizes.append(span["size"])

        # Remove special hyphenation
        block_text = block_text.replace('‐ ', '')

        # Replace line breaks / extra whitespace
        block_text = ' '.join(block_text.split())

        if not block_text:
            continue

        clean_blocks.append({
            "text": block_text,
            "page": page_num,
            "font_size": max(font_sizes)
        })

    return clean_blocks

def create_chunks(blocks, max_words=200):

    chunks = []

    current_chunk = []
    current_words = 0

    for block in blocks:

        text = block["text"]
        word_count = len(text.split())

        # If this is a heading, finish the previous chunk
        if is_heading(block) and current_chunk:

            chunks.append({
                "text": " ".join(block["text"] for block in current_chunk),
                "page_start": current_chunk[0]["page"],
                "page_end": current_chunk[-1]["page"]
            })

            current_chunk = []
            current_words = 0

        # If adding this block would exceed the limit
        if current_words + word_count > max_words and current_chunk:

            chunks.append({
                "text": " ".join(block["text"] for block in current_chunk),
                "page_start": current_chunk[0]["page"],
                "page_end": current_chunk[-1]["page"]
            })

            current_chunk = []
            current_words = 0

        current_chunk.append(block)
        current_words += word_count

    # Add remaining blocks
    if current_chunk:

        chunks.append({
            "text": " ".join(block["text"] for block in current_chunk),
            "page_start": current_chunk[0]["page"],
            "page_end": current_chunk[-1]["page"]
        })

    return chunks

def is_heading(block):
     return block["font_size"] > 15

def retrieve(query, chunks, embeddings, k=5):

    query_embedding = model.encode(query)

    similarities = []

    for i, embedding in enumerate(embeddings):

        similarity = cos_sim(query_embedding, embedding).item()

        similarities.append((similarity, i))

    similarities.sort(reverse=True)

    results = []

    for similarity, i in similarities[:k]:

        results.append({
            "text": chunks[i]["text"],
            "similarity": similarity,
            "page_start": chunks[i]["page_start"],
            "page_end": chunks[i]["page_end"]
        })

    return results

all_blocks = []

for page_num, page in enumerate(doc, start = 1):

    blocks = extract_page(page, page_num )

    for block in blocks:
        all_blocks.append(block)

content_blocks = all_blocks[54:]

chunks = create_chunks(content_blocks)

model = SentenceTransformer('all-MiniLM-L6-v2')

chunk_texts = [chunk['text'] for chunk in chunks]

embeddings = model.encode(chunk_texts)

query = input('Enter query: ')

results = retrieve(query, chunks, embeddings)

for result in results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Pages: {result['page_start']} - {result['page_end']}")
    print(result["text"])
    print("-" * 80)