import pymupdf
from sentence_transformers import SentenceTransformer
import chromadb

doc = pymupdf.open('Introduction to Machine Learning.pdf')

def extract_page(page, page_num):

    page_height = page.rect.height

    blocks = page.get_text("dict")["blocks"]

    clean_blocks = []

    for block in blocks:

        if "lines" not in block:
            continue

        y0 = block["bbox"][1]

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

all_blocks = []

#Extract blocks from each page of the document
for page_num, page in enumerate(doc, start = 1):

    blocks = extract_page(page, page_num )

    for block in blocks:
        all_blocks.append(block)

#Remove the starting pages blocks which are irrelevant
content_blocks = all_blocks[54:]

# Create chunks from the extracted blocks
chunks = create_chunks(content_blocks)
print("Number of chunks:", len(chunks))

# Embed the chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(chunk_texts)
print("Embedding shape:", embeddings.shape)

# Create and store the chunks in Chroma DB
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection('ml-book')
ids = []
document_text = []
metadata = []
embedding_list = []
for i, chunk in enumerate(chunks):
    ids.append(str(i))
    document_text.append(chunk['text'])
    metadata.append({
        'page_start' : chunk['page_start'],
        'page_end' : chunk['page_end']
    })
    embedding_list.append(embeddings[i].tolist())
collection.add(ids=ids, documents=document_text, metadatas=metadata, embeddings=embedding_list)
print("Documents stored:", collection.count())