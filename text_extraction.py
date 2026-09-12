import pymupdf

doc = pymupdf.open('Introduction to Machine Learning.pdf')

def extract_page(page):

    page_height = page.rect.height

    blocks = page.get_text("blocks")

    clean_blocks = []

    for block in blocks:

        x0, y0, x1, y1, block_text, block_no, block_type = block

        # Ignore blocks at the bottom of the page
        if y0 > page_height - 51.3:
            continue

        # Replace line breaks inside a block with spaces
        block_text = block_text.replace('\n', ' ')

        #Remove special hyphens
        block_text = block_text.replace('‐ ', '')

        # Remove extra spaces
        block_text = ' '.join(block_text.split())

        clean_blocks.append(block_text)

    return clean_blocks

def create_chunks(blocks, max_words = 200):

    chunks = []

    current_chunk = []
    current_words = 0

    for block in blocks:

        text = block['text']
        words = text.split()
        word_count = len(words)

        if current_words + word_count > max_words and current_chunk:
            chunks.append({
                "text": " ".join(block["text"] for block in current_chunk),
                'page_start' : current_chunk[0]['page'],
                'page_end' : current_chunk[-1]['page']
            })

            current_chunk = []
            current_words = 0

        current_chunk.append(block)
        current_words += word_count

    if current_chunk:
        chunks.append({
                "text": " ".join(block["text"] for block in current_chunk),
                'page_start' : current_chunk[0]['page'],
                'page_end' : current_chunk[-1]['page']
        })

    return chunks

all_blocks = []

#Extract blocks from each page and store them along with their page numbers
for page_num, page in enumerate(doc):

    blocks = extract_page(page)

    for block in blocks:

        all_blocks.append({
            "text": block,
            "page": page_num + 1
        })

#print("Total blocks:", len(all_blocks))

content_blocks = all_blocks[54:]

#Printing first 100 blocks
#for i in range(100):
#    print(f'Block {i} | Page {content_blocks[i]['page']}')
#    print(content_blocks[i]['text'])
#    print('-'*80)

chunks = create_chunks(content_blocks)

print('Number of chunks: ', len(chunks))

for i in range(10):
    print(f"\nChunk {i}")
    print(f"Pages: {chunks[i]['page_start']} - {chunks[i]['page_end']}")
    print(f"Words: {len(chunks[i]['text'].split())}")
    print(chunks[i]["text"])
    print("-" * 80)