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


all_blocks = []

for page_num, page in enumerate(doc):

    blocks = extract_page(page)

    for block in blocks:

        all_blocks.append({
            "text": block,
            "page": page_num + 1
        })

print("Total blocks:", len(all_blocks))

b = extract_page(doc[14])
print(all_blocks[:5])