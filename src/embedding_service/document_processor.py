# PDF -> extractable content. Two independent pipelines live here:
#   - pdf_to_images / image_to_base64: legacy, page-as-image pipeline (see
#     src/embedding_service/main.py). Not used by the live text-chunk pipeline.
#   - pdf_to_chunks: the live pipeline, used by watcher.py. Uses unstructured.io to pull
#     text (via OCR, since these are scanned PDFs) out of each page and group it into
#     semantically coherent chunks ready for embedding.
import base64
import os
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# unstructured's OCR page-rendering path reads this DPI from the environment (it ignores
# the pdf_image_dpi= argument passed to partition_pdf below) and defaults to 350. On the
# MIG29 manual's oversized pages (2000x2999 pts) that renders each page to a ~425MB bitmap,
# and pages are rendered in batches of 10 -- easily exhausting RAM on a modest machine.
# Lowering it here keeps OCR usable while staying well under available memory.
os.environ.setdefault("PDF_RENDER_DPI", "100")

def pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of PIL Images, one per page.
    """
    print(f"Loading PDF and converting to images: {pdf_path}")
    # You might need poppler installed on the system (e.g., brew install poppler)
    images = convert_from_path(pdf_path, dpi=150)
    print(f"Successfully converted {len(images)} pages to images.")
    
    
    resized_images = []
    for idx, img in enumerate(images):
        w, h = img.size
        max_dim = 1500
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_images.append(img)
    return resized_images

def image_to_base64(image):
    """
    Converts a PIL Image to a Base64 encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def pdf_to_chunks(pdf_path, max_characters=1500, new_after_n_chars=1200):
    """
    Partitions a PDF into text/table elements using unstructured.io and groups
    them into semantically coherent chunks (grouped by section title).
    Returns a list of dicts: {"text": str, "page_number": int | None}.
    """
    print(f"Partitioning PDF with unstructured.io: {pdf_path}")
    # strategy="ocr_only" runs Tesseract directly on each page image. Used because the
    # MIG29 flight manual is a scanned/image-only PDF with no embedded text layer, so the
    # lighter "fast" (pdfminer) strategy extracts nothing. Avoids "hi_res", which would
    # also download a layout-detection model on first use. (pdf_image_dpi=100 here is
    # belt-and-suspenders: the OCR page-rendering path actually reads DPI from the
    # PDF_RENDER_DPI env var set above, not this argument -- see that comment for why.)
    elements = partition_pdf(filename=pdf_path, strategy="ocr_only", pdf_image_dpi=100)
    print(f"Extracted {len(elements)} elements.")

    # chunk_by_title groups consecutive elements (paragraphs, list items, etc.) under the
    # same section heading into one chunk, splitting when a new heading appears or a chunk
    # would exceed max_characters. This keeps each embedded chunk topically coherent
    # instead of splitting mid-thought at an arbitrary character count.
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
    )
    print(f"Grouped into {len(chunks)} chunks.")

    results = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        page_number = getattr(chunk.metadata, "page_number", None) if chunk.metadata else None
        results.append({"text": text, "page_number": page_number})
    return results
