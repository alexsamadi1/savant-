from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from docx import Document as DocxDocument
import re

def enrich_pdf_chunks(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()
    enriched_chunks = []

    section_pattern = re.compile(r"\n?(\d{3,4}\s+[A-Z][^\n]{3,}|[A-Z][A-Za-z\s]+\n)")

    for page_num, page in enumerate(raw_pages):
        text = page.page_content
        matches = list(section_pattern.finditer(text))
        positions = [m.start() for m in matches]

        if not positions:
            enriched_chunks.append(Document(
                page_content=text.strip(),
                metadata={"source": f"document_page_{page_num + 1}"}
            ))
            continue

        positions.append(len(text))  # end of last section

        for i in range(len(positions) - 1):
            chunk_text = text[positions[i]:positions[i + 1]].strip()
            title_line = chunk_text.split("\n")[0].strip()
            title = re.sub(r"[^\w\s:]", "", title_line)

            enriched_text = (
                f"SECTION: {title}\n"
                f"Keywords: policy, procedures, guidelines, onboarding, processes, workflows, documentation, organization.\n\n"
                f"{chunk_text}"
            )

            enriched_chunks.append(Document(
                page_content=enriched_text,
                metadata={
                    "source": f"document_page_{page_num + 1}",
                    "section_title": title
                }
            ))

    return enriched_chunks

def chunk_docx_with_metadata(docx_path: str) -> list:
    doc = DocxDocument(docx_path)

    HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3"}
    MIN_CHUNK_SIZE = 100

    current_heading = "Introduction"
    current_paragraphs = []
    sections = []  # list of (heading, paragraphs_text)

    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    para_map = {p._element: p for p in doc.paragraphs}
    table_map = {t._element: t for t in doc.tables}

    for child in doc.element.body:
        if child in para_map:
            para = para_map[child]
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style and para.style.name else ""
            if style_name in HEADING_STYLES:
                sections.append((current_heading, "\n".join(current_paragraphs)))
                current_heading = text
                current_paragraphs = []
            else:
                current_paragraphs.append(text)
        elif child in table_map:
            table = table_map[child]
            for row in table.rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        current_paragraphs.append(cell_text)

    sections.append((current_heading, "\n".join(current_paragraphs)))

    chunks = []
    for heading, body in sections:
        if not body.strip():
            continue

        enriched_text = (
            f"SECTION: {heading}\n"
            f"Keywords: policy, procedures, guidelines, onboarding, processes, workflows, documentation, organization.\n\n"
            f"{body}"
        )

        if len(body) < MIN_CHUNK_SIZE and chunks:
            chunks[-1] = Document(
                page_content=chunks[-1].page_content + "\n\n" + enriched_text,
                metadata=chunks[-1].metadata
            )
        else:
            chunks.append(Document(
                page_content=enriched_text,
                metadata={"source": "orientation_guide", "section_title": heading}
            ))

    return chunks