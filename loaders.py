from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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
                metadata={"source": f"employee_handbook_page_{page_num + 1}"}
            ))
            continue

        positions.append(len(text))  # end of last section

        for i in range(len(positions) - 1):
            chunk_text = text[positions[i]:positions[i + 1]].strip()
            title_line = chunk_text.split("\n")[0].strip()
            title = re.sub(r"[^\w\s:]", "", title_line)

            enriched_text = (
                f"SECTION: {title}\n"
                f"Keywords: vacation, PTO, benefits, remote work, telecommute, timecard, leave, supervisor, holiday, HR, policy.\n\n"
                f"{chunk_text}"
            )

            enriched_chunks.append(Document(
                page_content=enriched_text,
                metadata={
                    "source": f"employee_handbook_page_{page_num + 1}",
                    "section_title": title
                }
            ))

    return enriched_chunks

def chunk_docx_with_metadata(docx_path: str) -> list:
    loader = UnstructuredWordDocumentLoader(docx_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = "orientation_guide"
    return chunks