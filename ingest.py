"""Build the vector index from everything in data/ (PDFs + Markdown).

Run again whenever you add or change files in data/:
    python ingest.py
"""
import re
from pathlib import Path

import chromadb
import ollama
import pymupdf
# pymupdf4llm is imported lazily in pdf_pages(): importing it switches PyMuPDF into a
# layout mode that strips spaces from table cells, so all tables are extracted first.

DATA_DIR = Path("data")
DB_DIR = "db"
COLLECTION = "ditm"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1200   # characters
OVERLAP = 200


def clean(text: str) -> str:
    text = text.replace("**", "").replace("<br>", " ")
    text = re.sub(r"\|(\s*\|)+", "|", text)        # collapse empty table cells
    text = re.sub(r"^\|?-{3,}.*$", "", text, flags=re.M)  # drop table separator rows
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_heading(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") or bool(re.match(r"^\*\*\d+\.\s+\S", s))


def join_cell(cell) -> str:
    """Re-join a table cell's wrapped lines: 'BSC-\\nMATH-\\n101G' -> 'BSC-MATH-101G',
    separate items in a list cell with '; '."""
    out = ""
    for piece in (p.strip() for p in str(cell or "").split("\n")):
        if not piece:
            continue
        if not out or out.endswith("-"):
            out += piece
        elif (re.search(r"(\band|&|\bof|\bfor|\bin|\bto|\bwith|[(,/])$", out)
              or piece[0].islower() or piece[0] in "()&"):
            out += " " + piece
        else:
            out += "; " + piece
    return out


def merge_cells(a, b):
    return [f"{x}; {y}".strip("; ") for x, y in zip(a, b)]


def table_groups(rows, state):
    """Group table rows by their first column. Empty first-column cells (merged cells,
    or a row continued from the previous page) inherit the value above."""
    rows = [[join_cell(c) for c in r] for r in rows]
    if not rows:
        return None
    if any(rows[0]) and rows[0][0] != state.get("carry"):
        state["header"], rows = [h.replace("; ", " ") for h in rows[0]], rows[1:]
    header = state.get("header") or [f"Col{i + 1}" for i in range(len(rows[0]) if rows else 0)]
    groups = []  # {"key", "cells", "continued"}
    for r in rows:
        if groups and (not r[0] or r[0] == groups[-1]["key"]):
            groups[-1]["cells"] = merge_cells(groups[-1]["cells"], r[1:])
        else:
            key = r[0] or state.get("carry", "")
            groups.append({"key": key, "cells": r[1:], "continued": not r[0]})
        state["carry"] = groups[-1]["key"]
    return {"header": header, "groups": groups}


def render_table(table) -> str:
    header = table["header"]
    code_col = next((i for i, h in enumerate(header) if "code" in h.lower()), None)
    name_col = next((i for i, h in enumerate(header) if "name" in h.lower()), None)
    lines = []
    for g in table["groups"]:
        cells = [g["key"]] + g["cells"]
        fields = list(zip(header, cells))
        if code_col is not None and name_col is not None:
            # Syllabus-style table: pair each course code with its subject name
            codes, names = cells[code_col].split("; "), cells[name_col].split("; ")
            if len(codes) == len(names):
                subjects = "; ".join(f"{n} ({c})" for c, n in zip(codes, names))
            else:  # wrapped names didn't line up with codes; keep both lists
                subjects = f"{cells[name_col]} (course codes: {cells[code_col]})"
            fields = [(h, v) for i, (h, v) in enumerate(fields) if i not in (code_col, name_col)]
            fields.append(("Subjects", subjects))
        lines.append("- " + " | ".join(f"{h}: {v}" if h else v for h, v in fields if v))
    return "\n".join(lines)


def extract_tables(path: Path):
    """Readable text for every table, per page. Must run before pymupdf4llm is imported.
    A row continued from the previous page is merged into that page's table, so e.g.
    all of 'Sem 3' ends up in one place."""
    state, pages, last = {}, [], None
    for page in pymupdf.open(path):
        tables = []
        for t in page.find_tables().tables:
            table = table_groups(t.extract(), state)
            if not table:
                continue
            first = table["groups"][0] if table["groups"] else None
            if first and first["continued"] and last and last["groups"] and \
                    last["groups"][-1]["key"] == first["key"]:
                last["groups"][-1]["cells"] = merge_cells(last["groups"][-1]["cells"], first["cells"])
                table["groups"].pop(0)
            tables.append(table)
            last = table
        pages.append(tables)
    return [[render_table(t) for t in tables] for tables in pages]


def pdf_pages(path: Path, page_tables):
    """Markdown per page, with markdown tables replaced by readable row lines."""
    import pymupdf4llm
    for p in pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False):
        n = p["metadata"]["page_number"]
        tables = page_tables[n - 1]
        out, in_table, keep = [], False, False
        for line in p["text"].splitlines():
            if line.lstrip().startswith("|"):
                if not in_table:  # start of a markdown table block
                    in_table, keep = True, not tables
                    if tables:
                        out.append(tables.pop(0))
                if keep:  # no converted table left, keep the original markdown
                    out.append(line)
                continue
            in_table = False
            out.append(line)
        out += tables  # any tables not matched to a markdown block
        yield n, "\n".join(out)


def load_documents():
    """Yield (source_name, page, markdown_text) for every file in data/."""
    paths = sorted(DATA_DIR.rglob("*"))
    tables = {p: extract_tables(p) for p in paths if p.suffix.lower() == ".pdf"}
    for path in paths:
        if path.suffix.lower() == ".pdf":
            for page, text in pdf_pages(path, tables[path]):
                yield path.name, page, text
        elif path.suffix.lower() in (".md", ".txt"):
            yield path.name, 1, path.read_text(encoding="utf-8")


def split_sections(text: str, heading: str):
    """Split markdown into (heading, body) sections; heading carries over from previous page."""
    body = []
    for line in text.splitlines():
        if is_heading(line) and "".join(body).strip():
            yield heading, "\n".join(body)
            body = []
        if is_heading(line):
            heading = clean(line).lstrip("# ").strip()
        body.append(line)
    if "".join(body).strip():
        yield heading, "\n".join(body)


def chunk_text(text: str):
    """Greedy line-based chunking with overlap."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > CHUNK_SIZE and current.strip():
            chunks.append(current.strip())
            current = current[-OVERLAP:]
        current += line
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    headings = {}  # last heading seen per file, so it carries across pages
    for source, page, raw in load_documents():
        title = Path(source).stem.replace("_", " ").replace("-", " ")
        for heading, section in split_sections(raw, headings.get(source, "")):
            headings[source] = heading
            for i, chunk in enumerate(chunk_text(clean(section))):
                if len(chunk) < 40:
                    continue
                label = f"[Document: {title}" + (f" | Section: {heading}" if heading else "") + "]"
                ids.append(f"{source}-p{page}-{len(ids)}")
                docs.append(f"{label}\n{chunk}")
                metas.append({"source": source, "page": page, "section": heading[:200]})
        print(f"  read {source} page {page}")

    print(f"Embedding {len(docs)} chunks with {EMBED_MODEL} ...")
    for start in range(0, len(docs), 32):
        batch = docs[start:start + 32]
        # nomic-embed-text expects task prefixes
        embs = ollama.embed(model=EMBED_MODEL, input=[f"search_document: {d}" for d in batch])["embeddings"]
        col.add(ids=ids[start:start + 32], documents=batch, embeddings=embs,
                metadatas=metas[start:start + 32])
    print(f"Done. Indexed {col.count()} chunks into ./{DB_DIR}")


if __name__ == "__main__":
    main()
