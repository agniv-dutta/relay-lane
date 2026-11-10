"""Ingest arXiv papers: download PDFs, extract text per page, chunk, save JSONL.

Usage:
    python -m relay.ingest
    python -m relay.ingest --sizes 200 400 800
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import requests
import tiktoken

PAPERS = [
    ("1706.03762", "v7"),
    ("2410.00037", "v1"),
    ("2507.07955", "v2"),
]

ARXIV_PDF_URL = "https://arxiv.org/pdf/{pid}{ver}.pdf"

ENCODING = tiktoken.get_encoding("cl100k_base")


def download_pdfs(raw_dir: Path, force: bool = False) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for pid, ver in PAPERS:
        name = f"{pid}{ver}.pdf"
        dest = raw_dir / name
        if dest.exists() and not force:
            print(f"[download] cached: {dest}")
        else:
            url = ARXIV_PDF_URL.format(pid=pid, ver=ver)
            print(f"[download] fetching {url}")
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[download] saved {dest} ({dest.stat().st_size} bytes)")
        paths[(pid, ver)] = dest
    return paths


def _body_font_size(lines: list[dict]) -> float:
    sizes = [round(l["size"], 1) for l in lines if l["text"].strip()]
    if not sizes:
        return 11.0
    try:
        return statistics.mode(sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


_HEADING_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+|[A-Z][A-Za-z].{0,60})$"
)


def _is_heading(line: dict, body_size: float) -> bool:
    text = line["text"].strip()
    if not text or len(text) > 140:
        return False
    size = line["size"]
    bold = line["bold"]
    low = text.lower().strip().rstrip(".")
    known = {
        "abstract", "introduction", "conclusion", "conclusions", "references",
        "bibliography", "appendix", "acknowledgements", "acknowledgments",
        "related work", "background", "method", "methods", "methodology",
        "experiments", "results", "discussion", "limitations",
    }
    if low in known:
        return True
    # Numbered section headings like "1. Introduction" or "2.1 Model"
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
        return True
    # Large font relative to body (titles, section headers)
    if size >= body_size + 2.0:
        return True
    # Bold, slightly larger, and short
    if bold and size >= body_size + 0.5 and len(text) < 80 and re.match(r"^[A-Z0-9]", text):
        return True
    return False


def extract_page(page, page_num: int) -> tuple[str, list[str], list[str]]:
    """Return (text, sections_on_page, warnings)."""
    warnings = []
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            sizes = [s.get("size", 0) for s in spans] or [0]
            bold = any(("Bold" in s.get("font", "")) or (s.get("flags", 0) & 2) for s in spans)
            lines.append({"text": text, "size": max(sizes), "bold": bold, "bbox": line.get("bbox")})

    body_size = _body_font_size(lines)

    # Two-column heuristic
    blocks = page.get_text("blocks")
    if blocks:
        xs = [b[0] for b in blocks]
        width = page.rect.width
        left = sum(1 for x in xs if x < width * 0.35)
        right = sum(1 for x in xs if x > width * 0.5)
        middle_span = sum(1 for b in blocks if b[0] < width * 0.45 and b[2] > width * 0.55)
        if len(blocks) >= 8 and left >= 3 and right >= 3 and middle_span <= 2:
            warnings.append("possible two-column layout garbling")

    current_section = "Body"
    sections = []
    out_lines = []
    math_count = 0
    for line in lines:
        text = line["text"]
        if _is_heading(line, body_size):
            current_section = text
            if current_section not in sections:
                sections.append(current_section)
            out_lines.append(f"## {text}")
            continue
        # math/equation heuristic
        if re.search(r"[∫∑∏√≈≠≤≥±×÷∂∇∈∉⊂⊃∪∩]|[\u03b1-\u03c9\u0391-\u03a9]", text):
            math_count += 1
        if current_section not in sections:
            sections.append(current_section)
        out_lines.append(text)

    if math_count >= 4:
        warnings.append(f"possible equations on page ({math_count} math-heavy lines)")

    text = "\n".join(out_lines)
    return text, sections, warnings


def detect_tables_pdfplumber(pdf_path: Path, page_num: int) -> int:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables() or []
            return len(tables)
    except Exception:
        return 0


def extract_paper(pid: str, ver: str, pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    pages_out = []
    all_warnings = []
    for i, page in enumerate(doc, start=1):
        text, sections, warnings = extract_page(page, i)
        # table detection
        n_tables = detect_tables_pdfplumber(pdf_path, i)
        if n_tables:
            warnings = list(warnings) + [f"possible tables on page ({n_tables} detected)"]
        # references flag
        if any("reference" in s.lower() or "bibliograph" in s.lower() for s in sections):
            warnings = list(warnings) + ["references/bibliography section present"]
        if warnings:
            all_warnings.append({"page": i, "warnings": warnings})
        pages_out.append({
            "paper_id": pid,
            "version": ver,
            "page": i,
            "sections": sections,
            "text": text,
            "warnings": warnings,
        })
    doc.close()
    return {"paper_id": pid, "version": ver, "pages": pages_out, "warnings": all_warnings}


def atomize(extracted: dict) -> list[dict]:
    atoms = []
    pid = extracted["paper_id"]
    ver = extracted["version"]
    for page in extracted["pages"]:
        page_num = page["page"]
        current_section = "Body"
        for raw_line in page["text"].split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("## "):
                current_section = line[3:].strip()
                # include heading as atom
                atoms.append({"text": line, "paper_id": pid, "version": ver,
                             "page": page_num, "section": current_section})
                continue
            # update section? already tracked via headings
            atoms.append({"text": line, "paper_id": pid, "version": ver,
                         "page": page_num, "section": current_section})
    return atoms


def chunk_atoms(atoms: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    # group by paper to avoid cross-paper chunks
    by_paper = defaultdict(list)
    for a in atoms:
        by_paper[(a["paper_id"], a["version"])].append(a)

    all_chunks = []
    for (pid, ver), paper_atoms in by_paper.items():
        # token counts
        tok_counts = [len(ENCODING.encode(a["text"], disallowed_special=())) for a in paper_atoms]
        counter = 0
        start = 0
        n = len(paper_atoms)
        while start < n:
            end = start
            toks = 0
            while end < n and toks + tok_counts[end] <= chunk_size:
                toks += tok_counts[end]
                end += 1
            if end == start:
                end = start + 1
                toks = tok_counts[start]
            text = "\n".join(a["text"] for a in paper_atoms[start:end])
            meta = paper_atoms[start]
            counter += 1
            all_chunks.append({
                "paper_id": pid,
                "version": ver,
                "page": meta["page"],
                "section": meta["section"],
                "chunk_id": f"{pid}_{ver}_chunk_{counter:04d}",
                "text": text,
                "token_count": toks,
            })
            if end >= n:
                break
            # overlap
            overlap_toks = 0
            new_start = end
            while new_start > start and overlap_toks < overlap:
                new_start -= 1
                overlap_toks += tok_counts[new_start]
            if new_start <= start:
                new_start = start + 1
            start = new_start
    return all_chunks


import sys

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 400, 800])
    parser.add_argument("--overlap", type=int, default=None, help="overlap in tokens (overrides --overlap-ratio)")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    extracted_dir = data_dir / "extracted"
    chunks_dir = data_dir / "chunks"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    paths = download_pdfs(raw_dir, force=args.force_download)

    # Extract
    extracted_list = []
    for pid, ver in PAPERS:
        pdf_path = paths[(pid, ver)]
        print(f"\n[extract] {pid}{ver}")
        ext = extract_paper(pid, ver, pdf_path)
        extracted_list.append(ext)
        out_path = extracted_dir / f"{pid}{ver}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ext, f, ensure_ascii=False, indent=2)
        # warnings
        if ext["warnings"]:
            print(f"  warnings: {len(ext['warnings'])} page(s)")
            for w in ext["warnings"]:
                print(f"    p{w['page']}: {w['warnings']}")
        else:
            print("  no warnings")

    # Atomize
    all_atoms = []
    for ext in extracted_list:
        all_atoms.extend(atomize(ext))

    # Chunk for each size
    for size in args.sizes:
        if args.overlap is not None:
            overlap = args.overlap
        else:
            overlap = max(1, int(size * args.overlap_ratio))
        chunks = chunk_atoms(all_atoms, size, overlap)
        out_file = chunks_dir / f"chunks_{size}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\n[chunk] size={size} overlap={overlap} -> {len(chunks)} chunks -> {out_file}")

        # samples
        print(f"  samples (first {args.sample}):")
        for c in chunks[: args.sample]:
            preview = c["text"].replace("\n", " ")[:200]
            print(f"    {c['chunk_id']} | p{c['page']} | {c['section'][:40]} | {c['token_count']}tok | {preview}...")

    # Overall warning summary
    print("\n=== EXTRACTION WARNING SUMMARY ===")
    total_warn_pages = 0
    for ext in extracted_list:
        if ext["warnings"]:
            total_warn_pages += len(ext["warnings"])
            print(f"{ext['paper_id']}{ext['version']}: {len(ext['warnings'])} page(s) with warnings")
            for w in ext["warnings"]:
                print(f"   p{w['page']}: {', '.join(w['warnings'])}")
    print(f"Total pages with warnings: {total_warn_pages}")
    print("\nNotes:")
    print("- Two-column detection is heuristic; inspect samples for garbled reading order.")
    print("- Table/equation extraction is lossy; verify chunks containing them.")
    print("- References sections are included but may add noise; filter by section if needed.")


if __name__ == "__main__":
    main()
