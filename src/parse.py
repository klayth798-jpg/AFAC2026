"""文档解析与建库。

职责:
1. doc_id -> 文件路径解析(处理各领域命名、大小写扩展名差异)。
2. PDF(PyMuPDF)/ TXT / HTML(bs4)抽取文本 + 表格结构化。
3. 章节感知切块,注入元数据(doc_id, domain, chunk 序号, is_table)。
4. 缓存到 cache/<doc_id>.json,避免重复解析(不计入答题 token)。

关键:财报/保险的数值题依赖表格,PyMuPDF find_tables 把表格抽成
Markdown 保留行列关系,作为独立 chunk,避免拍平成乱序文本。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from config import CACHE_ROOT, DOMAIN_DIRS

CHUNK_SIZE = 1000          # 目标 chunk 字符数(中文,约等价 ~1800 英文字符)
CHUNK_OVERLAP = 150


def resolve_doc_path(domain: str, doc_id: str) -> Path | None:
    """把 doc_id 映射到实际文件路径,兼容大小写扩展名与子目录。"""
    base = DOMAIN_DIRS[domain]

    if domain == "regulatory":
        # 法规正文在 txt/,证监会文件在 html/ 或 attachments/
        candidates = [
            base / "txt" / f"{doc_id}.txt",
            base / "html" / f"{doc_id}.html",
            base / "attachments" / f"{doc_id}.pdf",
        ]
    else:
        candidates = [base / f"{doc_id}.pdf", base / f"{doc_id}.PDF"]

    for c in candidates:
        if c.exists():
            return c

    # 兜底:大小写不敏感地在目录树里找
    stem = doc_id.lower()
    for p in base.rglob("*"):
        if p.is_file() and p.stem.lower() == stem:
            return p
    return None


def _table_to_markdown(rows: list[list]) -> str:
    """把表格行列转成紧凑 Markdown,清理空列。"""
    cleaned = []
    for r in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in r]
        # 去掉整行空
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return ""
    lines = [" | ".join(c for c in row if c is not None) for row in cleaned]
    return "\n".join(lines)


def _extract_pdf(path: Path) -> tuple[str, list[str]]:
    """返回 (正文文本, 表格Markdown列表)。"""
    doc = fitz.open(path)
    text_parts = []
    tables_md = []
    for page in doc:
        text_parts.append(page.get_text("text"))
        try:
            tabs = page.find_tables()
            for t in tabs.tables:
                md = _table_to_markdown(t.extract())
                if md and len(md) > 20:
                    tables_md.append(md)
        except Exception:
            pass
    doc.close()
    return "\n".join(text_parts), tables_md


def _extract_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def _extract_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, doc_id: str, domain: str) -> list[dict]:
    """按段落聚合到 ~CHUNK_SIZE 字符,带重叠。"""
    text = _clean(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= CHUNK_SIZE:
            buf += para + "\n"
        else:
            if buf:
                chunks.append(buf.strip())
            tail = buf[-CHUNK_OVERLAP:] if buf else ""
            buf = tail + para + "\n"
    if buf.strip():
        chunks.append(buf.strip())

    return [
        {"doc_id": doc_id, "domain": domain, "chunk_id": i,
         "text": c, "is_table": False}
        for i, c in enumerate(chunks)
    ]


def _chunk_tables(tables_md: list[str], doc_id: str, domain: str,
                  start_id: int) -> list[dict]:
    """每个表格作为独立 chunk;超长表格按行切分。"""
    out = []
    cid = start_id
    for md in tables_md:
        if len(md) <= CHUNK_SIZE * 1.5:
            pieces = [md]
        else:  # 超长表格按行切
            lines = md.split("\n")
            pieces, buf = [], ""
            for ln in lines:
                if len(buf) + len(ln) + 1 > CHUNK_SIZE:
                    pieces.append(buf.strip())
                    buf = ""
                buf += ln + "\n"
            if buf.strip():
                pieces.append(buf.strip())
        for p in pieces:
            out.append({"doc_id": doc_id, "domain": domain, "chunk_id": cid,
                        "text": "[表格]\n" + p, "is_table": True})
            cid += 1
    return out


def build_doc(domain: str, doc_id: str, use_cache: bool = True) -> list[dict]:
    """解析单个文档为分块列表(正文 + 表格),带缓存。"""
    CACHE_ROOT.mkdir(exist_ok=True)
    cache_file = CACHE_ROOT / f"{domain}__{doc_id}.json"

    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    path = resolve_doc_path(domain, doc_id)
    if path is None:
        print(f"[WARN] 未找到文档: domain={domain} doc_id={doc_id}")
        return []

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw, tables_md = _extract_pdf(path)
        chunks = chunk_text(raw, doc_id, domain)
        chunks += _chunk_tables(tables_md, doc_id, domain, len(chunks))
    elif suffix in (".html", ".htm"):
        chunks = chunk_text(_extract_html(path), doc_id, domain)
    else:
        chunks = chunk_text(_extract_text_file(path), doc_id, domain)

    cache_file.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    return chunks


if __name__ == "__main__":
    for dom, did in [("financial_reports", "annual_byd_2024_report"),
                     ("regulatory", "strict_v3_017_中华人民共和国反洗钱法")]:
        cs = build_doc(dom, did, use_cache=False)
        n_tab = sum(1 for c in cs if c.get("is_table"))
        total = sum(len(c["text"]) for c in cs)
        print(f"{dom}/{did}: {len(cs)} chunks ({n_tab} 表格), {total} chars")
