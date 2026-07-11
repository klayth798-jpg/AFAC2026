"""检索:doc_ids 锁定范围 + BM25 多 query 粗筛 + rerank 精排。

设计权衡(受 500 万 token 预算约束):
- 放弃对整份 300 页财报做全量 embedding(实测每题 ~40 万 token,爆预算)。
- 保留研究证明性价比最高的 cross-encoder rerank(PwC:MRR +59%)。
- BM25(多 query)先粗筛出 pool 个候选,再用 qwen3-reranker 精排到 top_k。
- rerank 候选正文截断到 600 字,控制 rerank 的 token 开销。
"""
from __future__ import annotations

import re

import jieba
from rank_bm25 import BM25Okapi

from parse import build_doc

_STOP = set("的 了 和 与 及 或 在 是 为 对 以 等 中 上 下 之 其 该 本 第 条 款 项 者 "
            "。 ， 、 ； ： （ ） 《 》 “ ” ？ ! ? . , ; :".split())


def _tokenize(text: str) -> list[str]:
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and t not in _STOP]


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion:融合多个 query 的 BM25 排序。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return scores


class DocRetriever:
    """针对单题引用的若干文档构建 BM25 索引,配合 rerank 精排。"""

    def __init__(self, domain: str, doc_ids: list[str], llm=None) -> None:
        self.llm = llm
        self.doc_ids = doc_ids
        self.chunks: list[dict] = []
        for did in doc_ids:
            self.chunks.extend(build_doc(domain, did))
        self._corpus_tokens = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens) if self.chunks else None

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)

    def _local_rerank(self, cand: list[int], options: list[str],
                      domain: str) -> list[int]:
        """本地零成本重排:选项词/数字重叠 + 表格优先(财报/保险)。

        比纯 BM25 更贴题:优先带有选项里出现的数字/关键词的 chunk;
        财报/保险数值题优先表格块。避免额外 LLM 调用(受限流约束)。
        """
        # 选项里的数字与较长词元
        opt_text = " ".join(options)
        opt_nums = set(re.findall(r"\d[\d,\.%]*", opt_text))
        opt_terms = set(t for t in _tokenize(opt_text) if len(t) >= 2)
        table_boost = domain in ("financial_reports", "insurance")

        def score(i: int) -> float:
            c = self.chunks[i]
            txt = c["text"]
            s = 0.0
            # 数字重叠(数值题关键)
            for n in opt_nums:
                if len(n) >= 2 and n in txt:
                    s += 3.0
            # 词元重叠
            toks = set(_tokenize(txt))
            s += len(opt_terms & toks) * 0.5
            # 表格优先
            if table_boost and c.get("is_table"):
                s += 2.0
            return s

        # 稳定排序:先按本地分,同分保持 BM25 原序
        return sorted(cand, key=lambda i: score(i), reverse=True)

    def retrieve(self, query: str, options: list[str],
                 pool: int = 20, top_k: int = 8, domain: str = "") -> list[dict]:
        """按 doc 配额均衡召回,避免多文档题里某个文档被挤占(漏选主因)。

        对每个引用文档单独跑 BM25 多 query + 本地重排,各取 top_k/n_docs;
        再对合并候选用 qwen3-rerank 统一精排(不足则保留均衡结果)。
        """
        if not self.bm25:
            return []
        queries = [query] + [f"{query} {o}" for o in options]

        # 1) 按 doc 均衡:每个文档独立取候选
        n_docs = max(1, len(self.doc_ids))
        per_doc = max(4, top_k // n_docs + 2)  # 每文档配额,略放宽
        balanced: list[int] = []
        for did in self.doc_ids:
            doc_idx = [i for i, c in enumerate(self.chunks) if c["doc_id"] == did]
            if not doc_idx:
                continue
            idx_set = set(doc_idx)
            rank_lists = []
            for q in queries:
                ranked = [i for i in self._bm25_rank(q) if i in idx_set][:pool]
                rank_lists.append(ranked)
            fused = _rrf(rank_lists)
            cand = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]
            cand = self._local_rerank(cand, options, domain)[:per_doc]
            balanced.extend(cand)

        # 2) 统一 rerank 精排(可选);保证不丢均衡覆盖
        if self.llm is not None and len(balanced) > top_k:
            docs = [self.chunks[i]["text"][:600] for i in balanced]
            order = self.llm.rerank(query, docs, top_n=len(balanced))
            if order:
                balanced = [balanced[o] for o in order]

        return [self.chunks[i] for i in balanced[: max(top_k, per_doc * n_docs)]]


def build_context(chunks: list[dict], max_chars: int = 10000) -> str:
    """把召回 chunk 拼成上下文,带来源标注,控制总长度。"""
    parts = []
    used = 0
    for c in chunks:
        header = f"【来源: {c['doc_id']} #chunk{c['chunk_id']}】\n"
        body = c["text"]
        if used + len(header) + len(body) > max_chars:
            body = body[: max(0, max_chars - used - len(header))]
        parts.append(header + body)
        used += len(header) + len(body)
        if used >= max_chars:
            break
    return "\n\n".join(parts)
