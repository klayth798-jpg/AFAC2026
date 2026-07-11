"""配置、LLM 客户端封装、全局 Token 记账。

评测规则要求统计全流程所有 API 调用的 token 消耗(检索摘要、上下文压缩、
证据判断、答案生成、自检),写入 summary.total_tokens。因此所有对模型的
调用都必须经过本模块的 LLMClient,以保证 token 被累加。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "public_dataset_upload"
RAW_ROOT = DATA_ROOT / "raw"
QUESTIONS_ROOT = DATA_ROOT / "questions" / "group_a"
CACHE_ROOT = PROJECT_ROOT / "cache"          # 解析后的分块缓存
OUTPUT_ROOT = PROJECT_ROOT / "output"        # answer.csv / summary.json

# ---- 领域 -> 原始文档目录 ----
DOMAIN_DIRS = {
    "insurance": RAW_ROOT / "insurance",
    "financial_reports": RAW_ROOT / "financial_reports",
    "financial_contracts": RAW_ROOT / "financial_contracts",
    "research": RAW_ROOT / "research",
    "regulatory": RAW_ROOT / "regulatory",   # 含 txt/ html/ attachments/
}

# ---- 评测预算 ----
TOKEN_BUDGET = 5_000_000

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.6-plus")
_BASE_URL = os.environ.get("OPENAI_BASE_URL")
# 兼容多平台 key 变量名
_API_KEY = (os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("ARK_API_KEY"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")
RERANK_URL = os.environ.get("RERANK_URL", "")

# 端点实测可持续速率约 1 次 / 15-20s,用全局节流器串行限速,避免 429。
import threading as _threading
import time as _time

_RATE_LOCK = _threading.Lock()
_last_call_ts = [0.0]
MIN_CALL_INTERVAL = float(os.environ.get("MIN_CALL_INTERVAL", "16"))


def _throttle() -> None:
    with _RATE_LOCK:
        now = _time.time()
        wait = MIN_CALL_INTERVAL - (now - _last_call_ts[0])
        if wait > 0:
            _time.sleep(wait)
        _last_call_ts[0] = _time.time()


class TokenMeter:
    """全局 token 计数器。每次 API 调用后累加 usage.total_tokens。"""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.calls += 1

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_calls": self.calls,
        }


class LLMClient:
    """字节方舟 doubao-seed-2.1-pro 的 OpenAI 兼容封装,内置 token 记账。

    - chat: seed2.1-pro,默认开启思考模式(thinking.enabled)以保准确率。
    - embed/rerank: 方舟未提供该端点,retrieve 退化为纯 BM25;这两个方法
      保留为安全 no-op,便于后续换回带 rerank 的平台。
    """

    def __init__(self, meter: TokenMeter) -> None:
        if not _API_KEY or not _BASE_URL:
            raise RuntimeError("缺少 ARK_API_KEY / OPENAI_BASE_URL,请检查 .env")
        # 每次请求 120s 超时 + 最多重试 2 次(思考模式响应较慢)
        self.client = OpenAI(
            api_key=_API_KEY, base_url=_BASE_URL, timeout=120.0, max_retries=2
        )
        self.meter = meter

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> str:
        import time as _t
        for attempt in range(6):
            _throttle()
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": thinking},
                )
            except Exception as e:
                # 429 限流 / 网络抖动:退避后重试(逐次加长)
                etype = type(e).__name__
                if ("429" in str(e) or "RateLimit" in etype
                        or "Connection" in etype or "Timeout" in etype):
                    _t.sleep(15 * (attempt + 1))
                    continue
                print(f"[LLM ERROR] {etype}: {e}")
                return ""
            self.meter.add(resp.usage)
            content = resp.choices[0].message.content or ""
            return content.strip()
        print("[LLM ERROR] 429 重试耗尽")
        return ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not EMBEDDING_MODEL:
            return []  # 方舟无 embedding 端点,退化为纯 BM25
        try:
            resp = self.client.embeddings.create(
                model=EMBEDDING_MODEL, input=texts
            )
        except Exception as e:
            print(f"[EMBED ERROR] {type(e).__name__}: {e}")
            return []
        self.meter.add(getattr(resp, "usage", None))
        return [d.embedding for d in resp.data]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        """qwen3-rerank 精排,返回按相关度降序的 document 索引(前 top_n 个)。
        无 rerank 配置或失败时返回空,由调用方退化为本地重排。"""
        if not RERANK_MODEL or not RERANK_URL:
            return []
        import httpx
        try:
            r = httpx.post(
                RERANK_URL,
                headers={"Authorization": f"Bearer {_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": RERANK_MODEL, "query": query,
                      "documents": documents, "top_n": top_n},
                timeout=60.0,
            )
            data = r.json()
            results = data.get("results", [])
            usage = data.get("usage")
            if usage:
                t = usage.get("total_tokens", 0) or 0
                self.meter.total_tokens += t
                self.meter.prompt_tokens += t
                self.meter.calls += 1
            ranked = sorted(results, key=lambda x: x.get("relevance_score", 0),
                            reverse=True)
            return [x["index"] for x in ranked[:top_n]]
        except Exception as e:
            print(f"[RERANK ERROR] {type(e).__name__}: {e}")
            return []
