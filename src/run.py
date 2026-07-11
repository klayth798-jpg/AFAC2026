"""主流程:加载题目 -> 逐题作答 -> 输出 answer.csv + summary.json。

用法:
    python src/run.py            # 全量
    python src/run.py --limit 5  # 只跑前 5 题(MVP 验证)
    python src/run.py --domains financial_reports insurance
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (OUTPUT_ROOT, QUESTIONS_ROOT, TOKEN_BUDGET, LLMClient,
                    TokenMeter)
from answer import answer_question

DOMAIN_FILES = {
    "insurance": "insurance_questions.json",
    "financial_reports": "financial_reports_questions.json",
    "financial_contracts": "financial_contracts_questions.json",
    "regulatory": "regulatory_questions.json",
    "research": "research_questions.json",
}


def load_questions(domains: list[str] | None) -> list[dict]:
    domains = domains or list(DOMAIN_FILES.keys())
    out = []
    for dom in domains:
        path = QUESTIONS_ROOT / DOMAIN_FILES[dom]
        out.extend(json.loads(path.read_text(encoding="utf-8")))
    return out


def token_score(total: int) -> float:
    return max(0.0, min(1.0, (TOKEN_BUDGET - total) / TOKEN_BUDGET))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题,0=全量")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑:跳过 answer.csv 中已成功作答(token>0)的题目")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    questions = load_questions(args.domains)
    if args.limit:
        questions = questions[: args.limit]

    rows_by_qid = {}  # qid -> row dict
    lock = threading.Lock()
    t0 = time.time()
    csv_path = OUTPUT_ROOT / "answer.csv"

    # 断点续跑:载入已有结果,token>0 视为成功,跳过重跑
    if args.resume and csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["qid"] == "summary":
                    continue
                tot = int(r.get("total_tokens") or 0)
                if tot > 0:
                    rows_by_qid[r["qid"]] = {
                        "qid": r["qid"], "answer": r["answer"],
                        "prompt_tokens": int(r.get("prompt_tokens") or 0),
                        "completion_tokens": int(r.get("completion_tokens") or 0),
                        "total_tokens": tot,
                    }
        pending = [q for q in questions if q["qid"] not in rows_by_qid]
        print(f"[RESUME] 已完成 {len(rows_by_qid)} 题,待跑 {len(pending)} 题",
              flush=True)
    else:
        pending = questions

    def write_csv():
        # 汇总所有已完成题目的 token,summary 行取全局合计
        g_prompt = sum(r["prompt_tokens"] for r in rows_by_qid.values())
        g_completion = sum(r["completion_tokens"] for r in rows_by_qid.values())
        g_total = sum(r["total_tokens"] for r in rows_by_qid.values())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["qid", "answer", "prompt_tokens",
                            "completion_tokens", "total_tokens"],
            )
            writer.writeheader()
            writer.writerow({
                "qid": "summary", "answer": "",
                "prompt_tokens": g_prompt,
                "completion_tokens": g_completion,
                "total_tokens": g_total,
            })
            # 按原题序输出
            for q in questions:
                if q["qid"] in rows_by_qid:
                    writer.writerow(rows_by_qid[q["qid"]])

    def worker(q):
        # 每题独立 meter+client,token 归属清晰,线程安全
        m = TokenMeter()
        client = LLMClient(m)
        try:
            ans = answer_question(client, q)
        except Exception as e:
            print(f"[ERROR] {q['qid']}: {e}", flush=True)
            ans = list(q["options"].keys())[0]
        return q, ans, m

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(worker, q): q for q in pending}
        for fut in as_completed(futures):
            q, ans, m = fut.result()
            with lock:
                done += 1
                rows_by_qid[q["qid"]] = {
                    "qid": q["qid"], "answer": ans,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                }
                g_total = sum(r["total_tokens"] for r in rows_by_qid.values())
                print(f"[{done}/{len(pending)}] {q['qid']} ({q['answer_format']}) "
                      f"-> {ans}  | +{m.total_tokens} tok  累计={g_total}", flush=True)
                write_csv()

    # 全局汇总
    g_prompt = sum(r["prompt_tokens"] for r in rows_by_qid.values())
    g_completion = sum(r["completion_tokens"] for r in rows_by_qid.values())
    total_tokens = sum(r["total_tokens"] for r in rows_by_qid.values())

    # 输出 summary.json
    summary = {
        "total_tokens": total_tokens,
        "prompt_tokens": g_prompt,
        "completion_tokens": g_completion,
        "token_budget": TOKEN_BUDGET,
        "token_score": round(token_score(total_tokens), 4),
        "num_questions": len(questions),
        "num_answered": len(rows_by_qid),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n==== 完成 ====")
    print(f"题目数: {len(questions)}  已答: {len(rows_by_qid)}")
    print(f"total_tokens: {total_tokens}  (预算 {TOKEN_BUDGET})")
    print(f"token_score: {summary['token_score']}")
    print(f"输出: {csv_path}")


if __name__ == "__main__":
    main()
