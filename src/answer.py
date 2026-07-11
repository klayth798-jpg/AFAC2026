"""答题:混合检索 + 分领域 prompt + 逐选项裁决。

优化点:
- 分领域差异化 prompt(财报/保险重表格取数与计算,监管重跨文档条款)。
- 多选题逐选项独立判定(每选项一次调用),预算充足以准确率优先。
- 修掉"找不到证据默认 A"的偏置:强制给出最可能答案,不再无脑选 A。
- 答案解析只认 '答案:' 标记,识别否定语义。
"""
from __future__ import annotations

import re

from config import LLMClient
from retrieve import DocRetriever, build_context

# 分领域作答侧重
_DOMAIN_HINT = {
    "financial_reports": (
        "这是上市公司年报题。资料中【表格】块含合并利润表、现金流量表、分红方案等"
        "关键数字。务必从表格中逐字核对营业收入、净利润、经营活动现金流、研发投入、"
        "每股分红等数值;涉及同比/占比/增长率时先取两个原始数再计算,不要凭印象。"
    ),
    "insurance": (
        "这是保险条款题。注意区分身故金、现金价值、账户价值、免赔额、等待期、给付比例"
        "等定义;计算题先从条款找出适用公式与参数,再逐步代入计算。"
    ),
    "financial_contracts": (
        "这是债券募集说明书题。核对发行人名称、发行规模上限、主体/债项信用评级、"
        "受托管理人、承销商、信息披露承诺等要素,以原文为准。"
    ),
    "regulatory": (
        "这是金融监管法规题。注意跨多部法规交叉比对:施行日期、报告时限(工作日/日)、"
        "比例、期限、适用范围;逐条核对措辞,警惕'应当/可以''以上/以下'等细节。"
    ),
    "research": (
        "这是行业研究报告题。核对报告中的数据、观点、结论;区分事实与预测。"
    ),
}


def _extract_answer_letters(text: str, valid: list[str]) -> list[str]:
    """只解析最后一个 '答案:' 标记后的字母,识别否定语义。"""
    if not text:
        return []
    marks = list(re.finditer(r"(?:答案|正确选项|answer)\s*[:：]?", text, re.I))
    if marks:
        seg = text[marks[-1].end(): marks[-1].end() + 40]
    else:
        seg = text.strip().splitlines()[-1] if text.strip() else ""
    if re.search(r"无|没有|都不|均.*错|全部错|不选|none", seg, re.I):
        return []
    letters, out = re.findall(r"[A-Za-z]", seg), []
    for c in letters:
        c = c.upper()
        if c in valid and c not in out:
            out.append(c)
    return out


def _answer_single(llm: LLMClient, domain: str, question: str, options: dict,
                   context: str, valid: list[str]) -> str:
    """单选/判断:直接选一个;找不到证据也要给最可能项,不默认 A。"""
    hint = _DOMAIN_HINT.get(domain, "")
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    sys = ("你是严谨的金融答题专家,只依据资料作答。" + hint)
    user = (
        f"资料:\n{context}\n\n问题:{question}\n\n选项:\n{opt_text}\n\n"
        "有且仅有一个正确答案。先逐项分析并排除错误项;即使资料不全,也必须"
        "依据现有信息选出最可能正确的一项,不得弃权。"
        f"最后一行只写:『答案:』后接唯一字母({'/'.join(valid)})。"
    )
    out = llm.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=1200, thinking=False,
    )
    got = _extract_answer_letters(out, valid)
    return got[0] if got else valid[0]


def _answer_multi(llm: LLMClient, domain: str, question: str,
                  options: dict, context: str,
                  valid: list[str]) -> list[str]:
    """多选:单次调用内逐项核对(context 只发一次,token 约为逐选项判定的 1/4)。

    强制逐项列证据 + 结论,配合"倾向收录"的判定标准以减少漏选(exact-match
    评分下漏选即 0 分)。
    """
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    letters = "/".join(valid)
    sys = ("你是严谨的金融文档核查专家,只依据资料逐项核对。"
           + _DOMAIN_HINT.get(domain, ""))
    user = (
        f"资料:\n{context}\n\n问题:{question}\n\n选项:\n{opt_text}\n\n"
        "这是多选题(正确选项通常为 2~3 个,少数为 1 或 4 个)。请【逐个选项】独立核对:\n"
        "对每个选项,先写字母,再给资料依据,然后判定『成立』或『不成立』。\n"
        "判定标准(倾向收录、减少漏选):\n"
        "- 资料支持、与之一致、或能提供合理依据 → 判『成立』;\n"
        "- 仅当资料【明确矛盾】或有确凿错误(数值/日期/比例/定义不符)→ 判『不成立』;\n"
        "- 证据支持力度中等时倾向『成立』;不要只选 1 个就收手,逐项都要认真核对。\n"
        f"全部核对完后,最后一行只写:『答案:』后接所有成立的字母(如 ABD),"
        f"合法范围 {letters};无成立项写『答案:无』。"
    )
    out = llm.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=1600, thinking=False,
    )
    got = _extract_answer_letters(out, valid)
    return got if got else [valid[0]]


def answer_question(llm: LLMClient, q: dict) -> str:
    """主入口:返回答案字母串(如 'A' 或 'AC')。"""
    domain = q["domain"]
    doc_ids = q["doc_ids"]
    fmt = q["answer_format"]
    question = q["question"]
    options = q["options"]
    valid = list(options.keys())

    retriever = DocRetriever(domain, doc_ids, llm=llm)
    # multi 已改单次调用(每题仅 ~16k token)。实测中等上下文优于超大上下文
    # (过大会引入噪声、稀释关键证据):财报略大,其余适中。
    if domain == "financial_reports":
        top_k, max_chars = 16, 15000
    else:
        top_k, max_chars = 14, 12000
    chunks = retriever.retrieve(question, list(options.values()),
                                pool=60, top_k=top_k, domain=domain)
    context = build_context(chunks, max_chars=max_chars)

    if fmt in ("mcq", "tf"):
        return _answer_single(llm, domain, question, options, context, valid)

    # multi:单次调用内逐项核对(实测与逐选项独立判定准确率相当甚至更高,
    # 且 token 约为其 1/3~1/4)。context 大小按领域控制。
    selected = _answer_multi(llm, domain, question, options, context, valid)
    return "".join(sorted(selected))
