# AFAC2026 · 金融长文本 Agent 的动态记忆压缩与高效问答

赛题链接：https://tianchi.aliyun.com/competition/entrance/532486/information

## 赛题简介

AFAC2026 金融智能创新大赛「挑战组」赛题四。主办方提供海量、结构复杂的金融长文档（年报、保险条款、募集说明书、监管法规、研究报告等），选手需构建一个 Agent，在**控制 Token 成本**的前提下，基于这些长上下文对给定问题做出**精准问答**。

核心难点：

- **文档结构复杂**：大量交叉引用、多级标题、密集表格、附录与批注，一个否定词或一个单元格错位就会导致答案错误。
- **上下文超长**：单份文档动辄上百页，无法整体塞入模型窗口，需要检索 / 分块 / 动态记忆压缩等工程手段。
- **成本约束**：在保证准确率的同时优化 Token 消耗，考验召回与压缩策略的取舍。

## 数据集结构

数据集位于同级目录 `../public_dataset_upload/`：

```
public_dataset_upload/
├── questions/
│   └── group_a/                       # A 榜题目（共 100 题，每个领域 20 题）
│       ├── insurance_questions.json
│       ├── financial_reports_questions.json
│       ├── financial_contracts_questions.json
│       ├── regulatory_questions.json
│       └── research_questions.json
└── raw/                               # 题目引用的原始文档
    ├── insurance/                     # 保险产品条款 PDF（1.pdf ~ 16.pdf）
    ├── financial_reports/             # 上市公司年报 PDF（比亚迪/宁德/中建/招行/美的等）
    ├── financial_contracts/           # 债券募集说明书等合同 PDF（text01 ~ text14）
    ├── research/                      # 行业研究报告 PDF（pack2_text01 ~ pack2_text20）
    └── regulatory/                    # 监管法规
        ├── txt/                       # 法规正文（6 篇 .txt）
        └── attachments/               # 法规附件 PDF（csrc_xxxx_attN.pdf，共 130 个）
```

### 五个业务领域

| 领域 | 目录 | 文档类型 | 题目文件 |
|------|------|----------|----------|
| 保险 insurance | `raw/insurance` | 养老 / 寿险 / 医疗险产品条款 | `insurance_questions.json` |
| 财报 financial_reports | `raw/financial_reports` | 上市公司年度报告 | `financial_reports_questions.json` |
| 金融合同 financial_contracts | `raw/financial_contracts` | 债券募集说明书等 | `financial_contracts_questions.json` |
| 监管 regulatory | `raw/regulatory` | 央行 / 金监总局 / 证监会法规 | `regulatory_questions.json` |
| 研究报告 research | `raw/research` | 行业研究报告 | `research_questions.json` |

## 题目格式

每道题为一个 JSON 对象，示例：

```json
{
  "qid": "fin_a_001",
  "domain": "financial_reports",
  "split": "A",
  "question": "根据比亚迪连续两年的年度报告，下列关于公司经营业绩变化的描述中，哪些是准确的？",
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "answer_format": "multi",
  "type": "财务指标对比分析",
  "doc_ids": ["annual_byd_2024_report", "annual_byd_2025_report"]
}
```

字段说明：

- `qid`：题目唯一 ID。
- `domain`：所属领域（对应上表）。
- `question` / `options`：题干与选项。
- `answer_format`：答案类型，取值见下表。
- `type`：题目细分类型（如计算题、推理判断、财务指标对比等）。
- `doc_ids`：该题引用的原始文档标识，对应 `raw/<domain>/` 下的文件名（不含扩展名）。

### 答案类型分布（A 榜 100 题）

| answer_format | 含义 | 数量 |
|---------------|------|------|
| `multi` | 多选题（多个正确选项） | 65 |
| `tf` | 判断题（正确 / 错误） | 20 |
| `mcq` | 单选题 | 15 |

> 注：公开数据集中的题目**不含标准答案**，答案由官方评测时持有。

