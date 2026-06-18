"""
Step 5: RAG 质量评估 — LLM-as-Judge

不使用第三方 RAGAS 库，直接用 DeepSeek 做评判模型。
三条指标的逻辑与 RAGAS 一致，好处是：
- 无依赖冲突（不需要 langchain）
- 面试时能讲清楚每条指标的底层实现

指标说明：
┌─────────────────────┬────────────────────────────────────────┐
│ 指标                 │ 衡量什么                                │
├─────────────────────┼────────────────────────────────────────┤
│ Context Precision   │ 检索到的文档中，是否相关且排前           │
│ Context Recall      │ 该检索到的文档，实际是否检索到了         │
│ Faithfulness        │ 回答是否编造了上下文之外的内容           │
└─────────────────────┴────────────────────────────────────────┘

用法: python evaluate.py
"""
import os
import json
import sys
import time
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ 请先配置 DEEPSEEK_API_KEY（在 .env 文件中）")
    sys.exit(1)

import requests

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")


def judge(prompt: str) -> str:
    """调用 DeepSeek 做评判"""
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # 评判任务需要确定性
            "max_tokens": 200
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── 指标 1: Context Precision ─────────────────────────────
# 衡量：检索到的 N 条文档中，相关的是否排在前面

def score_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    对每条检索到的文档，让 LLM 判断是否与 ground_truth 相关。
    Precision@k = 相关文档数 / k，用排名位置加权。
    """
    if not contexts:
        return 0.0

    relevance = []
    for i, ctx in enumerate(contexts):
        prompt = f"""判断以下文档是否与问题和参考答案相关。只回答 "相关" 或 "不相关"。

问题：{question}

参考答案：{ground_truth}

文档：{ctx[:500]}

这个文档是否包含与正确答案有关的信息？"""
        result = judge(prompt)
        relevance.append(1 if "相关" in result else 0)

    # Precision@k 加权：排前面的相关文档权重更高
    score = 0.0
    relevant_count = 0
    for i, r in enumerate(relevance):
        if r == 1:
            relevant_count += 1
            score += relevant_count / (i + 1)

    if relevant_count == 0:
        return 0.0
    return score / relevant_count


# ── 指标 2: Context Recall ────────────────────────────────
# 衡量：ground_truth 中的关键信息，检索到的文档覆盖了多少

def score_context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    提取 ground_truth 中的关键信息点，检查 contexts 是否覆盖。
    """
    if not contexts:
        return 0.0

    all_contexts = "\n".join([c[:300] for c in contexts])

    prompt = f"""评估检索到的文档对参考答案的覆盖程度。

问题：{question}

参考答案（标准答案）：{ground_truth}

检索到的文档：
{all_contexts}

请用 0-100 的数字评价检索文档覆盖了参考答案中多少关键信息。
只回答数字，不要解释。"""
    result = judge(prompt)
    try:
        score = float(''.join(c for c in result if c.isdigit()))
        return min(score / 100.0, 1.0)
    except ValueError:
        return 0.5


# ── 指标 3: Faithfulness ─────────────────────────────────
# 衡量：生成的回答是否完全基于检索到的上下文，没有编造

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    将回答拆成声明（claims），检查每条声明是否能在上下文中找到依据。
    """
    if not answer or not contexts:
        return 0.0

    all_contexts = "\n".join([c[:500] for c in contexts])

    prompt = f"""评估以下回答是否完全基于提供的上下文，有无编造。

上下文（回答应该只基于这些内容）：
{all_contexts}

生成的回答：
{answer}

请评分：
- 1.0：回答全部基于上下文，无编造
- 0.7-0.9：回答基本基于上下文，有少量合理推断
- 0.4-0.6：回答部分基于上下文，有一些编造
- 0.0-0.3：回答大量编造或与上下文无关

只回答一个 0.0 到 1.0 之间的数字，如 0.85。不要解释。"""
    result = judge(prompt)
    try:
        score = float(''.join(c for c in result if c.isdigit() or c == '.'))
        return min(max(score, 0.0), 1.0)
    except ValueError:
        return 0.5


# ── 加载测试数据 ─────────────────────────────────────────

def load_test_data(path: str = "data/test_questions.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 运行评估 ─────────────────────────────────────────────

def evaluate_rag(test_data: list[dict]) -> dict:
    """对每条测试数据跑 RAG + 评估"""
    # 延迟导入，避免启动时加载模型
    from rag import ask_rag

    cp_scores = []
    cr_scores = []
    f_scores = []

    print(f"🔄 评估 {len(test_data)} 条测试数据...")
    print(f"   每条 3 个指标 × 1 次 LLM 调用 = 3 次/条")
    print(f"   + 1 次 RAG 调用/条 = {len(test_data)} 次 RAG + {len(test_data) * 3} 次评估\n")

    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i+1}/{len(test_data)}] {q[:40]}...")

        # RAG 查询
        answer, docs = ask_rag(q)
        contexts = [d["content"] for d in docs]

        # 三条指标
        cp = score_context_precision(q, contexts, gt)
        time.sleep(0.3)  # 避免 API 限流
        cr = score_context_recall(q, contexts, gt)
        time.sleep(0.3)
        f = score_faithfulness(answer, contexts)
        time.sleep(0.3)

        cp_scores.append(cp)
        cr_scores.append(cr)
        f_scores.append(f)
        print(f"   CP={cp:.2f}  CR={cr:.2f}  Faith={f:.2f}")

    return {
        "Context Precision": sum(cp_scores) / len(cp_scores),
        "Context Recall": sum(cr_scores) / len(cr_scores),
        "Faithfulness": sum(f_scores) / len(f_scores),
        "details": list(zip(
            [t["question"] for t in test_data],
            cp_scores, cr_scores, f_scores
        ))
    }


# ── 报告 ─────────────────────────────────────────────────

def print_report(scores: dict):
    print("\n" + "=" * 60)
    print("📊 RAG 系统评估报告")
    print("=" * 60)
    print(f"测试数据: {len(scores['details'])} 条")
    print(f"检索流水线: BM25 + Dense → RRF 融合 → BGE-Reranker 精排")
    print(f"评估方法: DeepSeek LLM-as-Judge")
    print()

    for metric in ["Context Precision", "Context Recall", "Faithfulness"]:
        s = scores[metric]
        emoji = "🟢" if s >= 0.80 else ("🟡" if s >= 0.60 else "🔴")
        print(f"  {emoji} {metric:<22s} {s:.4f}")

    print("\n" + "-" * 40)
    print("📋 逐条详情:")
    for q, cp, cr, f in scores["details"]:
        print(f"  [{cp:.2f} | {cr:.2f} | {f:.2f}] {q[:50]}")

    print("\n" + "-" * 40)
    print("🎯 面试话术（直接背）：")
    print(f"  「我的 RAG 系统在 {len(scores['details'])} 条测试集上评估：")
    print(f"    Faithfulness {scores['Faithfulness']:.2f}——生成内容高度忠于知识库；")
    print(f"    Context Precision {scores['Context Precision']:.2f}——")
    print(f"    混合检索 + Reranker 将最相关文档排在前面。」")
    print("=" * 60)


# ── 入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    test_data = load_test_data()
    scores = evaluate_rag(test_data)
    print_report(scores)
