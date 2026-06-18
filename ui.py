"""
Step 4: Streamlit 聊天界面
用法: streamlit run ui.py
"""
import streamlit as st
import requests

st.set_page_config(page_title="RAG 知识库问答", page_icon="🧠")
st.title("🧠 AI 知识库问答系统")

st.markdown("""
**RAG v2.0** — 混合检索 + 重排序

检索流水线：BM25 + Dense → RRF 融合 → BGE-Reranker 精排 → DeepSeek 生成
""")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 混合检索 → 重排序 → 生成..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": prompt},
                    timeout=30
                )
                data = resp.json()
                answer = data["answer"]
                method = data.get("retrieval_method", "")

                # 显示检索来源（含 RRF + Rerank 分数）
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📎 检索来源（{method}，共 {len(sources)} 条）"):
                        for i, s in enumerate(sources):
                            rrf = s.get("rrf_score", "N/A")
                            rerank = s.get("rerank_score", "N/A")
                            st.caption(
                                f"[{i+1}] RRF: {rrf} → Rerank: {rerank}"
                            )
                            st.text(s["content"][:300])

            except Exception as e:
                answer = f"❌ 连接失败: {e}\n\n请确保 FastAPI 已启动（`python app.py`）"

        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
