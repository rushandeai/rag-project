"""
Step 4: Streamlit 界面 - 可视化聊天 UI
用法: streamlit run ui.py
"""

import streamlit as st
import requests

st.set_page_config(page_title="RAG 知识库问答", page_icon="🧠")
st.title("🧠 AI 知识库问答系统")

st.markdown("""
基于 **RAG（检索增强生成）** 的智能问答系统。
提问关于 AI 技术的任何问题，系统会从知识库中检索相关内容并生成回答。
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
    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用后端 API
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索知识库..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": prompt},
                    timeout=30
                )
                answer = resp.json()["answer"]
            except Exception as e:
                answer = f"❌ 连接失败: {e}\n\n请确保 FastAPI 服务已启动（`uvicorn app:app --port 8000`）"

        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
