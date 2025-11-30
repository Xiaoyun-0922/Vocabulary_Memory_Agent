import os
import traceback
from typing import List, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

from config import VOCAB_FILE, detect_available_providers, build_llm
from workflow import load_vocabulary, build_prompt


@st.cache_data(show_spinner=False)
def get_vocabulary(file_path: str, file_mtime: Optional[float]):
    return load_vocabulary(file_path)


def main():
    st.set_page_config(page_title="Vocabulary Memory Agent", layout="wide")

    if "generated_article" not in st.session_state:
        st.session_state["generated_article"] = ""

    st.title("Vocabulary Memory Agent")
    st.markdown(
        "Agent辅助背诵六级单词：勾选不认识的单词，让模型生成包含这些单词的英文文章，并导出TXT文件。"
    )

    # Sidebar: model selection
    st.sidebar.header("模型配置")

    providers = detect_available_providers()
    if not providers:
        st.sidebar.error(
            "未检测到可用的模型提供商，请先在项目根目录创建 .env 文件并配置：\n"
            "- OpenAI: OPENAI_API_KEY, 可选 OPENAI_BASE_URL\n"
            "- DeepSeek: DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL，或在 OPENAI_BASE_URL 中使用 DeepSeek 地址"
        )
        st.stop()

    model_options = []

    if "openai" in providers:
        openai_models = ["gpt-4o", "gpt-4o-mini"]
        for model_name in openai_models:
            model_options.append(
                {
                    "key": f"openai::{model_name}",
                    "provider": "openai",
                    "model": model_name,
                    "label": f"OpenAI · {model_name}",
                }
            )

    if "deepseek" in providers:
        deepseek_models = ["deepseek-chat"]
        for model_name in deepseek_models:
            model_options.append(
                {
                    "key": f"deepseek::{model_name}",
                    "provider": "deepseek",
                    "model": model_name,
                    "label": f"DeepSeek · {model_name}",
                }
            )

    if not model_options:
        st.sidebar.error("已检测到模型提供商，但未配置任何可用模型，请检查环境变量中的 *MODEL 配置。")
        st.stop()

    option_keys = [opt["key"] for opt in model_options]

    def _format_model_option(key: str) -> str:
        for opt in model_options:
            if opt["key"] == key:
                return opt["label"]
        return key

    selected_key = st.sidebar.selectbox(
        "选择模型",
        options=option_keys,
        index=0,
        format_func=_format_model_option,
    )

    selected_option = next(opt for opt in model_options if opt["key"] == selected_key)
    selected_provider = selected_option["provider"]
    selected_model = selected_option["model"]

    if selected_provider == "openai":
        os.environ["OPENAI_MODEL"] = selected_model
    elif selected_provider == "deepseek":
        os.environ["DEEPSEEK_MODEL"] = selected_model

    st.sidebar.markdown("---")

    # Load vocabulary data
    try:
        vocab_mtime = os.path.getmtime(VOCAB_FILE)
    except OSError:
        vocab_mtime = None

    vocab_entries = get_vocabulary(VOCAB_FILE, vocab_mtime)
    if not vocab_entries:
        st.error(f"未找到词表文件 `{VOCAB_FILE}` 或文件为空，请确认文件存在且为 UTF-8 编码。")
        st.stop()

    st.sidebar.write(f"总单词数：**{len(vocab_entries)}**")

    st.subheader("1. 勾选你不认识的单词")
    st.caption("提示：词表可能较长，建议配合浏览器搜索或分段浏览来勾选生词。")

    unknown_entries: List[Dict[str, str]] = []

    # Render vocabulary list and checkboxes
    num_cols = 4
    cols = st.columns(num_cols)

    for idx, entry in enumerate(vocab_entries):
        col = cols[idx % num_cols]
        label = f"{entry['word']}  {entry['meaning']}"
        with col:
            checked = st.checkbox(label, key=f"unknown_{idx}")
        if checked:
            unknown_entries.append(entry)

    st.sidebar.write(f"当前勾选生词数：**{len(unknown_entries)}**")

    st.markdown("---")
    st.subheader("2. 生成包含生词的英文文章")

    generate_button = st.button("生成文章", type="primary")

    st.markdown("<a id='article-section'></a>", unsafe_allow_html=True)

    article_header_placeholder = st.empty()
    article_placeholder = st.empty()

    article_from_state = (st.session_state.get("generated_article") or "").strip()

    if generate_button:
        if not unknown_entries:
            st.warning("请至少勾选一个不认识的单词。")
            return

        components.html("<script>window.location.hash = '#article-section';</script>", height=0)

        llm = build_llm(selected_provider)
        prompt = build_prompt(unknown_entries)

        full_text = ""

        try:
            with st.spinner("正在调用大语言模型生成文章，请稍候..."):
                for chunk in llm.stream(prompt):
                    chunk_content = getattr(chunk, "content", "")
                    if not isinstance(chunk_content, str):
                        chunk_content = str(chunk_content)
                    if not chunk_content:
                        continue
                    full_text += chunk_content
                    article_header_placeholder.markdown("### 例文")
                    article_placeholder.markdown(full_text)
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"生成文章时出错：{type(e).__name__}: {e}")
            st.code("".join(traceback.format_exception(e)), language="text")
            return

        article = full_text.strip()
        if not article:
            st.error("模型未返回任何内容，请稍后重试或检查模型配置。")
            return

        st.session_state["generated_article"] = article
        article_from_state = article
        st.success("文章生成完成！")

    if article_from_state:
        # Display the full article and provide TXT download
        article_header_placeholder.markdown("### 例文")
        article_placeholder.markdown(article_from_state)

        st.download_button(
            label="下载文章 TXT",
            data=article_from_state.encode("utf-8"),
            file_name="vocabulary_article.txt",
            mime="text/plain; charset=utf-8",
        )


if __name__ == "__main__":
    main()

