import os
from typing import List, Dict, TypedDict

import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from fpdf import FPDF


VOCAB_FILE = "CET6_vocabulary.txt"


class AgentState(TypedDict):
    """State for the LangGraph workflow."""

    selected_entries: List[Dict[str, str]]  # [{"word": str, "meaning": str}]
    article: str
    model_provider: str  # "openai" or "deepseek"


@st.cache_data(show_spinner=False)
def load_vocabulary(file_path: str) -> List[Dict[str, str]]:
    """Load vocabulary from a text file.

    Expected format per line (UTF-8):
        word<TAB>中文释义
    or:
        word<space>中文释义
    """

    entries: List[Dict[str, str]] = []
    if not os.path.exists(file_path):
        return entries

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                word, meaning = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if not parts:
                    continue
                word = parts[0]
                meaning = parts[1] if len(parts) > 1 else ""
            entries.append({"word": word.strip(), "meaning": meaning.strip()})

    return entries


def detect_available_providers() -> Dict[str, str]:
    """Detect which model providers are available based on environment variables.

    Returns a mapping from provider_id -> label.
    """

    providers: Dict[str, str] = {}

    # ChatGPT / OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers["openai"] = "ChatGPT (OpenAI)"

    # DeepSeek: prefer dedicated vars, but also try to infer from base URL
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    deepseek_base = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    if deepseek_key and (os.getenv("DEEPSEEK_API_KEY") or "deepseek" in deepseek_base.lower()):
        providers["deepseek"] = "DeepSeek (OpenAI 兼容接口)"

    return providers


def build_llm(provider: str) -> ChatOpenAI:
    """Construct a ChatOpenAI client for the given provider.

    We rely on environment variables so that we don't depend on
    specific keyword arguments of ChatOpenAI.
    """

    if provider == "deepseek":
        # 优先使用专用 DeepSeek 环境变量，如果没有则回退到 OPENAI_* 约定
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
        if not api_key:
            raise ValueError("未检测到 DeepSeek 的 API Key，请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。")

        # 通过统一的 OPENAI_* 环境变量传递给 ChatOpenAI（DeepSeek 兼容 OpenAI 协议）
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    else:  # default: openai
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError("未检测到 OpenAI 的 API Key，请设置 OPENAI_API_KEY。")

        os.environ["OPENAI_API_KEY"] = api_key
        # 允许用户不显式配置 OPENAI_BASE_URL
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 实例化 LLM（从环境变量读取 api_key / base_url）
    return ChatOpenAI(model=model_name)


def build_prompt(selected_entries: List[Dict[str, str]]) -> str:
    """Build the prompt for the LLM.

    We include Chinese meanings as reference in the prompt, but ask the
    model to annotate meanings in English in the article, so that the
    generated text remains ASCII-friendly for PDF export.
    """

    vocab_lines = "\n".join(
        f"{idx + 1}. {item['word']} - {item['meaning']}" for idx, item in enumerate(selected_entries)
    )

    prompt = f"""You are an English teacher.

Write an engaging English article suitable for CET-6 level students.

Requirements:
1. The article must naturally include ALL of the following vocabulary words at least once:
{vocab_lines}
2. When you use each vocabulary word for the FIRST time, make it bold using Markdown and immediately follow it with a short English explanation in parentheses. Example: **abandon (to give up something completely)**.
3. Do not use any headings. Write the article as 3-6 coherent paragraphs of plain text.
4. Output only the article content in Markdown format.

Remember: every target word must appear, and its first occurrence must be in the pattern **word (English explanation)**.
"""

    return prompt


def generate_article_node(state: AgentState) -> AgentState:
    """LangGraph node that calls the LLM to generate the article."""

    selected_entries = state.get("selected_entries", [])
    provider = state.get("model_provider", "openai")

    if not selected_entries:
        raise ValueError("No vocabulary selected.")

    llm = build_llm(provider)
    prompt = build_prompt(selected_entries)

    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)

    new_state: AgentState = {
        "selected_entries": selected_entries,
        "article": content,
        "model_provider": provider,
    }
    return new_state


@st.cache_resource(show_spinner=False)
def create_graph():
    """Create and compile the LangGraph workflow."""

    graph = StateGraph(AgentState)
    graph.add_node("generate_article", generate_article_node)
    graph.set_entry_point("generate_article")
    graph.add_edge("generate_article", END)
    return graph.compile()


def parse_markdown_bold_segments(text: str):
    """Parse a line of Markdown and split into (segment, is_bold) pairs.

    Supports only **bold** markers, which is sufficient for our use case.
    """

    segments = []
    current = []
    bold = False
    i = 0
    length = len(text)

    while i < length:
        if text[i : i + 2] == "**":
            if current:
                segments.append(("".join(current), bold))
                current = []
            bold = not bold
            i += 2
        else:
            current.append(text[i])
            i += 1

    if current:
        segments.append(("".join(current), bold))

    return segments


def build_pdf_from_markdown(markdown_text: str) -> bytes:
    """Convert a simple Markdown string (only **bold**) into a PDF.

    Note: We use core Latin fonts, so the article should be mostly ASCII
    (which is why we ask the model to annotate meanings in English).
    """

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    lines = markdown_text.splitlines() or [markdown_text]

    for line in lines:
        if not line.strip():
            pdf.ln(6)
            continue

        segments = parse_markdown_bold_segments(line)
        for segment, is_bold in segments:
            style = "B" if is_bold else ""
            pdf.set_font("Arial", style=style, size=12)
            pdf.write(6, segment)
        pdf.ln(6)

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="ignore")
    return out


def main():
    st.set_page_config(page_title="Vocabulary Memory Agent", layout="wide")

    st.title("Vocabulary Memory Agent")
    st.markdown(
        "使用 LangGraph + LLM 帮你背六级单词：勾选不认识的单词，让模型生成包含这些单词的英文文章，并导出 PDF。"
    )

    # Sidebar: model selection
    st.sidebar.header("模型配置")

    providers = detect_available_providers()
    if not providers:
        st.sidebar.error(
            "未检测到可用的模型提供商，请先在环境变量中配置：\n"
            "- OpenAI: OPENAI_API_KEY, 可选 OPENAI_BASE_URL\n"
            "- DeepSeek: DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL，或在 OPENAI_BASE_URL 中使用 DeepSeek 地址"
        )
        st.stop()

    provider_ids = list(providers.keys())
    default_index = 0
    provider_label = st.sidebar.selectbox(
        "选择模型提供商",
        options=provider_ids,
        index=default_index,
        format_func=lambda pid: providers[pid],
    )

    st.sidebar.markdown("---")

    # Load vocabulary
    vocab_entries = load_vocabulary(VOCAB_FILE)
    if not vocab_entries:
        st.error(f"未找到词表文件 `{VOCAB_FILE}` 或文件为空，请确认文件存在且为 UTF-8 编码。")
        st.stop()

    st.sidebar.write(f"总单词数：**{len(vocab_entries)}**")

    st.subheader("1. 勾选你不认识的单词")
    st.caption("提示：词表可能较长，建议配合浏览器搜索或分段浏览来勾选生词。")

    unknown_entries: List[Dict[str, str]] = []

    # 显示词表和勾选框
    for idx, entry in enumerate(vocab_entries):
        cols = st.columns([2, 5, 2])
        with cols[0]:
            st.write(entry["word"])
        with cols[1]:
            st.write(entry["meaning"])
        with cols[2]:
            checked = st.checkbox("不认识", key=f"unknown_{idx}")
        if checked:
            unknown_entries.append(entry)

    st.sidebar.write(f"当前勾选生词数：**{len(unknown_entries)}**")

    st.markdown("---")
    st.subheader("2. 生成包含生词的英文文章，并导出 PDF")

    generate_button = st.button("生成文章并预览 / 导出 PDF", type="primary")

    if generate_button:
        if not unknown_entries:
            st.warning("请至少勾选一个不认识的单词。")
            return

        graph = create_graph()

        with st.spinner("正在调用大语言模型生成文章，请稍候..."):
            try:
                result: AgentState = graph.invoke(
                    {
                        "selected_entries": unknown_entries,
                        "article": "",
                        "model_provider": provider_label,
                    }
                )
            except Exception as e:  # pylint: disable=broad-except
                st.error(f"生成文章时出错：{e}")
                return

        article = result.get("article", "").strip()
        if not article:
            st.error("模型未返回任何内容，请稍后重试或检查模型配置。")
            return

        st.success("文章生成完成！")
        st.markdown("### 生成的文章（支持 Markdown 加粗显示）")
        st.markdown(article)

        pdf_bytes = build_pdf_from_markdown(article)

        st.download_button(
            label="下载文章 PDF",
            data=pdf_bytes,
            file_name="vocabulary_article.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
