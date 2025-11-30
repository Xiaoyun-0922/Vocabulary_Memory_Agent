import os
from typing import List, Dict, TypedDict

from langgraph.graph import StateGraph, END

from config import build_llm, VOCAB_FILE


class AgentState(TypedDict):
    """State used by the LangGraph-style workflow."""

    selected_entries: List[Dict[str, str]]  # list of {"word": str, "meaning": str}
    article: str
    model_provider: str  # "openai" or "deepseek"


def load_vocabulary(file_path: str | None = None) -> List[Dict[str, str]]:
    """Load vocabulary entries from a UTF-8 text file.

    Expected format per line:
        word<TAB>Chinese meaning
      or word<space>Chinese meaning
    """

    if file_path is None:
        file_path = VOCAB_FILE

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


def build_prompt(selected_entries: List[Dict[str, str]]) -> str:
    """Build the prompt for the LLM.

    Chinese meanings are included as reference, but the article itself
    is written in English with inline Chinese explanations for each
    target word on first occurrence.
    """

    vocab_lines = "\n".join(
        f"{idx + 1}. {item['word']} - {item['meaning']}" for idx, item in enumerate(selected_entries)
    )

    prompt = f"""You are an English teacher.

Write an engaging English article suitable for CET-6 level students.

Requirements:
1. The article must naturally include ALL of the following vocabulary words at least once:
{vocab_lines}
2. When you use each vocabulary word for the FIRST time, make it bold using Markdown and immediately follow it with a short Chinese explanation in parentheses. Example: **abandon (V. 放弃)**.
3. Do not use any headings. Write the article as plain text.
4. The format of the article should be as narrative as possible. If not all the words are included, make the article long enough to include all the words given to you earlier
5. article generated includes as much as possible normal words except those unknown words, helping user to understand more easily, and make a title for the article.
6. Output only the article content in Markdown format.

Remember: every target word must appear, and its first occurrence must be in the pattern **word (Chinese Explaination)**.
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


def create_graph():
    """Create and compile the LangGraph workflow."""

    graph = StateGraph(AgentState)
    graph.add_node("generate_article", generate_article_node)
    graph.set_entry_point("generate_article")
    graph.add_edge("generate_article", END)
    return graph.compile()
