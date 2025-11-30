# Vocabulary Memory Agent

一个基于 Streamlit + 大语言模型的六级单词记忆小助手。

- 从 `CET6_vocabulary.txt` 中加载**全部六级单词**及其中文释义。
- 在网页中浏览整张词表，**手动勾选自己“不认识/不熟悉”的单词**。
- 调用 LLM（支持 OpenAI / DeepSeek）生成一篇英文文章：
  - 在文中自然地使用你勾选的所有单词；
  - 第一次出现时用 Markdown 加粗，并在括号中给出简短中文解释。
- 在浏览器中预览文章，并导出为 **TXT 文件** 方便保存和打印。

> 说明：
> 目前词表只包含 **CET-6（大学英语六级）** 单词。
> 从这么长的词表中挑出自己不认识的词，这一步确实有点耗时，
> 但也是记忆过程中**必须亲自经历的一步**：
> 你在勾选的过程中，其实已经在做一次快速的“自我测试”。

---

## 环境与技术栈

- 推荐 Python 版本：**3.10**（建议使用 conda 虚拟环境）。
- 技术栈：
  - **LangChain / langchain-openai**：封装 LLM 调用；
  - **OpenAI / DeepSeek**：OpenAI 协议兼容接口；
  - **Streamlit**：构建 Web 界面。

---

## 1. 创建 conda 环境

在终端中执行（Windows / macOS / Linux 通用）：

```bash
conda create -n vocabulary_memory_agent python=3.10
conda activate vocabulary_memory_agent
```

---

## 2. 安装依赖

确保当前终端所在目录是本项目根目录（包含 `requirements.txt` 的目录），然后执行：

```bash
pip install -r requirements.txt
```

如果已安装 `make`，也可以直接：

```bash
make install
```

---

## 3. 大模型配置（OpenAI & DeepSeek）

### 3.1 OpenAI

以 PowerShell 为例，设置环境变量：

```bash
# PowerShell
$env:OPENAI_API_KEY = "你的_openai_api_key"
$env:OPENAI_BASE_URL = "https://你的代理域名/v1"
```

程序会从环境变量中读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。

### 3.2 DeepSeek（OpenAI 兼容接口）

DeepSeek 提供 OpenAI 协议兼容接口，有两种配置方式：

- 在 `.env` 中使用专门的变量：

  ```env
  DEEPSEEK_API_KEY=...
  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
  ```

- 或者直接把通用的 OPENAI 变量指向 DeepSeek：

  ```powershell
  $env:OPENAI_API_KEY = "your_deepseek_api_key"
  $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
  ```

---

## 4. 词表文件格式（`CET6_vocabulary.txt`）

请将六级词汇表保存在项目根目录的 `CET6_vocabulary.txt` 中（UTF-8 编码）。
每行一个单词，例如：

```text
abandon	抛弃；放弃
ability	能力；才能
abnormal	反常的；不正常的
...
```

英文单词与中文释义之间可以用 **制表符 (Tab)** 或 **空格** 分隔。

---

## 5. 运行应用

1. 激活环境：

   ```bash
   conda activate vocabulary_memory_agent
   ```

2. 启动 Streamlit：

   ```bash
   streamlit run main.py
   ```

   如果你更习惯 `make`：

   ```bash
   make run
   ```

3. 在浏览器界面中：

   - 浏览所有六级单词及其中文释义；
   - 为“不认识/不熟悉”的单词打勾（这一步会比较花时间，但非常重要）；
   - 点击按钮，让大模型生成一篇包含所有勾选单词的英文文章；
   - 在页面中预览文章，并将其导出为 TXT 文件保存。



