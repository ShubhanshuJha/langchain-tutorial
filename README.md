# LangChain Tutorial — From Basics to Advanced

A structured walkthrough of the **LangChain** ecosystem: what it is, why it exists, and how each moving part fits together. This guide moves from first principles (models and prompts) through the pieces most projects actually assemble in production (chains, RAG, agents) and out to the frontier work (LangGraph, evaluation, multi-agent systems).

Every section is written to be read in order, but each concept is self-contained enough to skim to when you need it.

---

## Table of contents

**Foundations**
1. [What is LangChain?](#1-what-is-langchain)
2. [Installation and setup](#2-installation-and-setup)
3. [The LangChain ecosystem](#3-the-langchain-ecosystem)

**Basics**
4. [LLMs vs Chat Models](#4-llms-vs-chat-models)
5. [Prompt Templates](#5-prompt-templates)
6. [Output Parsers](#6-output-parsers)
7. [Chains: the classic view](#7-chains-the-classic-view)
8. [LangChain Expression Language (LCEL)](#8-langchain-expression-language-lcel)

**Intermediate**
9. [Memory](#9-memory)
10. [Document Loaders](#10-document-loaders)
11. [Text Splitters](#11-text-splitters)
12. [Embeddings](#12-embeddings)
13. [Vector Stores](#13-vector-stores)
14. [Retrievers](#14-retrievers)
15. [Retrieval-Augmented Generation (RAG)](#15-retrieval-augmented-generation-rag)

**Advanced**
16. [Tools and Function Calling](#16-tools-and-function-calling)
17. [Agents](#17-agents)
18. [LangGraph: stateful, cyclical workflows](#18-langgraph-stateful-cyclical-workflows)
19. [Streaming, Callbacks, and Tracing](#19-streaming-callbacks-and-tracing)
20. [Structured Output](#20-structured-output)
21. [Multi-modal chains](#21-multi-modal-chains)
22. [Evaluation with LangSmith](#22-evaluation-with-langsmith)
23. [Production concerns](#23-production-concerns)

**Reference**
24. [Common pitfalls](#24-common-pitfalls)
25. [Further reading](#25-further-reading)

---

## 1. What is LangChain?

LangChain is a framework for building applications powered by Large Language Models (LLMs). It exists because a raw call to an LLM API is only ever the last step of a real application — everything around it (fetching the right context, formatting the prompt, parsing the output, calling tools, remembering prior turns, retrying failures) is what LangChain gives you a vocabulary and a set of composable primitives for.

**The mental model.** Think of LangChain as three layers on top of the model API:

| Layer | Purpose | Example primitives |
|---|---|---|
| **I/O** | Format input, parse output | `PromptTemplate`, `OutputParser` |
| **Composition** | Combine steps into pipelines | LCEL (`\|` operator), `Runnable` |
| **Application** | Domain patterns | RAG, agents, memory, tools |

**When to reach for it — and when not to.** LangChain shines when your application composes multiple LLM calls, or mixes LLM calls with retrieval, tools, or control flow. For a single one-shot prompt against a single model, the vendor SDK is often enough and lighter. Reach for LangChain when the shape of your application starts to look like a *pipeline* rather than a *call*.

---

## 2. Installation and setup

The framework is modular — you install the pieces you use rather than one monolithic package.

```bash
# Core abstractions (Runnable, PromptTemplate, etc.)
pip install langchain-core

# The "batteries-included" package for common chains and legacy imports
pip install langchain

# Community integrations (many vector stores, loaders, tools)
pip install langchain-community

# Provider packages — install only what you need
pip install langchain-openai         # OpenAI
pip install langchain-anthropic      # Anthropic (Claude)
pip install langchain-google-genai   # Google Gemini
pip install langchain-ollama         # Local models via Ollama

# Graph-based orchestration
pip install langgraph
```

**API keys.** Providers are configured through environment variables. Use a `.env` file loaded with `python-dotenv` — do not hardcode keys.

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
```

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 3. The LangChain ecosystem

LangChain is not one library — it is a family of related packages, each solving one problem:

| Package | What it is |
|---|---|
| `langchain-core` | The bedrock abstractions: `Runnable`, `BaseMessage`, `PromptTemplate`, `OutputParser`. Zero heavy dependencies. |
| `langchain` | Higher-level chains and application-level building blocks that compose the core primitives. |
| `langchain-community` | Community-contributed integrations — loaders, vector stores, tools that live outside the first-party providers. |
| `langchain-<provider>` | Official integration packages, one per model provider (OpenAI, Anthropic, Google, Ollama, …). |
| `langgraph` | Graph-based orchestration for stateful, cyclical multi-step agents. |
| `langsmith` | Observability, tracing, evaluation, prompt management. |
| `langserve` | Deploy a LangChain `Runnable` as a REST API with one line. |

The split matters: `langchain-core` is intentionally small so it can be a stable dependency, while integrations evolve independently in their own packages.

---

## 4. LLMs vs Chat Models

LangChain distinguishes two model interfaces:

- **LLMs** (`BaseLLM`) — take a string, return a string. The older, "completion" style.
- **Chat Models** (`BaseChatModel`) — take a list of `BaseMessage` objects (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`), return a `BaseMessage`. This is what every modern provider actually exposes.

**Always prefer Chat Models.** Even for single-turn use, chat models give you role separation (system vs user), native tool-calling support, and forward compatibility.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

response = llm.invoke([
    SystemMessage(content="You are a terse Python expert."),
    HumanMessage(content="Explain list comprehensions in two sentences."),
])
print(response.content)
```

**Common parameters.**

| Parameter | Effect |
|---|---|
| `temperature` | Randomness. `0` for deterministic extractive work, `0.7+` for creative. |
| `max_tokens` | Cap on output length. |
| `top_p` | Nucleus sampling — usually leave at default. |
| `timeout` | Request timeout in seconds. |
| `max_retries` | Automatic retries on transient errors. |

---

## 5. Prompt Templates

A `PromptTemplate` is a string with named placeholders and the machinery to fill them safely. It lets you separate the *shape* of a prompt from the *data* that flows through it.

**`PromptTemplate` — single-turn.**

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Translate the following English text to {language}:\n\n{text}"
)
prompt = template.invoke({"language": "French", "text": "Good morning"})
```

**`ChatPromptTemplate` — the one you actually want.** It builds a list of role-tagged messages, which is what chat models consume:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialising in {domain}."),
    ("human", "{question}"),
])

messages = prompt.invoke({"domain": "astronomy", "question": "Why is the sky blue?"})
```

**`MessagesPlaceholder` — for conversation history.** Use this when a list of messages needs to be injected at a specific position:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

**Few-shot prompting.** For teaching by example, use `FewShotChatMessagePromptTemplate`:

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "3 * 5", "output": "15"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator."),
    few_shot,
    ("human", "{input}"),
])
```

---

## 6. Output Parsers

An `OutputParser` converts the model's raw string response into a structured Python object. Parsers also generate the *format instructions* that get injected into the prompt so the model knows what shape to emit.

**`StrOutputParser`** — the identity parser, useful in LCEL to extract `.content` from a message:

```python
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
```

**`JsonOutputParser`** — parses JSON responses:

```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
```

**`PydanticOutputParser`** — parses into a validated Pydantic model, giving you typed access and validation for free:

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")

parser = PydanticOutputParser(pydantic_object=Person)
```

Prefer the model's native structured-output support (see [Section 20](#20-structured-output)) when the provider exposes it — parsers are the fallback for models that don't.

---

## 7. Chains: the classic view

Historically, a "chain" was any object that composed multiple steps — `LLMChain`, `SequentialChain`, `RetrievalQA`, `ConversationChain`, and so on. These are now **legacy**. Modern LangChain composes chains using LCEL (next section), which is more explicit, streamable, and debuggable.

You will still see legacy chains in older codebases and tutorials:

```python
# Legacy — avoid in new code
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(question="What is Python?")
```

Every legacy chain has a direct LCEL equivalent. The rest of this guide uses LCEL exclusively.

---

## 8. LangChain Expression Language (LCEL)

LCEL is a small declarative language for composing `Runnable` objects with the pipe operator (`|`). It is the modern way to build every chain in LangChain.

**The core insight.** Anything that can be `.invoke()`d — a prompt, a model, a parser, a retriever, an arbitrary Python function — is a `Runnable`. Piping two runnables produces a new runnable whose output of the first feeds the input of the second.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | llm | parser
chain.invoke({"topic": "database indexes"})
```

**Every LCEL chain automatically supports:**

| Method | What it does |
|---|---|
| `.invoke(input)` | Run synchronously, return a single result |
| `.ainvoke(input)` | Async version |
| `.batch(inputs)` | Run over a list, parallelised where possible |
| `.stream(input)` | Yield tokens as they arrive |
| `.astream(input)` | Async streaming |

You do not need to opt in to any of these — implementing one runnable gives you all five interfaces for free.

**`RunnableParallel` — fan out to multiple branches:**

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    joke=prompt_joke | llm | parser,
    poem=prompt_poem | llm | parser,
) | combine_step
```

Both branches run concurrently.

**`RunnablePassthrough` — inject the input into the output shape:**

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
chain.invoke("What is LCEL?")
```

The dict here is a common RAG shape: the question is passed through unchanged while the retriever fetches context in parallel.

**`RunnableLambda` — lift any function into a chain:**

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

chain = prompt | llm | parser | RunnableLambda(uppercase)
```

**Why LCEL matters.** The pipe syntax is not just sugar — it is what enables uniform streaming, batching, async, and tracing across every combination. When you compose custom logic with `RunnableLambda`, it participates in the same interface as everything else.

---

## 9. Memory

An LLM has no memory of prior conversation turns. Memory in LangChain is the pattern of storing past messages and reinjecting them into subsequent prompts.

**Modern approach — `RunnableWithMessageHistory`.** Wrap any chain with a per-session message store:

```python
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, BaseChatMessageHistory] = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "Hi, I'm Alice."},
    config={"configurable": {"session_id": "user-123"}},
)
```

**Trimming.** Full history grows without bound. Use `trim_messages` to keep only the last N tokens or messages:

```python
from langchain_core.messages import trim_messages
trimmer = trim_messages(max_tokens=1000, strategy="last", token_counter=llm)
```

**Persistent memory.** `InMemoryChatMessageHistory` disappears on restart. For durability, use `RedisChatMessageHistory`, `PostgresChatMessageHistory`, or the file-based `FileChatMessageHistory` — all in `langchain-community`.

**Legacy memory classes** (`ConversationBufferMemory`, `ConversationSummaryMemory`, etc.) are deprecated in favor of the runnable-based approach. Avoid them in new code.

---

## 10. Document Loaders

A `DocumentLoader` reads content from a source (a file, a URL, a database) and returns a list of `Document` objects, each with `page_content` and `metadata`.

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

pdf_docs = PyPDFLoader("report.pdf").load()          # one Document per page
txt_docs = TextLoader("notes.txt").load()
web_docs = WebBaseLoader("https://example.com").load()
```

**Common loaders.**

| Loader | Source | Notes |
|---|---|---|
| `PyPDFLoader` | PDF | One page per document — enables page-level citations |
| `TextLoader` | `.txt` | UTF-8 by default |
| `CSVLoader` | `.csv` | One row per document |
| `Docx2txtLoader` | `.docx` | Requires `docx2txt` |
| `UnstructuredExcelLoader` | `.xlsx` | Requires `unstructured` + `openpyxl` |
| `JSONLoader` | `.json` | Needs `jq_schema` to select fields |
| `WebBaseLoader` | URL | Scrapes and cleans HTML |
| `DirectoryLoader` | folder | Dispatches per-format |
| `GitHubIssuesLoader` | GitHub | Auth via token |

**Metadata carries provenance.** The metadata dict — typically `{"source": ..., "page": ...}` — is what makes citations possible downstream. Every loader emits its own metadata shape; keep this in mind when building unified pipelines.

---

## 11. Text Splitters

Embedding models and LLMs have context limits, so long documents must be chunked. The splitter decides *where* to cut.

**`RecursiveCharacterTextSplitter` — the default choice.** It tries a list of separators in descending granularity — paragraphs first, then lines, then words — so cuts land at natural boundaries whenever possible:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
```

**Overlap matters.** A fact split across a chunk boundary must appear intact in *one* of the chunks, or it becomes unretrievable. 10–15% of `chunk_size` is a sensible default.

**Other splitters worth knowing.**

| Splitter | Use case |
|---|---|
| `CharacterTextSplitter` | Simple single-separator splitting — usually inferior to the recursive version |
| `TokenTextSplitter` | Splits on model tokens rather than characters — accurate for context-budget planning |
| `MarkdownHeaderTextSplitter` | Chunks by heading level, preserving hierarchy in metadata |
| `RecursiveCharacterTextSplitter.from_language` | Language-aware splitting for source code (Python, JS, etc.) |
| `SemanticChunker` | Cuts on embedding-similarity discontinuities — smarter but slower |

**Choose chunk size to match your content.** Prose and technical documentation tolerate 1000–1500 chars. Dense reference material and Q&A do better at 500–800, where each chunk stays on-topic. Long-form narrative can go 2000+.

---

## 12. Embeddings

An embedding model maps a piece of text to a dense vector in some high-dimensional space, such that semantically similar texts land near each other.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vec = embeddings.embed_query("What is LangChain?")           # single vector
vecs = embeddings.embed_documents(["doc one", "doc two"])    # batch
```

**Common providers.**

| Provider | Class | Notes |
|---|---|---|
| OpenAI | `OpenAIEmbeddings` | `text-embedding-3-small` (1536-dim) and `-large` (3072-dim) |
| Cohere | `CohereEmbeddings` | Strong on retrieval benchmarks |
| HuggingFace | `HuggingFaceEmbeddings` | Local; wide model choice |
| Ollama | `OllamaEmbeddings` | Fully local, no API key |
| Google | `GoogleGenerativeAIEmbeddings` | Gemini family |

**The critical invariant.** The embedding model used to *index* documents must be the same one used to *query* them. Different models produce vectors in incompatible spaces — querying across them yields confident nonsense. If you change embedding models, delete and rebuild the vector store.

---

## 13. Vector Stores

A vector store persists embeddings alongside their source text and metadata, and answers similarity queries efficiently.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

results = vector_store.similarity_search("What is LCEL?", k=4)
```

**The landscape.**

| Store | Deployment | Notes |
|---|---|---|
| **Chroma** | Local, embedded | SQLite + HNSW. Great for prototypes and small production. |
| **FAISS** | In-process | Fast, no persistence unless you save/load. |
| **Pinecone** | Managed cloud | Scales to billions of vectors. Metadata filtering. |
| **Weaviate** | Self-hosted or managed | Hybrid search built in. |
| **Qdrant** | Self-hosted or managed | Rust-based, strong metadata filtering. |
| **PGVector** | Postgres extension | Vectors alongside relational data. |
| **Milvus** | Self-hosted or managed | Purpose-built for very large scale. |
| **Redis** | Redis Stack | Vectors alongside your existing cache. |

**Metadata filtering.** Every serious vector store supports filtering by metadata alongside similarity — indispensable for multi-tenant systems or narrowing to a specific document set:

```python
results = vector_store.similarity_search(
    "renewal terms",
    k=4,
    filter={"source": "master_agreement.pdf"},
)
```

**Similarity vs distance.** Different stores return different scales. Chroma returns cosine *distance*, so `similarity = 1 - distance`. Read the store's docs before threshold-tuning.

---

## 14. Retrievers

A `Retriever` is any `Runnable` that takes a query string and returns a list of `Document`s. Every vector store exposes one via `.as_retriever()`, but retrievers are broader than vector stores.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
docs = retriever.invoke("What are the renewal terms?")
```

**Search types.**

| Type | Behaviour |
|---|---|
| `similarity` | Standard top-k nearest neighbours |
| `similarity_score_threshold` | Filter out results below a score floor |
| `mmr` | Maximum Marginal Relevance — trades relevance for diversity, avoiding near-duplicate chunks |

**Advanced retrievers.**

- **`MultiQueryRetriever`** — asks an LLM to rewrite the query in multiple forms, retrieves for each, and merges results. Helps with imprecise queries.
- **`ContextualCompressionRetriever`** — pipes retrieved chunks through a compressor (often an LLM) that extracts only the query-relevant sentences.
- **`ParentDocumentRetriever`** — indexes small chunks for precise retrieval but returns their larger parent documents for context.
- **`SelfQueryRetriever`** — parses natural-language queries into `(semantic query, metadata filter)` pairs using an LLM.
- **`EnsembleRetriever`** — combines multiple retrievers (e.g. BM25 keyword + dense semantic) with Reciprocal Rank Fusion.
- **`BM25Retriever`** — classical keyword retrieval; strong on identifier-heavy technical content where exact tokens matter.

Hybrid search — dense semantic **plus** BM25 keyword, fused — is the single most impactful upgrade over pure vector search for real-world corpora.

---

## 15. Retrieval-Augmented Generation (RAG)

RAG is the pattern of grounding an LLM's answer in retrieved documents rather than parametric memory. The canonical LCEL shape:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context. If the context does not
contain the answer, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What are the renewal terms?")
```

**Anatomy.** Every RAG system has the same six stages:

1. **Load** documents from sources.
2. **Split** them into chunks.
3. **Embed** the chunks.
4. **Store** them in a vector store.
5. **Retrieve** the top-k relevant chunks per query.
6. **Generate** an answer conditioned on those chunks, with citations.

Stages 1–4 are indexing (offline, one-time). Stages 5–6 are the query path (per-request).

**Design decisions that actually matter.**

- **Chunk size and overlap** — controls recall vs coherence.
- **Retriever strategy** — dense-only vs hybrid; single-query vs multi-query.
- **Top-k and threshold** — recall vs noise. Too much context degrades answer quality even when the model *can* handle it.
- **Prompt discipline** — instruct the model to abstain when context is insufficient. Grounded models must refuse to guess.
- **Reranking** — a cross-encoder reranker between retrieval and generation (fetch top-20, keep top-5) dramatically improves quality on hard queries.
- **Citations** — surface source metadata alongside answers. Answers you cannot audit are answers you cannot trust.

**Advanced RAG patterns worth studying:**

- **HyDE** (Hypothetical Document Embeddings) — embed a hallucinated answer rather than the raw query.
- **Query decomposition** — break a multi-part question into sub-queries, retrieve for each, then synthesise.
- **Corrective RAG (CRAG)** — grade retrieved documents and fall back to web search on low confidence.
- **Self-RAG** — the model decides *whether* to retrieve, and self-critiques its own output.
- **GraphRAG** — build a knowledge graph over the corpus and traverse it during retrieval.

---

## 16. Tools and Function Calling

A **tool** is a Python function the LLM can choose to call. The framework handles the schema conversion, invocation, and result routing.

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 22 degC and sunny."
```

The `@tool` decorator introspects the function signature and docstring to build the JSON schema the model needs. Docstrings are not decoration — they are what the model reads to decide when to use the tool.

**Bind tools to a model:**

```python
llm_with_tools = llm.bind_tools([multiply, get_weather])
response = llm_with_tools.invoke("What's 27 times 43?")
print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 27, 'b': 43}, 'id': '...'}]
```

**Execute the calls and feed results back:**

```python
from langchain_core.messages import HumanMessage, ToolMessage

messages = [HumanMessage("What is 27 * 43?")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for call in ai_msg.tool_calls:
    tool_fn = {"multiply": multiply, "get_weather": get_weather}[call["name"]]
    result = tool_fn.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = llm_with_tools.invoke(messages)
```

This loop — call model → execute tools → feed results → call model again — is exactly what an agent does.

---

## 17. Agents

An agent is a loop: the LLM chooses actions (tool calls), tools execute, results feed back, the LLM decides what to do next. It stops when the model returns a plain answer instead of a tool call.

**Modern approach: LangGraph's prebuilt ReAct agent.**

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[multiply, get_weather])
result = agent.invoke({
    "messages": [("user", "What's the weather in Paris, and what's 15 * 27?")]
})
```

Under the hood this is a graph: a "call model" node and a "call tools" node, with a conditional edge back to the model until it stops requesting tools.

**Legacy agents** (`initialize_agent`, `AgentExecutor`, `ZeroShotAgent`, etc.) are deprecated. Modern LangChain routes all agent construction through LangGraph.

**When agents work well.** When the model needs to reason across several steps that depend on intermediate results (fetch data → compute → decide next fetch). When the tool call structure is dynamic — you don't know in advance which tools will be needed or in what order.

**When they don't.** For deterministic pipelines, a fixed LCEL chain is faster, cheaper, and more predictable. Agents introduce nondeterminism and latency; use them only when their flexibility earns its keep.

---

## 18. LangGraph: stateful, cyclical workflows

LangGraph is the successor to `AgentExecutor`. It models an application as a **state graph**: nodes are functions that read and update shared state, edges route between nodes (conditionally or unconditionally), and cycles are first-class.

**The mental model.** LCEL composes runnables into a DAG — directed and acyclic. LangGraph adds cycles, branching, and durable state. Anything that involves "keep going until X" (agents, planners, self-critique loops) belongs in LangGraph.

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)

app = graph.compile()
app.invoke({"messages": [HumanMessage("Hello!")]})
```

**Why it matters.**

- **Persistence.** Compile with a `checkpointer` and every state transition is durable. Resume conversations across processes.
- **Human-in-the-loop.** Interrupt before a node executes, wait for human approval, then resume.
- **Time travel.** Rewind to any previous state and re-run from there.
- **Streaming.** Stream at the token, message, or state-transition level.
- **Subgraphs.** Compose graphs as nodes within other graphs.

**Common patterns.**

- **ReAct agents** — model node + tools node + conditional edge (via `create_react_agent`).
- **Plan-and-execute** — planner node emits a step list, executor node runs them one at a time.
- **Reflection loops** — generator node produces output, critic node evaluates it, cycle until the critic passes.
- **Multi-agent** — several specialist agents as nodes, a supervisor node that routes between them.

LangGraph is where advanced LangChain lives today. If you are building anything more sophisticated than a linear pipeline, learn it.

---

## 19. Streaming, Callbacks, and Tracing

**Streaming.** Every LCEL chain supports `.stream()` and `.astream()` — no extra plumbing:

```python
for chunk in chain.stream({"topic": "cats"}):
    print(chunk, end="", flush=True)
```

For finer-grained events (which node ran, what it emitted), use `.astream_events(version="v2")`. This yields structured events for every runnable in the chain and is what production streaming UIs are built on.

**Callbacks.** The `BaseCallbackHandler` interface lets you hook into lifecycle events — `on_llm_start`, `on_llm_new_token`, `on_tool_end`, and so on. Use it for custom logging, metrics, or UI updates:

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.tokens = 0
    def on_llm_new_token(self, token, **kwargs):
        self.tokens += 1

counter = TokenCounter()
chain.invoke({"topic": "cats"}, config={"callbacks": [counter]})
```

**Tracing with LangSmith.** Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in your environment. Every LCEL run is automatically traced — each step's input, output, latency, and token count is captured in a UI you can drill into. This is the single fastest way to debug a chain you don't understand.

---

## 20. Structured Output

The reliable way to get typed output from a modern model is **`with_structured_output`**, which uses the provider's native tool-calling machinery under the hood.

```python
from pydantic import BaseModel, Field
from typing import Literal

class Classification(BaseModel):
    """Classification of a customer support ticket."""
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="One-sentence summary")

structured_llm = llm.with_structured_output(Classification)

result = structured_llm.invoke("My credit card was charged twice for last month.")
print(result.category, result.urgency)
```

The return value is a validated Pydantic instance. No parsing, no format-instruction gymnastics.

**When to use each approach:**

| Approach | Use when |
|---|---|
| `.with_structured_output(schema)` | Provider supports tool calling (OpenAI, Anthropic, Google, most others). This is the default. |
| `JsonOutputParser` | Provider has no native structured output, or you need a fallback. |
| `PydanticOutputParser` | Same, plus you want validation. |

---

## 21. Multi-modal chains

Modern chat models (GPT-4o, Claude, Gemini) accept images alongside text. LangChain surfaces this as content-block messages:

```python
from langchain_core.messages import HumanMessage
import base64

with open("chart.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

message = HumanMessage(content=[
    {"type": "text", "text": "What does this chart show?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
])
response = llm.invoke([message])
```

The same shape works for URL-referenced images, and for Anthropic's document-input format. Audio and video are supported by provider-specific integrations where the model itself supports them.

---

## 22. Evaluation with LangSmith

You cannot improve what you cannot measure. LangSmith provides datasets, evaluators, and experiment tracking for LLM apps.

**The workflow.**

1. Curate a **dataset** — inputs (and optionally reference outputs) representative of production.
2. Define **evaluators** — functions that grade an output on some criterion.
3. Run your chain against the dataset and record scores.
4. Iterate on the chain and compare experiments.

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def exact_match(run, example):
    return {"key": "exact_match", "score": run.outputs["answer"] == example.outputs["answer"]}

results = evaluate(
    lambda inputs: {"answer": chain.invoke(inputs["question"])},
    data="my-dataset-name",
    evaluators=[exact_match],
    experiment_prefix="rag-v2",
)
```

**Evaluator types.**

- **Heuristic** — exact match, regex, embedding similarity to reference.
- **LLM-as-judge** — a model grades outputs against criteria (helpfulness, faithfulness, safety). Widely used, but calibrate against human labels before trusting it.
- **Human** — annotators grade outputs in the LangSmith UI. The gold standard.
- **Pairwise** — for A/B comparison of two chain versions on the same input.

**RAG-specific metrics** to track: retrieval hit-rate (was the right chunk retrieved?), answer faithfulness (did the answer stick to the retrieved context?), answer relevance (did it address the question?).

---

## 23. Production concerns

Real deployments care about the boring parts. LangChain has answers for most of them.

**Rate limiting.** Every chat model accepts a `rate_limiter` argument built from `InMemoryRateLimiter` — the framework blocks calls to stay under provider quotas.

**Retries.** Wrap any runnable in `.with_retry()`:

```python
robust_chain = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```

**Fallbacks.** If the primary model or route fails, fall through to an alternative:

```python
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])
```

Common uses: fall back from a large model to a smaller one, or from one provider to another during outages.

**Caching.** Enable an in-memory or Redis-backed cache to deduplicate identical requests:

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
set_llm_cache(InMemoryCache())
```

**Deployment with LangServe.** Expose any runnable as a REST API:

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()
add_routes(app, chain, path="/chat")
```

You get REST, streaming, batch, and OpenAPI docs for free.

**Cost tracking.** Every provider's chat-model response includes `usage_metadata` with input, output, and total tokens. Multiply by provider pricing to attribute cost per request.

**Security.**

- Never trust LLM-generated code, SQL, or shell commands without a sandbox.
- Sanitise retrieved content before rendering — RAG can surface prompt-injection payloads sitting in documents.
- Rate-limit and authenticate any tool that touches shared systems.

---

## 24. Common pitfalls

Recurring mistakes worth knowing about before you hit them:

1. **Mixing embedding models across index and query.** Vectors from different models are not comparable. Rebuild the index whenever the embedding model changes.
2. **Chunk size too large.** A 4000-char chunk contains multiple topics — retrieval finds it for anything, and none of it precisely.
3. **`top_k` too high.** More context is not always better; noise degrades answers.
4. **No abstention prompt.** Without an instruction to say "I don't know," models fill gaps with parametric memory and lie confidently.
5. **Blindly trusting LLM-as-judge scores.** Calibrate against a human-labelled slice before treating them as ground truth.
6. **Legacy chains in new code.** `LLMChain`, `RetrievalQA`, `ConversationChain`, `initialize_agent` — all superseded. Use LCEL and LangGraph.
7. **Unbounded conversation history.** Memory grows until you hit the context limit and everything breaks at once. Trim from the start.
8. **Hardcoded API keys.** Use `.env` and add it to `.gitignore`.
9. **No observability.** Chains are opaque; enable LangSmith tracing in dev at minimum.
10. **Streaming lost.** If any step in an LCEL chain buffers (e.g. a parser that needs the full output), streaming stops working end-to-end. Use `JsonOutputParser` variants that support partial parsing.

---

## 25. Further reading

**Official documentation**
- LangChain: https://python.langchain.com
- LangGraph: https://langchain-ai.github.io/langgraph
- LangSmith: https://docs.smith.langchain.com

**Foundational papers**
- **RAG** — Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)
- **ReAct** — Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
- **HyDE** — Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022)
- **Self-RAG** — Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023)
- **Corrective RAG** — Yan et al., *Corrective Retrieval Augmented Generation* (2024)

**Recommended path if you are starting today**

1. Build a linear LCEL chain (`prompt | llm | parser`) and read every step's output.
2. Add a retriever and turn it into a minimal RAG chain.
3. Wrap it in `RunnableWithMessageHistory` for multi-turn.
4. Add one tool and let the model call it via `bind_tools`.
5. Convert to `create_react_agent` and observe the difference.
6. Introduce LangSmith tracing and, once you have a dataset, evaluation.
7. Model your first cyclical workflow explicitly in LangGraph.

By the time you are comfortable with all seven, you are past "tutorial" and into the working knowledge that production systems rely on.