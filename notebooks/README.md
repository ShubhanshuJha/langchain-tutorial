# Notebooks — Hands-on LangChain with a Local Ollama Model

This directory contains the runnable, hands-on companion to the concept guide in the [repository-root README](../README.md). Each notebook builds on the previous one, moving from *"call an LLM once"* to *"drive an agent with tools and give it a memory."* All examples run against a **local model served by [Ollama](https://ollama.com)** — no cloud API keys are required to work through them.

---

## Table of contents

1. [What you will build](#1-what-you-will-build)
2. [Prerequisites](#2-prerequisites)
3. [Project layout](#3-project-layout)
4. [Shared configuration — `config.yaml` and `config_loader.py`](#4-shared-configuration--configyaml-and-config_loaderpy)
5. [Notebook `first.ipynb` — LLMs, Prompts, Chains, Sequential Chains](#5-notebook-firstipynb--llms-prompts-chains-sequential-chains)
6. [Script `second.py` — Streamlit UI on top of a Sequential Chain](#6-script-secondpy--streamlit-ui-on-top-of-a-sequential-chain)
7. [Notebook `third.ipynb` — Tools, Agents, Structured Output, Self-Verification](#7-notebook-thirdipynb--tools-agents-structured-output-self-verification)
8. [Notebook `fourth.ipynb` — Memory: Buffer, Window, and ConversationChain](#8-notebook-fourthipynb--memory-buffer-window-and-conversationchain)
9. [Concepts glossary](#9-concepts-glossary)
10. [Common issues](#10-common-issues)
11. [Where to go next](#11-where-to-go-next)

---

## 1. What you will build

Four progressively richer LLM applications, each isolating one core LangChain idea:

| # | File | Idea in one line |
|---|---|---|
| 1 | `first.ipynb` | Wrap a local LLM in a `PromptTemplate`, chain two prompts together with `SimpleSequentialChain` / `SequentialChain`. |
| 2 | `second.py` | The same chain, but exposed through a Streamlit UI — a "restaurant name & menu generator." |
| 3 | `third.ipynb` | Turn the LLM into a *tool-using agent* (calculator, datetime, DuckDuckGo web search) and add a *self-rectifying* verifier using structured output. |
| 4 | `fourth.ipynb` | Give the model a memory — `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ConversationChain` — and see why unbounded memory is dangerous. |

Together they cover the seven concepts that show up in every LangChain project: **models, prompts, chains, sequential chains, tools, agents, and memory.**

---

## 2. Prerequisites

**Python + packages.** From the repository root:

```powershell
pip install -r ..\requirements.txt
```

Key packages the notebooks import: `langchain-core`, `langchain-classic`, `langchain-community`, `langchain-ollama`, `langgraph`, `pydantic`, `pyyaml`, `python-dotenv`, `numexpr`, `pytz`, `duckduckgo-search`, `streamlit`, `ipykernel`.

**Ollama.** Install from https://ollama.com/download, then pull the two models used across the notebooks:

```powershell
# Used by first.ipynb and second.py — no tool calling needed
ollama pull gemma2:9b

# Used by third.ipynb and fourth.ipynb — needs native tool-calling support
ollama pull qwen3:8b
```

**Why two models?** `gemma2:9b` is fine for plain generation but does **not** expose the `tools` capability that agents require. Anything that calls `create_agent`, `initialize_agent`, or `bind_tools` needs a tool-capable model such as `qwen3`, `llama3.1`, `mistral`, or `command-r`. Verify with:

```powershell
ollama show qwen3:8b
```

and confirm that `tools` appears under **Capabilities**.

**Keep Ollama running.** The Ollama daemon (default `http://localhost:11434`) must be running while any notebook cell executes. `ChatOllama` talks to that endpoint over HTTP.

**Environment variables (optional).** `first.ipynb` calls `load_dotenv()` at the top. If you place a `.env` file next to the notebooks it will be loaded automatically. Nothing here requires an OpenAI key; the loader is there so you can swap `ChatOllama` for `ChatOpenAI` without touching import order.

> **Security note.** Never hardcode API keys in tracked files. `.env`, `secret_keys.py`, and any other file that could contain credentials are excluded via the top-level `.gitignore`. If you accidentally commit a real key, rotate it immediately at the provider — deleting the file from the working tree does *not* invalidate a key that already exists in git history.

---

## 3. Project layout

```
notebooks/
├── README.md              # this file
├── config.yaml            # runtime knobs for the LLM (model, temperature, ...)
├── config_loader.py       # typed loader for config.yaml
├── first.ipynb            # models, prompts, LLMChain, SequentialChain
├── second.py              # Streamlit app wrapping the sequential chain
├── third.ipynb            # tools + agents + structured-output verifier
└── fourth.ipynb           # memory patterns (buffer, window, ConversationChain)
```

> Local-only files such as `.env` and `secret_keys.py` are intentionally ignored by the repo's `.gitignore`. Create them yourself when needed and keep them out of commits.

---

## 4. Shared configuration — `config.yaml` and `config_loader.py`

Every notebook builds its LLM from the same YAML file, so tuning happens in one place.

**`config.yaml`** — the runtime knobs:

```yaml
llm:
  backend: ollama
  model: gemma2:9b
  temperature: 0.4
  max_tokens: 1024
  num_ctx: 8192      # context-window size the model is asked to allocate
  reasoning: false   # disable Qwen3's "thinking" step for faster, more predictable output
  keep_alive: 30m    # how long Ollama keeps the model resident in RAM after last use
```

**`config_loader.py`** — a thin, typed wrapper:

- `LLMConfig` / `AppConfig` are frozen `@dataclass` types, so the loaded config is immutable and IDE-completable.
- `load_config()` reads the YAML and returns an `AppConfig`.
- `load_config_as_dict()` — the function every notebook actually uses — returns the same values as a plain dict, which is the shape `ChatOllama(**cfg)` expects.

**Why this abstraction exists.** Every notebook needs the same 6-line block that constructs a `ChatOllama`. Centralising the values in YAML means switching models (say from `gemma2:9b` to `qwen3:8b`) or lowering `temperature` for a demo is a one-line edit, not a hunt through four notebooks.

**Field cheatsheet.**

| Field | Meaning |
|---|---|
| `backend` | Tag only — the loader does not switch on it. Useful when you later fork the loader for `ChatOpenAI` etc. |
| `model` | The Ollama model tag. Must already be pulled locally. |
| `temperature` | Sampling randomness. `0` = deterministic, `0.7+` = creative. |
| `max_tokens` | Cap on generated tokens. Passed to Ollama as `num_predict`. |
| `num_ctx` | Context window Ollama allocates. Higher = more history fits, more VRAM used. |
| `reasoning` | Toggles Qwen3-family "thinking" tokens. Turn off for faster, cheaper responses. |
| `keep_alive` | How long Ollama keeps the model in memory after a request. Set `-1` to pin forever. |

---

## 5. Notebook `first.ipynb` — LLMs, Prompts, Chains, Sequential Chains

This is the "hello world" of the whole tutorial. Read the cells top to bottom.

### 5.1 Instantiating a chat model

```python
from langchain_ollama import ChatOllama
from config_loader import load_config_as_dict

cfg = load_config_as_dict()['llm']
llm_model = ChatOllama(
    model=cfg['model'],
    temperature=cfg['temperature'],
    num_predict=cfg['max_tokens'],   # Ollama's name for max_tokens
)
```

`ChatOllama` is LangChain's `BaseChatModel` implementation for the Ollama backend. Every chat model in LangChain — `ChatOpenAI`, `ChatAnthropic`, `ChatOllama`, ... — implements the same `.invoke() / .stream() / .batch()` interface. That uniformity is what makes provider swaps painless.

**Note:** `num_predict` is the Ollama-native name for what OpenAI calls `max_tokens`. A common source of confusion when moving code between providers.

### 5.2 Rendering the output nicely

```python
from IPython.display import display, Markdown

def llm_output_parser(output, text=False):
    return Markdown(output.text if text else output.content)
```

`llm_model.invoke(query)` returns an `AIMessage` — the payload lives in `.content` (or `.text` for the newer property). This tiny helper renders the model's markdown-formatted answer inline in the notebook. It is not a LangChain `OutputParser` — for that, see [Concepts glossary](#9-concepts-glossary).

### 5.3 `PromptTemplate` — parameterising a prompt

```python
from langchain_classic.prompts import PromptTemplate

query = "I want to open a restaurant for {cuisine} food. Suggest a fancy name. **SHARE JUST THE NAMES**, nothing else."
prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template=query,
)
prompt_template_name.format(cuisine='Indian')
```

A `PromptTemplate` separates the *shape* of a prompt from the *data* that flows through it. You declare the placeholders, then `.format(...)` fills them safely. This is what lets you reuse a single prompt across hundreds of cuisines, users, documents — anything.

### 5.4 `LLMChain` — one prompt, one model, one output

```python
from langchain_classic.chains.llm import LLMChain

llm_chain = LLMChain(llm=llm_model, prompt=prompt_template_name)
llm_chain.invoke('American')
```

`LLMChain` glues a prompt to a model. Calling `.invoke(x)` fills the template with `x`, sends it to the model, and returns the response.

> **Deprecation note.** `LLMChain` is legacy. Modern LangChain writes the same thing as `prompt | llm | StrOutputParser()` using LCEL (the pipe operator). The notebook uses `LLMChain` because that's still what most existing tutorials and codebases contain — knowing it lets you read them.

### 5.5 `SimpleSequentialChain` — output of A feeds input of B

```python
from langchain_classic.chains.sequential import SimpleSequentialChain

chain = SimpleSequentialChain(chains=[rest_name_chain, food_items_chain])
chain.invoke("India")
```

`SimpleSequentialChain` requires each step to have **exactly one input and one output**. The first chain generates a restaurant name; the second generates a menu from that name. Simple, restrictive, and hard to introspect — you get only the final answer.

### 5.6 `SequentialChain` — multiple inputs/outputs, named keys

```python
from langchain_classic.chains.sequential import SequentialChain

rest_name_chain = LLMChain(..., output_key='restaurant_name')
food_items_chain = LLMChain(..., output_key='menu_items')

chain = SequentialChain(
    chains=[rest_name_chain, food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name', 'menu_items'],
)
chain.invoke({'cuisine': 'Arabic'})
# {'cuisine': 'Arabic', 'restaurant_name': '...', 'menu_items': '...'}
```

`SequentialChain` is the grown-up version. Each step names its output with `output_key`; downstream steps can reference any earlier output as a placeholder, and you decide which of them come back to the caller.

**Takeaway.** After this notebook you understand: model → prompt → chain → sequential-chain — the linear pipeline that most non-agent LangChain apps still boil down to.

---

## 6. Script `second.py` — Streamlit UI on top of a Sequential Chain

Same chain as `first.ipynb`, but wired to a small interactive UI so a non-technical user can drive it. Run it with:

```powershell
streamlit run notebooks/second.py
```

### What's new here

- **Lazy initialisation.** `llm_model` is a module-level global; `llm_initializer()` only builds it on first use. Streamlit re-runs the entire script on every user interaction, so eager global construction would rebuild the model on every keystroke.
- **Two inputs, two outputs.** The chain now takes both `cuisine` and `food_preference` (Veg / Non-Veg / Vegan / Anything) — a real `SequentialChain` job, not a `SimpleSequentialChain` one.
- **Streamlit primitives used:**
  - `st.title(...)` — page header.
  - `st.sidebar.selectbox(...)` — dropdowns rendered in the left sidebar.
  - `st.header(...)` / `st.write(...)` — dynamic content in the main pane.

### The chain, in one glance

```
{cuisine, food_preference}
        │
        ▼
rest_name_chain  ──▶ restaurant_name
        │                │
        ▼                │
food_items_chain  ◀──────┘  (uses restaurant_name + food_preference)
        │
        ▼
     menu_items
```

The Streamlit script owns the *presentation*; the chain owns the *logic*. Keeping the two apart means the same chain works equally well behind a REST endpoint (via LangServe) or a CLI.

---

## 7. Notebook `third.ipynb` — Tools, Agents, Structured Output, Self-Verification

This is where the LLM stops being a text generator and starts being an *actor* that can call functions.

### 7.1 Model swap — why `qwen3:8b`?

The first cell of the notebook says it plainly:

> `create_agent` (and any tool-using agent) requires the underlying model to have **native tool-calling support** — the model must understand a `tools` field in the request and reply with structured `tool_calls`, not just plain text.

Gemma 2 has no tool-call tokens. Trying to use it here raises *"does not support tools"* from Ollama. Confirmed working locally: **Llama 3.1+, Qwen 2.5+, Qwen3, Mistral, Command-R.**

The notebook overrides `cfg['model']` in code so you don't have to edit `config.yaml` while switching.

### 7.2 Building a custom tool with `@tool`

```python
from langchain_core.tools import tool
import numexpr

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression, e.g. "2 + 2" or "3**2"."""
    result = numexpr.evaluate(expression).item()
    return str(result)
```

Three things happen automatically when the decorator sees this function:

1. The **signature** (`expression: str`) becomes a JSON schema the model can read.
2. The **docstring** becomes the tool's description — this is what the model reads to decide *when* to invoke the tool. Docstrings are not decoration; they are the API contract with the LLM.
3. A `Tool` object is produced that the framework can bind to a model.

The second custom tool, `current_datetime(region: str)`, wraps `pytz` + `datetime.now(...)` so the model can ask for accurate wall-clock time — something an offline LLM can never know from parametric memory.

### 7.3 Community tools via `load_tools`

```python
from langchain_community.agent_toolkits.load_tools import load_tools

llm_tools = load_tools(['ddg-search'], llm=llm_model)
llm_tools += [calculator, current_datetime]
```

`load_tools` is a factory for community-maintained tools. `'ddg-search'` gives the model DuckDuckGo web search — no API key needed. You can mix loaded tools and custom `@tool`-decorated functions freely.

### 7.4 The legacy agent — `initialize_agent`

```python
from langchain_classic.agents import AgentType, initialize_agent

llm_agent = initialize_agent(
    llm_tools,
    llm_model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
llm_agent.invoke({"input": [{"role": "user", "content": query}]})
```

An **agent** is a loop:

```
model chooses a tool → tool runs → result feeds back → model chooses again → ... → model answers
```

`CHAT_ZERO_SHOT_REACT_DESCRIPTION` uses the **ReAct** pattern (Reason + Act): the model emits `Thought → Action → Action Input`, LangChain parses that, executes the action, and appends `Observation` back into the prompt. With `verbose=True` you see the whole trace.

> **Deprecation note.** `initialize_agent` prints a `LangChainDeprecationWarning` telling you to migrate to `langchain.agents.create_agent`. Both forms are shown in the notebook so you can read old code and write new code.

### 7.5 The modern agent — `create_agent` with a dynamic system prompt

```python
def build_system_prompt(tools):
    tool_lines = "\n".join(f"- {t.name}: {t.description.strip().splitlines()[0]}" for t in tools)
    return f"You are a helpful assistant with access to the following tools:\n{tool_lines}\n\nGuidelines: ..."

from langchain.agents import create_agent
llm_agent = create_agent(tools=llm_tools, model=llm_model, system_prompt=build_system_prompt(llm_tools))
```

`build_system_prompt` renders the system prompt from whatever tools are currently registered. Adding a new `@tool`-decorated function later automatically updates the prompt — no hand-editing of a hardcoded tool list.

The returned agent is a compiled LangGraph state graph. Its `.invoke(...)` expects `{"messages": [...]}`, and it returns a dict whose `messages` field contains the full trace: `HumanMessage`, `AIMessage` (with `tool_calls`), `ToolMessage`, and finally the model's plain-text answer.

### 7.6 Structured output as a *verifier*

```python
class ResponseCheck(BaseModel):
    is_correct: bool = Field(description="True if the response fully answers the query.")
    corrected_response: str = Field(description="Corrected answer if is_correct is False; else unchanged.")
    reasoning: str = Field(description="Brief explanation of what was missing or wrong.")

def rectify_response(query, response, llm):
    structured_llm = llm.with_structured_output(ResponseCheck)
    result = structured_llm.invoke(verification_prompt)
    ...
```

**`with_structured_output(schema)`** is the modern, reliable way to get typed data out of a chat model. Under the hood it uses the provider's native tool-calling machinery to force the reply into the shape of the given Pydantic model. No regex parsing, no format-instruction gymnastics — the return value is a validated `ResponseCheck` instance.

The `rectify_response` helper turns this into a **self-verification loop**: the agent produces an answer, a fresh LLM call grades it against the original query, and if it fails the grader supplies a corrected version. It is a lightweight, single-shot version of the reflection pattern that LangGraph makes multi-step.

### 7.7 Why the notebook ends with "no memory"

The final cell asks *"What does he do right now?"* — a follow-up question with no antecedent. The agent has no memory of the previous turn about Elon Musk, so it correctly complains about the ambiguity. This is the setup for `fourth.ipynb`.

---

## 8. Notebook `fourth.ipynb` — Memory: Buffer, Window, and ConversationChain

An LLM is stateless. Every request is independent unless *you* re-send prior turns. Memory is the mechanism by which LangChain does that automatically.

### 8.1 No memory — the baseline

```python
llm_chain = LLMChain(llm=llm_model, prompt=prompt_template_name, output_key='restaurant_name')
type(llm_chain.memory)   # NoneType
```

The chain calls the model, returns the answer, forgets everything. Try to ask a follow-up and you get gibberish because the model never saw the prior turn.

### 8.2 `ConversationBufferMemory` — remember everything

```python
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
llm_chain_with_memory = LLMChain(
    llm=llm_model,
    prompt=prompt_template_name,
    output_key='restaurant_name',
    memory=memory,
)
```

Now each call also updates a running transcript that is *reinjected* into the next prompt as the `{history}` variable. Ask for `'Indian'` then `'Chinese'`, and by turn 2 the prompt includes:

```
Human: Indian
AI: Aurora Bazaar
```

**The catch.** `ConversationBufferMemory` grows without bound. Every turn adds to the transcript, every turn spends more tokens re-sending it, until eventually you hit the context window and everything breaks at once. Fine for demos; dangerous in production.

### 8.3 `ConversationChain` — a chain with memory baked in

```python
from langchain_classic.chains import ConversationChain
convo_chain = ConversationChain(llm=llm_model)
convo_chain.run("Who won the first cricket world cup?")
convo_chain.run("Who was the captain of the winning team?")
```

`ConversationChain` bundles a default conversation-style prompt template + a `ConversationBufferMemory` + an `LLMChain` into one object. The second question — *"Who was the captain..."* — resolves correctly because the buffer still contains the West Indies context from turn 1.

The default template makes the memory contract explicit:

```
The following is a friendly conversation between a human and an AI. ...
Current conversation:
{history}
Human: {input}
AI:
```

### 8.4 `ConversationBufferWindowMemory` — keep only the last `k` turns

```python
from langchain_classic.memory import ConversationBufferWindowMemory

lim_llm_memory = ConversationBufferWindowMemory(k=1)   # only the last 1 exchange
convo_chain = ConversationChain(llm=llm_model, memory=lim_llm_memory)
```

Same buffer semantics but bounded. Turn 3 asks *"Who was the captain of the winning team?"* — because `k=1` only kept turn 2 (an unrelated math question), the model correctly answers *"I don't have enough information."* Not a bug: it's the trade-off of a small window. Choose `k` big enough to preserve the context users typically rely on, but small enough to keep token cost predictable.

### 8.5 The bigger picture

The notebook prints a `LangChainDeprecationWarning` on every memory class. The modern replacement — introduced in the root README's Section 9 — is `RunnableWithMessageHistory` (short-term) plus the LangGraph checkpointer / `Store` API (long-term). Legacy classes still work and are still the shortest way to *illustrate* the concept, which is why this notebook uses them.

**Two production-grade memory patterns worth knowing after this:**

| Pattern | Use case |
|---|---|
| `RunnableWithMessageHistory` + `trim_messages(max_tokens=...)` | Bounded short-term chat memory in an LCEL chain. |
| `create_react_agent(..., checkpointer=<...>)` | Long-lived, persistent, multi-turn agent state via LangGraph. |

---

## 9. Concepts glossary

Quick reference — each entry maps a term you see in the notebooks to what it actually is.

| Term | What it is | First appears in |
|---|---|---|
| **Chat model** | An object that takes messages, returns a message. Here: `ChatOllama`. | `first.ipynb` §5.1 |
| **`invoke`** | Standard method on every LangChain runnable — one input → one output. | `first.ipynb` §5.1 |
| **`AIMessage`** | The reply object returned by a chat model. `.content` holds the text; `.tool_calls` holds any requested tool invocations. | `first.ipynb` §5.2, `third.ipynb` §7.5 |
| **`PromptTemplate`** | A string with named `{placeholders}` and a `.format()` method — separates prompt shape from data. | `first.ipynb` §5.3 |
| **`LLMChain`** | Legacy: prompt + model glued together. Modern replacement: `prompt \| llm`. | `first.ipynb` §5.4 |
| **`SimpleSequentialChain`** | Two chains in series; single input, single output per step. | `first.ipynb` §5.5 |
| **`SequentialChain`** | Two+ chains in series with named `input_variables` / `output_variables`. | `first.ipynb` §5.6 |
| **Tool (`@tool`)** | A Python function exposed to the model. Signature → JSON schema; docstring → description. | `third.ipynb` §7.2 |
| **`load_tools`** | Factory for community-maintained tools (web search, math, Wikipedia, ...). | `third.ipynb` §7.3 |
| **Agent** | A loop where the model chooses tool calls until it produces a plain answer. ReAct = *Reason + Act*. | `third.ipynb` §7.4 |
| **`create_agent`** | Modern LangGraph-backed agent factory. Successor to `initialize_agent`. | `third.ipynb` §7.5 |
| **`with_structured_output(Schema)`** | Force a Pydantic-typed reply using the model's native tool-calling. Preferred over `PydanticOutputParser`. | `third.ipynb` §7.6 |
| **Memory** | Storing past turns and reinjecting them so a stateless model behaves as if stateful. | `fourth.ipynb` §8.2 |
| **`ConversationBufferMemory`** | Keep the full transcript. Simple; grows without bound. | `fourth.ipynb` §8.2 |
| **`ConversationBufferWindowMemory`** | Keep only the last `k` turns. Bounded cost, bounded recall. | `fourth.ipynb` §8.4 |
| **`ConversationChain`** | Legacy convenience: default prompt + buffer memory + `LLMChain` in one. | `fourth.ipynb` §8.3 |

---

## 10. Common issues

- **`registry.ollama.ai: dial tcp: no such host`** — Ollama daemon is not running. Start it (`ollama serve` or the desktop app).
- **`model 'X' does not support tools`** — you tried to bind tools to a model without tool-calling capability. Switch to `qwen3:8b`, `llama3.1:8b`, `mistral`, or `command-r`.
- **First response takes 30+ seconds, subsequent ones are fast** — Ollama is loading the model from disk. `keep_alive: 30m` in `config.yaml` keeps it resident.
- **Answers cite events after the model's training cut-off** — the model is hallucinating dates. This is *why* `third.ipynb` gives the agent `current_datetime` and `ddg-search`: to ground time-sensitive answers.
- **`ConversationChain` gives wrong follow-up answers** — likely the window / buffer size is too small, or (in `k=1` mode) an unrelated middle turn evicted the context. Increase `k` or switch to a summary-style memory.
- **`langchain_classic` imports fail** — recent LangChain releases moved legacy pieces out of `langchain` into `langchain-classic`. Install it separately: `pip install langchain-classic`.
- **`DuckDuckGoSearch` returns empty results** — DDG rate-limits aggressive callers. Retry after a minute; consider replacing with a proper search API in production.

---

## 11. Where to go next

Once the four notebooks make sense end-to-end:

1. Rewrite `first.ipynb` in LCEL: `prompt | llm | StrOutputParser()`.
2. Replace `SequentialChain` in `second.py` with two LCEL steps composed by `RunnableParallel` — the same shape, streaming for free.
3. Replace `ConversationBufferMemory` in `fourth.ipynb` with `RunnableWithMessageHistory` + `trim_messages(max_tokens=1000)`.
4. Turn `third.ipynb`'s agent into an explicit LangGraph state machine — add a `verify` node that runs `rectify_response` before the final message.
5. Wire the whole thing behind LangServe and expose it as a REST API.

Each step maps to a section in the [root README](../README.md), which is the *concept* reference to this notebook folder's *practice*.