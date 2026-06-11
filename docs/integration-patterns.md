# Integration patterns

TokenSentinel instruments LLM clients **at the API call layer**. Whatever framework, host, or orchestrator sits above the LLM client is largely irrelevant: as long as the underlying call goes through `anthropic.Anthropic`, `openai.OpenAI`, `google.genai.Client`, a Vertex client, or a boto3 Bedrock client, Sentinel sees every call and runs every rule.

That single integration point is what lets a five-line install cover MCP hosts, RAG pipelines, and every major orchestration framework. This doc shows the wrap site for each.

> **Where to wrap.** The rule is always the same: find the constructor for the client your framework uses to make the actual HTTP call, and pass that instance to `sentinel.wrap(...)` before handing it to the framework. Do this once, at startup, before any agent runs.

---

## 1. MCP servers

The Model Context Protocol moves tools out of the agent codebase and into external servers. The cost surface this introduces is well-documented:

- A 12-server install in Claude Desktop typically merges to ~58 tools.
- At ~950 tokens per tool definition, that is ~55K tokens of definitions injected on every user turn.
- For a 75K context window, that's 72% of context gone before the user has typed anything.

TokenSentinel covers the two MCP-specific failure modes natively:

| Leak | What it catches | Default thresholds |
|---|---|---|
| `tool_definition_bloat` | A single request ships oversized tool defs (≥30 tools or ≥30KB) | 30 tools / 30KB / 0.85 confidence |
| `tool_loop` (and `retrieval_thrash`) | An agent re-invokes the same MCP tool with near-identical args | min_calls=3, window=60s, cosine=0.70 (TF-IDF) |

Plus the rest of the V0 catalog (`context_bloat`, `embedding_waste`, `model_misroute`, `retry_storm`, `zombie`).

### Verified hosts

| Host | Integration | Notes |
|---|---|---|
| **Claude Desktop** | Custom integration via the Anthropic SDK inside the host runtime | Sentinel wraps the Anthropic client the host uses internally; tool defs from MCP servers appear in `raw_request['tools']` like any other tool block. |
| **Cline** (VS Code) | Wrap the LLM client Cline instantiates | Cline supports Anthropic, OpenAI, OpenRouter — wrap whichever you've configured. |
| **Cursor** | Wrap the LLM client in the model provider config | Same pattern as Cline. |
| **Custom MCP hosts** (Python MCP SDK) | Wrap the LLM client your host uses to call the model | The MCP SDK is transport-only; the LLM call still goes through Anthropic/OpenAI/Bedrock. |

### 5-line install for a custom MCP host

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-mcp-host", mode="alert")
client = sentinel.wrap(anthropic.Anthropic())  # tools-array bloat, loops, etc. all caught here
```

That's it. Whether the MCP server is filesystem, jira, supabase, or your own — every tool definition the host injects into `raw_request['tools']` is visible to `tool_definition_bloat`. Every tool invocation flows back into `tool_calls` and is visible to `tool_loop`.

---

## 2. RAG pipelines

The two highest-signal RAG leaks both fire transparently when you wrap the underlying LLM client:

| Leak | What it catches |
|---|---|
| `embedding_waste` | The same document/query is embedded twice in a session — exact-hash match, 0.99 confidence. |
| `retrieval_thrash` | A retrieval tool is invoked ≥3 times within 120s with mean cosine ≥0.65. Catches the canonical "agent rephrases the same question four times instead of widening the initial query" pattern. |

### LlamaIndex

Wrap the LLM and the embedding model where they're constructed. LlamaIndex routes both through the wrapped clients:

```python
from token_sentinel import Sentinel
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import openai

sentinel = Sentinel(project="rag-pipeline")
shared_client = sentinel.wrap(openai.OpenAI())  # used for both LLM and embeddings

# LlamaIndex accepts a custom client on its OpenAI wrappers (api_client kwarg
# in 0.11+; if your version differs, wrap the underlying provider client).
llm = LlamaOpenAI(model="gpt-5", api_client=shared_client)
embed = OpenAIEmbedding(model="text-embedding-3-small", api_client=shared_client)
```

`embedding_waste` will fire whenever the same chunk is re-embedded; `retrieval_thrash` will fire when a query engine spins on near-duplicate prompts.

### LangChain RAG

```python
from token_sentinel import Sentinel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai

sentinel = Sentinel(project="langchain-rag")
client = sentinel.wrap(openai.OpenAI())

llm = ChatOpenAI(model="gpt-5", client=client)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)
```

When LangChain's `RetrievalQA` or `MultiQueryRetriever` issues retrieval-tool invocations, those flow through `tool_calls` on each `CallRecord` and `retrieval_thrash` evaluates them.

### Haystack

```python
from token_sentinel import Sentinel
from haystack.components.generators.openai import OpenAIGenerator
import openai

sentinel = Sentinel(project="haystack-rag")
client = sentinel.wrap(openai.OpenAI())

generator = OpenAIGenerator(api_client=client, model="gpt-5")
```

Same coverage as LangChain. Haystack's retrievers are local components that don't go through the LLM — Sentinel will still flag any LLM-driven re-retrieval pattern but won't see pure vector-store thrashing that never reaches the model. That's by design: silent vector-store calls are not token waste.

### Custom vector-store-then-LLM patterns

```python
from token_sentinel import Sentinel
import openai

sentinel = Sentinel(project="custom-rag")
client = sentinel.wrap(openai.OpenAI())

# In your RAG loop, just use ``client`` everywhere:
embedding = client.embeddings.create(model="text-embedding-3-small", input=query)
chunks = vector_store.search(embedding.data[0].embedding)
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "Answer using these chunks: " + "\n\n".join(chunks)},
        {"role": "user", "content": query},
    ],
)
```

Wrap the OpenAI client; everything else is transparent. The same pattern works against Anthropic, Bedrock, Vertex, Gemini.

---

## 3. Orchestration frameworks

Each framework owns its own agent abstraction, but all of them ultimately call into a provider client. Wrap that client and Sentinel sees the full traffic.

### LangChain

```python
from token_sentinel import Sentinel
from langchain_anthropic import ChatAnthropic
import anthropic

sentinel = Sentinel(project="langchain-agent")
chat = ChatAnthropic(client=sentinel.wrap(anthropic.Anthropic()), model="claude-sonnet-4-6")
```

For OpenAI, swap `langchain_anthropic.ChatAnthropic` for `langchain_openai.ChatOpenAI` and pass the wrapped `openai.OpenAI()`.

### LangGraph

LangGraph reuses LangChain's chat models, so the wrap pattern is identical — wrap once at the chat-model construction site, then pass the chat model into your graph nodes:

```python
from token_sentinel import Sentinel
from langchain_anthropic import ChatAnthropic
import anthropic

sentinel = Sentinel(project="langgraph-agent")
chat = ChatAnthropic(client=sentinel.wrap(anthropic.Anthropic()), model="claude-sonnet-4-6")
# Pass `chat` into your StateGraph nodes as you would any LangChain chat model.
```

### CrewAI

CrewAI's `Agent` and `Crew` accept an `llm=...` kwarg. Construct the LLM with a wrapped client:

```python
from token_sentinel import Sentinel
from crewai import LLM
import openai

sentinel = Sentinel(project="crewai-agent")
llm = LLM(model="gpt-5", client=sentinel.wrap(openai.OpenAI()))
# Pass `llm` to Agent(llm=llm) / Crew(llm=llm).
```

### AutoGen

AutoGen's `AssistantAgent` takes a `model_client`:

```python
from token_sentinel import Sentinel
from autogen_ext.models.openai import OpenAIChatCompletionClient
import openai

sentinel = Sentinel(project="autogen-agent")
model_client = OpenAIChatCompletionClient(model="gpt-5", client=sentinel.wrap(openai.OpenAI()))
# Pass `model_client` to AssistantAgent(model_client=model_client).
```

### Pydantic AI

Pydantic AI's `Agent` constructor accepts a `model` instance. Build it with a wrapped client:

```python
from token_sentinel import Sentinel
from pydantic_ai.models.openai import OpenAIModel
import openai

sentinel = Sentinel(project="pydantic-ai-agent")
model = OpenAIModel("gpt-5", openai_client=sentinel.wrap(openai.OpenAI()))
# Pass `model` to Agent(model=model).
```

---

## 4. Self-hosted note

TokenSentinel works against any OpenAI-compatible endpoint, including **vLLM, Ollama, text-generation-inference, LM Studio, and LocalAI**. The wrap pattern is identical — point `openai.OpenAI(base_url=...)` at your local server and wrap it:

```python
from token_sentinel import Sentinel
import openai

sentinel = Sentinel(project="self-hosted")
client = sentinel.wrap(openai.OpenAI(base_url="http://localhost:8000/v1", api_key="local"))
```

Leak signals (loops, bloat, thrash, embedding waste, context growth) are real on self-hosted backends — they cost you GPU-minutes instead of dollars. The dollar `estimated_burn` field assumes priced API usage and will be off by a constant factor against self-hosted; treat it as a relative-cost signal, not an invoice.
