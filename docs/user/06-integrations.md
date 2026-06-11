# Integrations

TokenSentinel instruments LLM clients **at the API call layer**. Whatever framework, host, or orchestrator sits above the LLM client is largely irrelevant: as long as the underlying call goes through `anthropic.Anthropic`, `openai.OpenAI`, `google.genai.Client`, or a boto3 Bedrock client, Sentinel sees every call and runs every rule.

That single integration point is what lets a five-line install cover MCP hosts, RAG pipelines, and every major orchestration framework. This page shows the wrap site for each.

> **The rule is always the same.** Find the constructor for the client your framework uses to make the actual HTTP call, and pass that instance to `sentinel.wrap(...)` *before* handing it to the framework. Do this once, at startup, before any agent runs.

## MCP servers

The Model Context Protocol moves tools out of the agent codebase and into external servers. The cost surface this introduces is well-documented:

- A 12-server install in Claude Desktop typically merges to ~58 tools.
- At ~950 tokens per tool definition, that is ~55K tokens of definitions injected on every user turn.
- For a 75K context window, that's 72% of context gone before the user has typed anything.

TokenSentinel covers the two MCP-specific failure modes natively:

| Leak | What it catches |
|---|---|
| `tool_definition_bloat` | A single request ships oversized tool defs (≥30 tools or ≥30KB). |
| `tool_loop` and `retrieval_thrash` | An agent re-invokes the same MCP tool with near-identical args. |

### 5-line install for an MCP host

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-mcp-host", mode="alert")
client = sentinel.wrap(anthropic.Anthropic())
# Pass `client` to wherever your MCP host needs an Anthropic client.
# tools-array bloat, loops, retrieval thrash all caught.
```

That's it. Whether your MCP server is filesystem, jira, supabase, or your own — every tool definition the host injects into `raw_request['tools']` is visible to `tool_definition_bloat`. Every tool invocation flows back through `tool_calls` and is visible to `tool_loop`.

For Claude Desktop, Cline, or Cursor specifically, the wrap site is whichever Anthropic / OpenAI / OpenRouter client the host instantiates internally — wrap it before the host's agent loop runs. Custom Python MCP hosts (using `mcp`) wrap whatever client their host uses to call the model.

## RAG pipelines

The two highest-signal RAG leaks fire transparently when you wrap the underlying LLM client:

| Leak | What it catches |
|---|---|
| `embedding_waste` | The same document/query is embedded twice in a session. Exact-hash match, 0.99 confidence. |
| `retrieval_thrash` | A retrieval tool is invoked ≥3 times within 120s with mean cosine ≥0.65. Catches "agent rephrases the same question four times instead of widening the initial query". |

### LlamaIndex

```python
from token_sentinel import Sentinel
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import openai

sentinel = Sentinel(project="rag-pipeline")
shared = sentinel.wrap(openai.OpenAI())

llm = LlamaOpenAI(model="gpt-5", api_client=shared)
embed = OpenAIEmbedding(model="text-embedding-3-small", api_client=shared)
```

LlamaIndex routes both LLM and embedding calls through the wrapped client. `embedding_waste` fires on duplicate chunks, `retrieval_thrash` fires when a query engine spins on near-duplicate prompts.

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

`RetrievalQA` and `MultiQueryRetriever` issue retrieval-tool invocations that flow through `tool_calls` on each `CallRecord`, and `retrieval_thrash` evaluates them.

### Haystack

```python
from token_sentinel import Sentinel
from haystack.components.generators.openai import OpenAIGenerator
import openai

sentinel = Sentinel(project="haystack-rag")
client = sentinel.wrap(openai.OpenAI())

generator = OpenAIGenerator(api_client=client, model="gpt-5")
```

Note: Haystack's retrievers are local components that don't go through the LLM, so Sentinel won't see vector-store thrashing that never reaches the model. This is by design — silent vector-store calls are not token waste.

### Custom vector-store-then-LLM patterns

```python
from token_sentinel import Sentinel
import openai

sentinel = Sentinel(project="custom-rag")
client = sentinel.wrap(openai.OpenAI())

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

Wrap the OpenAI client; everything else is transparent. Same pattern works against Anthropic, Bedrock, Vertex, Gemini.

## Orchestration frameworks

Each framework owns its own agent abstraction, but all of them ultimately call into a provider client. Wrap that client and Sentinel sees the full traffic.

### LangChain

```python
from token_sentinel import Sentinel
from langchain_anthropic import ChatAnthropic
import anthropic

sentinel = Sentinel(project="langchain-agent")
chat = ChatAnthropic(client=sentinel.wrap(anthropic.Anthropic()), model="claude-sonnet-4-6")
```

For OpenAI, swap `langchain_anthropic.ChatAnthropic` for `langchain_openai.ChatOpenAI` and pass a wrapped `openai.OpenAI()`.

### LangGraph

LangGraph reuses LangChain's chat models, so the wrap pattern is identical — wrap once at the chat-model construction site, then pass the chat model into your graph nodes:

```python
from token_sentinel import Sentinel
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
import anthropic

sentinel = Sentinel(project="langgraph-agent")
chat = ChatAnthropic(client=sentinel.wrap(anthropic.Anthropic()), model="claude-sonnet-4-6")

# Pass `chat` into your StateGraph nodes as you would any LangChain chat model.
graph = StateGraph(...)
# graph.add_node("planner", lambda state: chat.invoke(state["messages"]))
```

### CrewAI

```python
from token_sentinel import Sentinel
from crewai import LLM, Agent
import openai

sentinel = Sentinel(project="crewai-agent")
llm = LLM(model="gpt-5", client=sentinel.wrap(openai.OpenAI()))

researcher = Agent(role="researcher", goal="...", llm=llm)
```

### AutoGen

```python
from token_sentinel import Sentinel
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import openai

sentinel = Sentinel(project="autogen-agent")
model_client = OpenAIChatCompletionClient(model="gpt-5", client=sentinel.wrap(openai.OpenAI()))

agent = AssistantAgent("assistant", model_client=model_client)
```

### Pydantic AI

```python
from token_sentinel import Sentinel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import openai

sentinel = Sentinel(project="pydantic-ai-agent")
model = OpenAIModel("gpt-5", openai_client=sentinel.wrap(openai.OpenAI()))

agent = Agent(model=model)
```

## Async frameworks

The same wrap pattern works for `AsyncAnthropic`, `AsyncOpenAI`, and the async `client.aio.models.*` surface on Gemini. Sentinel auto-detects async via `inspect.iscoroutinefunction`:

```python
from token_sentinel import Sentinel
import anthropic
import asyncio

sentinel = Sentinel(project="async-agent")
client = sentinel.wrap(anthropic.AsyncAnthropic())

async def run():
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "..."}],
    )
    return response

asyncio.run(run())
```

`Sentinel.record_call` is sync internally — it does in-process work only, so calling it from async code is safe.

## Self-hosted

TokenSentinel works against any OpenAI-compatible endpoint, including vLLM, Ollama, text-generation-inference, LM Studio, and LocalAI. Identical wrap pattern — point `openai.OpenAI(base_url=...)` at your local server:

```python
from token_sentinel import Sentinel
import openai

sentinel = Sentinel(project="self-hosted")
client = sentinel.wrap(openai.OpenAI(base_url="http://localhost:8000/v1", api_key="local"))
```

Leak signals are real on self-hosted backends — they cost you GPU-minutes instead of dollars. The dollar `estimated_burn` field assumes priced API usage and will be off; treat it as a relative-cost signal.

## Multiple agents in one process

You can wrap the same client object with multiple `Sentinel` instances if you want different rule sets / modes per logical agent:

```python
from token_sentinel import Sentinel
import anthropic

raw_client = anthropic.Anthropic()

# Production agent: log only
prod_sentinel = Sentinel(project="prod-agent", mode="log")
prod_client = prod_sentinel.wrap(raw_client)

# Dev agent: block on retry storms
dev_sentinel = Sentinel(project="dev-agent", mode="block", rules=["retry_storm"])
dev_client = dev_sentinel.wrap(anthropic.Anthropic())  # separate raw client recommended
```

Both Sentinel instances run independently — events from `prod_client` only go to `prod_sentinel`'s handlers. We recommend a separate raw client per Sentinel rather than reusing one — wrappers mutate the client in place, and chaining wraps on the same instance has the second wrapper instrumenting the first wrapper's instrumented method.

## Putting it all together

For a typical production stack:

```python
from token_sentinel import Sentinel, LeakDetected
import anthropic
import logging

log = logging.getLogger("tokensentinel")

sentinel = Sentinel(
    project="prod-agent",
    mode="log",                        # graduate to alert/block per [Modes](./03-modes.md)
    min_confidence=0.6,                # silence the lowest-confidence rule firings
    config={
        "tool_loop.cosine_threshold": 0.75,
        "context_bloat.slope_threshold": 2500,
    },
)

@sentinel.on_leak
def on_leak(event):
    log.warning(
        "leak event",
        extra={
            "leak_type": event.type,
            "confidence": event.confidence,
            "session_id": event.session_id,
            "estimated_burn": event.estimated_burn,
            "evidence": event.evidence,
        },
    )

client = sentinel.wrap(anthropic.Anthropic())

# In your agent code, pass a stable session id to group calls:
def run_agent(user_id, task_id, messages):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=messages,
        _sentinel_session_id=f"user-{user_id}-task-{task_id}",
    )
```

That's a fully integrated TokenSentinel deployment. Read [Modes](./03-modes.md) for graduation guidance, [Leak rules](./04-waste-rules.md) for tuning, and [API reference](./07-api-reference.md) for the full surface.
