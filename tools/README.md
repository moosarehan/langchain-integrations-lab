# LangChain Tools — Complete Guide

## Table of Contents
- [What is a Tool?](#what-is-a-tool)
- [Types of Tools](#types-of-tools)
- [How Tools Fit into the Agent Ecosystem](#how-tools-fit-into-the-agent-ecosystem)
- [Built-in Tools](#built-in-tools)
- [Custom Tools](#custom-tools)
- [Tools are Runnables](#tools-are-runnables)
- [How an LLM "Sees" a Tool](#how-an-llm-sees-a-tool)
- [Ways to Create Custom Tools](#ways-to-create-custom-tools)
  - [1. Using the `@tool` Decorator](#1-using-the-tool-decorator)
  - [2. Using `StructuredTool` & Pydantic](#2-using-structuredtool--pydantic)
  - [3. Using the `BaseTool` Class](#3-using-the-basetool-class)
- [Toolkits](#toolkits)

---

## What is a Tool?

A **tool** is just a Python function (or an API) that is packaged in a way an LLM can understand and call when needed.

LLMs (like GPT) are great at:
- **Reasoning** (thinking)
- **Language generation** (speaking)

But they **can't do things like**:
- Access live data (weather, news)
- Do reliable math
- Call APIs
- Run code
- Interact with a database

This is the core limitation tools solve — an LLM on its own is a closed system that only knows what was in its training data and what's in the current conversation. It cannot reach out into the world. A tool is the bridge that lets it do so: the LLM decides *that* an action is needed and *what* arguments to use, and the tool (a regular function) actually executes it and returns a result back to the LLM.

Conceptually, a tool is a function wrapped with metadata (name, description, input schema) so the LLM can:
1. Know the tool **exists**
2. Know **when** to use it
3. Know **what arguments** to pass to it

---

## Types of Tools

```
                Tools
               /     \
    Built-in Tools   Custom Tools
```

LangChain tools fall into two broad categories:
- **Built-in Tools** — already implemented by LangChain, ready to import and use.
- **Custom Tools** — tools you define yourself for your own use case.

---

## How Tools Fit into the Agent Ecosystem

> An **AI agent** is an LLM-powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal.

```
┌─────────────────────────── Agent ────────────────────────────┐
│  ┌───────────────────────┐        ┌────────────────────────┐ │
│  │  Reasoning & Decision  │        │        Action          │ │
│  │        Making          │        │                        │ │
│  └───────────────────────┘        └────────────────────────┘ │
│              LLM                            Tools             │
└─────────────────────────────────────────────────────────────┘
```

An agent is essentially an LLM sitting in a loop with access to tools:
1. The LLM **reasons** about the user's goal and decides what to do next.
2. If it needs external information or a capability it doesn't have, it picks a **tool** and generates the arguments for it.
3. The **tool executes** (the "Action" side) and returns a result.
4. The LLM **observes** that result and reasons again — continuing until it can give a final answer.

Tools are what turn a plain LLM (which can only think and speak) into an agent (which can also *act*).

---

## Built-in Tools

A **built-in tool** is a tool that LangChain already provides for you — it's pre-built, production-ready, and requires minimal or no setup. You don't have to write the function logic yourself — you just import and use it.

| Tool | Description |
|---|---|
| `DuckDuckGoSearchRun` | Web search via DuckDuckGo |
| `WikipediaQueryRun` | Wikipedia summary |
| `PythonREPLTool` | Run raw Python code |
| `ShellTool` | Run shell commands |
| `RequestsGetTool` | Make HTTP GET requests |
| `GmailSendMessageTool` | Send emails via Gmail |
| `SlackSendMessageTool` | Post message to Slack |
| `SQLDatabaseQueryTool` | Run SQL queries |

These are ideal when your need is generic and common enough that someone else has already solved it — web search, running code, hitting a REST endpoint, and so on.

---

## Custom Tools

A **custom tool** is a tool that *you* define yourself.

Use custom tools when:
- You want to call **your own APIs**
- You want to **encapsulate business logic**
- You want the LLM to interact with **your database, product, or app**

Basically, the moment your requirement is specific to your own system (not a generic capability LangChain already ships), you write a custom tool.

---

## Tools are Runnables

An important detail that's easy to miss: **every LangChain tool is also a `Runnable`**.

This means tools follow the same standard interface as every other LangChain component (prompts, LLMs, chains, retrievers, output parsers). Concretely, that gives you:
- A consistent `.invoke()` / `.ainvoke()` / `.batch()` / `.stream()` interface, just like an LLM or a chain.
- The ability to **compose tools inside LCEL pipelines** using the same `|` (pipe) syntax you'd use to chain a prompt into an LLM.
- Built-in support for things like retries, fallbacks, and async execution, since those are all `Runnable`-level features, not tool-specific ones.

In short — a tool isn't a special, separate concept bolted onto LangChain. It's a `Runnable` with extra metadata attached (a name, a description, and an input schema) so an LLM can decide when and how to call it.

---

## How an LLM "Sees" a Tool

An LLM never sees your actual Python function — it can't execute code directly. Instead, when you bind a tool to an LLM (e.g. `llm.bind_tools([my_tool])`), LangChain converts the tool into a **JSON Schema** description and sends *that* to the model as part of the request.

That JSON schema typically looks like this:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The name of the city to get weather for"
      }
    },
    "required": ["city"]
  }
}
```

Key fields the LLM relies on:
- **`name`** — how the model refers to the tool when it wants to call it.
- **`description`** — this is the single most important field. It's the *only* thing the model uses to decide *when* this tool is relevant. A vague description = a tool the model either never calls or calls incorrectly.
- **`properties`** — the arguments the tool accepts, each with its own type and description, so the model knows what to fill in and why.
- **`required`** — which arguments must be provided; the model won't consider the call valid without them.

So when the LLM "chooses to use a tool," what's actually happening is: the model receives this schema as context, matches it against the user's request, and — instead of replying with plain text — outputs a structured JSON object (name + arguments) that matches this schema. LangChain then intercepts that structured output, runs your actual Python function with those arguments, and feeds the result back into the conversation.

This is precisely why writing good names, descriptions, and typed parameters matters so much — it's the entire interface the LLM has into your code.

---

## Ways to Create Custom Tools

```
                          Ways to create Tools
                 /                 |                  \
     using @tool decorator   using StructuredTool    Using BaseTool
                                  & Pydantic              class
```

LangChain gives you three levels of control, from simplest to most powerful.

### 1. Using the `@tool` Decorator

This is the fastest, most common way to turn a plain Python function into a tool.

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b
```

How it works:
- The **type hints** (`a: int, b: int`) tell LangChain what input schema to generate — it infers the JSON schema `properties` directly from your function signature.
- The **return type hint** (`-> int`) tells LangChain (and downstream code) what kind of value to expect back.
- The **docstring** becomes the tool's `description` — this is what the LLM reads to decide when to call it, so it needs to be clear and specific, not just a one-word label.

This approach is great for quick, simple tools where the inputs are a handful of primitive types. Its main limitation: schema inference from type hints is convenient but implicit — for complex inputs (nested objects, custom validation, optional fields with defaults, field-level descriptions) it gets clunky fast.

### 2. Using `StructuredTool` & Pydantic

This approach uses a **Pydantic model** to explicitly define the tool's input schema, instead of relying on inferred type hints.

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number")
    b: int = Field(description="The second number")

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers together.",
    args_schema=MultiplyInput
)
```

Why this is **more strict** than the `@tool` decorator:
- Pydantic **validates** every input at runtime — wrong types, missing required fields, or out-of-range values are rejected *before* your function ever executes, with clear validation errors.
- You get **field-level descriptions** (via `Field(description=...)`), not just one description for the whole tool — this gives the LLM much finer-grained guidance on what each individual argument means.
- It supports the **full range of Pydantic features**: default values, optional fields, nested models, custom validators, constrained types (e.g. `conint(gt=0)`) — none of which you can express cleanly through bare type hints.
- The schema is **explicit and self-documenting**, decoupled from the function signature — useful when the function you're wrapping wasn't written with LangChain in mind (e.g. wrapping an existing internal utility).

In short: `@tool` infers a schema *for* you; `StructuredTool` + Pydantic has you *declare* the schema, which trades a little extra boilerplate for correctness, validation, and clarity — important once a tool's inputs get more complex than 1–2 simple arguments.

### 3. Using the `BaseTool` Class

Every tool in LangChain — whether created via `@tool`, `StructuredTool`, or a built-in tool like `WikipediaQueryRun` — ultimately **inherits from `BaseTool`**. It's the foundational abstract class that defines what a "tool" is in LangChain.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number")
    b: int = Field(description="The second number")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers together."
    args_schema: type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

    async def _arun(self, a: int, b: int) -> int:
        # optional: true async implementation instead of falling back to _run
        return a * b
```

Subclassing `BaseTool` directly is the most powerful — and most manual — of the three approaches. Its advantages:

- **Full control over execution behavior**, including implementing a real `_arun` for native async execution (rather than LangChain running your sync `_run` in a thread pool as a fallback), which matters for I/O-heavy tools (API calls, DB queries) at scale.
- **Access to lifecycle hooks and callbacks** — `BaseTool` integrates directly with LangChain's callback system (`run_manager`, `on_tool_start`, `on_tool_error`, etc.), so you can add logging, tracing, custom error handling, or side effects at a level the decorator-based approaches don't expose as directly.
- **Custom state or dependencies on the tool itself** — since it's a full class, you can give the tool its own `__init__`, store a database connection, an API client, configuration, or any stateful object as instance attributes, and reference them inside `_run`/`_arun`. This is awkward to do cleanly with a plain decorated function.
- **Fine-grained control over serialization and validation** — you can override how the tool reports its schema, add custom validation logic beyond what Pydantic's `args_schema` gives you, or customize error messages returned to the LLM when a call fails.
- **Consistency for tool libraries** — if you're building a whole package/toolkit of related tools (see below), subclassing `BaseTool` gives you a common, explicit interface to enforce across all of them, which is exactly how LangChain implements its own built-in tools internally.

The trade-off is verbosity: for a simple two-argument function, this is significant boilerplate compared to `@tool`. It earns its keep when a tool needs real internal state, custom async behavior, or deep integration with callbacks/observability — not for a quick utility function.

**Summary — which to pick:**

| Approach | Best for |
|---|---|
| `@tool` decorator | Quick, simple tools with primitive arguments |
| `StructuredTool` + Pydantic | Tools needing strict validation, rich/nested inputs, field-level descriptions |
| `BaseTool` subclass | Tools needing custom async logic, internal state, callbacks/observability, or built as part of a larger toolkit |

---

## Toolkits

A **toolkit** is just a collection (bundle) of related tools that serve a common purpose — packaged together for convenience and reusability.

Instead of importing and configuring five separate tools one by one every time you need them together, a toolkit gives you one object that bundles all of them, pre-wired and ready to use.

**Example — In LangChain:**

A toolkit might be: `GoogleDriveToolKit`, and it can contain the following tools:
- `GoogleDriveCreateFileTool` — Upload a file
- `GoogleDriveSearchTool` — Search for a file by name/content
- `GoogleDriveReadFileTool` — Read contents of a file

Instead of instantiating and managing each of these tools separately (each needing its own auth/config), you instantiate the toolkit once, and it hands you back the full list of ready-to-use tools:

```python
toolkit = GoogleDriveToolKit(credentials=my_creds)
tools = toolkit.get_tools()
# [GoogleDriveCreateFileTool(...), GoogleDriveSearchTool(...), GoogleDriveReadFileTool(...)]

agent = create_agent(llm, tools)
```

Toolkits are the natural next step once you have multiple tools that are always used *together* against the same underlying system (Google Drive, Slack, a SQL database, GitHub, etc.) — they keep related tools organized, share common configuration/auth, and save you from re-wiring the same set of tools in every project.