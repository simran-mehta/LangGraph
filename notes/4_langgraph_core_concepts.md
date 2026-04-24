# LangGraph Core Concepts

> **Source:** Video 4 — Agentic AI using LangGraph (by Nitesh)  
> **Topic:** Conceptual overview of LangGraph's architecture, state management, and LLM workflow patterns

---

## What is LangGraph?

LangGraph is an **orchestration framework** that represents and executes complex, stateful, multi-step LLM workflows as **graphs**.

| Component | Role |
|-----------|------|
| **Nodes** | Individual tasks in the workflow (Python functions) |
| **Edges** | Define execution order and conditions between nodes |

### Key Capabilities
- **Parallel execution** — multiple nodes run simultaneously
- **Loops / cycles** — repeat steps until a condition is met
- **Conditional branching** — decision-making within workflows
- **Memory & resumability** — pause and restart from a checkpoint

> Ideal for **agentic and production-grade AI applications**.

---

## LLM Workflows

An **LLM workflow** is an ordered series of tasks (many LLM-powered) that together achieve a goal.

Each step can involve: prompting, reasoning, tool calling, memory access, or decision-making.

### Workflow Types
| Type | Description |
|------|-------------|
| Linear | Steps run one after another |
| Parallel | Steps run simultaneously |
| Branched | Execution path chosen by conditions |
| Looped | Steps repeat based on feedback |

---

## Common LLM Workflow Patterns

### 1. Prompt Chaining
Sequential LLM calls that break a complex task into subtasks.

```
Topic → Generate Outline → Write Detailed Report
         [validate word count]
```

- Each step feeds into the next
- Validation checks between steps keep quality in check

---

### 2. Routing
The system dynamically decides **which LLM or handler** gets a query based on its nature.

```
User Query → Router → [Refund LLM | Technical LLM | Sales LLM]
```

- Example: Customer support chatbot

---

### 3. Parallelisation
A task is split into **independent subtasks** that run concurrently, then results are aggregated.

```
Video → [Community Guidelines Check]
      → [Misinformation Check      ] → Aggregate → Decision
      → [Inappropriate Content Check]
```

- Example: YouTube content moderation

---

### 4. Orchestrator-Worker
Like parallelisation, but subtasks are **dynamically assigned** based on the input — not known upfront.

```
Research Query → Orchestrator → [Google Scholar | Google News | ...]
                                      ↓
                               Aggregate Results
```

- Example: Research assistant querying multiple platforms

---

### 5. Evaluator-Optimizer
Iterative loop: a **generator** produces output, an **evaluator** scores it, and feedback drives revision until the output meets criteria.

```
Generator → Draft
               ↓
          Evaluator → [Pass] → Final Output
               ↓
          [Fail + Feedback] → Generator (loop)
```

- Example: Drafting emails or blog posts with quality checks

---

## Graphs, Nodes, and Edges

| Concept | Description |
|---------|-------------|
| **Graph** | Collection of interconnected Python functions |
| **Node** | A single task — one Python function |
| **Sequential Edge** | A → B, run in order |
| **Parallel Edge** | A → B and A → C simultaneously |
| **Conditional Edge** | Route to B or C based on a condition |
| **Loop Edge** | Cycle back to a previous node |

---

## State: Shared Mutable Memory

**State** is the central data store shared across all nodes throughout the workflow execution.

- Holds all inputs, intermediate results, and outputs
- Mutable — nodes can read and update it
- Implemented as a **TypedDict** (Python typed dictionary class)

### Example — Chatbot State
```python
from typing import TypedDict, List

class ChatState(TypedDict):
    messages: List[str]
    user_input: str
    response: str
```

Every node receives the current state, performs its task, and returns an updated version.

---

## Reducers: Controlling State Updates

Reducers define **how a node's state update is applied** to the existing state.

| Reducer Behavior | Effect |
|-----------------|--------|
| **Replace** | Overwrites the existing value |
| **Add / Append** | Appends new value (e.g., chat history) |
| **Merge** | Combines old and new data |

Each key in the state can have its own reducer logic.

### Why Reducers Matter
- In **chatbots**: append new messages instead of replacing history
- In **parallel workflows**: correctly merge concurrent updates without data loss
- In **iterative workflows**: decide whether to preserve or overwrite prior versions

```python
from langgraph.graph import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]  # uses add_messages reducer
```

---

## LangGraph Execution Model (Inspired by Google Pregel)

Execution happens in three phases:

```
1. DEFINITION  →  2. COMPILATION  →  3. EXECUTION
```

| Phase | What Happens |
|-------|-------------|
| **Definition** | Define nodes, edges, and initial state (TypedDict) |
| **Compilation** | Validate graph — catch issues like orphan nodes |
| **Execution** | Start from first node; nodes execute, update state, pass it via edges |

### Execution Flow
- **Message passing** — state updates travel through edges to the next node
- **Supersteps** — a single execution cycle that may include multiple parallel nodes
- Execution ends when no active nodes remain and no messages are in transit
- The framework manages all routing after the first node is invoked

> You only trigger the first node — LangGraph handles everything else automatically.

---

## Quick Reference — Key Terms

| Term | Definition |
|------|-----------|
| **LangGraph** | Orchestration framework for LLM workflows as graphs |
| **Node** | A single task represented as a Python function |
| **Edge** | Connection between nodes (sequential, parallel, conditional, loop) |
| **State** | Shared mutable memory (TypedDict) passed through all nodes |
| **Reducer** | Logic for how state updates are applied (replace / append / merge) |
| **Superstep** | One execution cycle — may involve multiple parallel node runs |
| **Message Passing** | Propagating updated state via edges from one node to the next |
| **Google Pregel** | Large-scale graph processing system that inspired LangGraph's execution model |

---

## Summary

```
LangGraph Workflow
      │
      ├── Nodes (Python functions = tasks)
      ├── Edges (control flow: sequential, parallel, conditional, loops)
      ├── State (TypedDict — shared, mutable data store)
      ├── Reducers (how state updates are applied)
      └── Execution Model (Definition → Compilation → Execution via Pregel)
```

LangGraph turns complex LLM workflows into manageable, inspectable graphs. Understanding **state**, **reducers**, and **edges** is the foundation for building any agentic AI application with LangGraph.

---

*Next: Hands-on coding of LangGraph workflows.*
