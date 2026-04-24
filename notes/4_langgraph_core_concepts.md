# LangGraph Core Concepts

> **Goal:** Build a deep, practical understanding of LangGraph for agentic AI development.  
> **Status:** Living document — updated as learning progresses.

---

## Table of Contents
1. [What is LangGraph?](#1-what-is-langgraph)
2. [LLM Workflows](#2-llm-workflows)
3. [Common Workflow Patterns](#3-common-workflow-patterns)
4. [Graphs, Nodes, and Edges](#4-graphs-nodes-and-edges)
5. [State — Shared Mutable Memory](#5-state--shared-mutable-memory)
6. [Reducers — Controlling State Updates](#6-reducers--controlling-state-updates)
7. [Execution Model](#7-execution-model)
8. [Building Your First Graph](#8-building-your-first-graph)
9. [Checkpointing and Memory](#9-checkpointing-and-memory)
10. [Key Terms Glossary](#10-key-terms-glossary)

---

## 1. What is LangGraph?

LangGraph is an **orchestration framework** built on top of LangChain that lets you represent and execute complex, stateful, multi-step LLM workflows as **directed graphs**.

Think of it like this:

```
Traditional code:   function_a() → function_b() → function_c()   (rigid, linear)

LangGraph:          node_a ──► node_b ──► node_c                  (flexible graph)
                               ↑               │
                               └───────────────┘  (can loop, branch, run in parallel)
```

### Why not just write plain Python functions?

| Plain Python | LangGraph |
|---|---|
| Hard to add branching/loops cleanly | Branching is a first-class concept |
| No built-in state management | Shared typed state out of the box |
| No pause/resume | Checkpointing built in |
| Hard to visualize flow | Graph can be rendered visually |
| No human-in-the-loop support | Interrupt and resume supported |

### Core Graph Components

| Component | Role | Python Equivalent |
|---|---|---|
| **Node** | A single unit of work | A Python function |
| **Edge** | Connection defining what runs next | Function call / routing logic |
| **State** | Shared memory across all nodes | A TypedDict passed by reference |

---

## 2. LLM Workflows

An **LLM workflow** is a sequence of coordinated tasks where one or more steps involve an LLM. Most real-world AI applications are workflows, not single prompts.

### Single Prompt vs Workflow

```
Single Prompt:
  User → [LLM] → Answer

Workflow (Hiring Pipeline Example):
  JD Creation → Job Posting → Resume Shortlisting → Interview Scheduling → Offer Letter
       ↓               ↓               ↓                     ↓                  ↓
    [LLM]           [LLM]           [LLM]               [LLM + Tool]         [LLM]
```

Each step can involve:
- **Prompting** — crafting and sending messages to an LLM
- **Reasoning** — LLM deciding what to do next
- **Tool calling** — LLM invoking external functions (search, DB, APIs)
- **Memory access** — reading/writing conversation or long-term memory
- **Decision making** — branching the workflow based on output

### Workflow Shape Types

```
Linear:     A → B → C → D

Parallel:   A → B ─┐
              → C ─┤→ E
              → D ─┘

Branched:   A → [condition] → B
                            → C

Looped:     A → B → [check] → C (done)
                  ↑___(retry)__↘
```

---

## 3. Common Workflow Patterns

### 3.1 Prompt Chaining

Break a complex task into sequential subtasks, each feeding into the next.

```
Topic Input
    │
    ▼
[Node 1: Generate Outline]
    │
    ▼ (validate: outline has ≥ 3 sections?)
[Node 2: Write Section Drafts]
    │
    ▼
[Node 3: Compile Final Report]
    │
    ▼
Output
```

**When to use:** Tasks where later steps depend on the output of earlier steps and intermediate quality checks matter.

---

### 3.2 Routing

The system classifies an input and **routes it to the appropriate handler**.

```
User Query
    │
    ▼
[Router Node — classify intent]
    │
    ├──► "refund"    → [Refund Handler LLM]
    ├──► "technical" → [Technical Support LLM]
    └──► "sales"     → [Sales LLM]
```

**Key insight:** The router can itself be an LLM that classifies the query, or simple rule-based logic. LangGraph's **conditional edges** power this pattern.

---

### 3.3 Parallelisation

Split a task into independent subtasks and run them **simultaneously**, then aggregate.

```
Input
  │
  ├──► [Check: Community Guidelines] ──┐
  ├──► [Check: Misinformation]        ──┼──► [Aggregator Node] → Decision
  └──► [Check: Inappropriate Content] ──┘
```

**When to use:** When subtasks don't depend on each other and latency matters. Running 3 checks in parallel is ~3x faster than sequentially.

---

### 3.4 Orchestrator-Worker

An **orchestrator node** breaks down the task dynamically at runtime, spawns worker nodes, then aggregates results. Unlike parallelisation, the workers are not predefined — they're decided based on the input.

```
Research Query
      │
      ▼
[Orchestrator — plan subtasks]
      │
      ├──► [Worker: Search Google Scholar]   (if academic)
      ├──► [Worker: Search Google News]      (if current events)
      └──► [Worker: Search Internal DB]      (if proprietary data)
                        │
                        ▼
              [Aggregator Node]
                        │
                        ▼
                  Final Answer
```

**Key difference from parallelisation:** Workers are dynamically chosen; the orchestrator decides at runtime what needs to happen.

---

### 3.5 Evaluator-Optimizer (Reflection Loop)

A **generator** produces a draft, an **evaluator** scores it against criteria, and feedback drives revision. Loops until output passes.

```
Input
  │
  ▼
[Generator] ──► Draft
                  │
                  ▼
            [Evaluator]
                  │
         ┌────────┴────────┐
      [Pass]           [Fail + Feedback]
         │                  │
         ▼                  └──► back to [Generator]
    Final Output
```

**Real-world uses:** Writing assistants, code generation with test feedback, automated grading, data validation pipelines.

---

## 4. Graphs, Nodes, and Edges

### Nodes

A node is just a **Python function** that:
1. Receives the current state
2. Does some work (calls LLM, runs logic, calls a tool)
3. Returns a dict of updated state values

```python
def my_node(state: MyState) -> dict:
    result = do_something(state["input"])
    return {"output": result}  # only return what changed
```

### Edge Types

| Edge Type | When to Use | LangGraph API |
|---|---|---|
| **Normal edge** | Always go from A to B | `graph.add_edge("a", "b")` |
| **Conditional edge** | Branch based on state | `graph.add_conditional_edges(...)` |
| **Entry point** | Define the starting node | `graph.set_entry_point("node_name")` |
| **End** | Terminate the graph | `graph.add_edge("node", END)` |

### Conditional Edges — The Routing Mechanism

```python
from langgraph.graph import END

def route(state: MyState) -> str:
    if state["score"] >= 7:
        return "accept"
    else:
        return "revise"

graph.add_conditional_edges(
    "evaluator",          # from this node
    route,                # call this function to decide
    {
        "accept": END,    # if "accept" → end the graph
        "revise": "generator"  # if "revise" → go back to generator
    }
)
```

---

## 5. State — Shared Mutable Memory

State is the **central data structure** that flows through every node. Every node reads from it and writes back to it.

### Defining State with TypedDict

```python
from typing import TypedDict, List

class HiringState(TypedDict):
    job_title: str
    job_description: str
    resumes: List[str]
    shortlisted: List[str]
    interview_notes: str
    offer_sent: bool
```

### How State Flows Through the Graph

```
Initial State: {"job_title": "Backend Engineer", "job_description": "", ...}
       │
       ▼
[Node: generate_jd]  → returns {"job_description": "We are looking for..."}
       │
       ▼
State: {"job_title": "Backend Engineer", "job_description": "We are looking for...", ...}
       │
       ▼
[Node: shortlist_resumes]  → returns {"shortlisted": ["Alice", "Bob"]}
       │
       ▼
State: {..., "shortlisted": ["Alice", "Bob"], ...}
```

**Key rule:** Nodes return only the keys they updated — LangGraph merges the update into the full state.

### State Design Tips

- Keep state flat where possible — nested dicts add complexity
- Use `Optional` for fields that aren't populated until later stages
- Think of state like a shared database row that each node can read/write

```python
from typing import TypedDict, List, Optional, Annotated

class ChatState(TypedDict):
    user_input: str
    messages: List[dict]           # full conversation history
    intent: Optional[str]          # populated by router node
    final_response: Optional[str]  # populated at the end
```

---

## 6. Reducers — Controlling State Updates

By default, when a node returns `{"messages": new_list}`, LangGraph **replaces** the existing `messages` with `new_list`. That would wipe out conversation history — not what you want.

**Reducers** let you define custom merge logic per state key.

### Default Behavior (Replace)

```python
# Node returns {"count": 5}
# Old state: {"count": 3}
# New state: {"count": 5}  ← replaced
```

### Append Behavior with `add_messages`

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # annotate with reducer
```

```python
# Node returns {"messages": [new_message]}
# Old state: {"messages": [msg1, msg2]}
# New state: {"messages": [msg1, msg2, new_message]}  ← appended
```

### Custom Reducer

You can write your own reducer for any key:

```python
def merge_scores(existing: list, new: list) -> list:
    return existing + new  # custom merge logic

class EvalState(TypedDict):
    scores: Annotated[list, merge_scores]
```

### Why Reducers Matter in Parallel Workflows

When two nodes run **in parallel** and both update the same state key, LangGraph needs to know how to combine the two updates. Without a reducer, this is ambiguous. With a reducer (like append), both updates are safely merged.

```
[Node A] → {"results": ["finding_1"]}  ─┐
                                         ├──► reducer merges → {"results": ["finding_1", "finding_2"]}
[Node B] → {"results": ["finding_2"]}  ─┘
```

---

## 7. Execution Model

LangGraph's execution is inspired by **Google Pregel** — a large-scale graph processing framework built for running algorithms on massive graphs at Google.

### Three Phases

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  DEFINITION │ ──► │ COMPILATION │ ──► │  EXECUTION  │
└─────────────┘     └─────────────┘     └─────────────┘
  Add nodes            Validate           Run graph
  Add edges            Check for          Pass state
  Set entry point      orphan nodes       via messages
```

### What Happens During Execution

1. Graph starts at the entry point node with the initial state
2. Node executes → returns partial state update
3. LangGraph applies the update to full state using reducers
4. Edges determine which node(s) run next
5. If multiple edges lead forward → those nodes run as a **superstep** (in parallel)
6. Repeat until `END` is reached or no active nodes remain

### Supersteps

A **superstep** is one round of execution. It may involve:
- A single node (sequential step)
- Multiple nodes running in parallel (parallel step)

```
Superstep 1: [node_a executes]
Superstep 2: [node_b] and [node_c] execute in parallel
Superstep 3: [node_d executes] (aggregates b+c results)
Superstep 4: END
```

---

## 8. Building Your First Graph

A minimal working LangGraph example — a simple prompt chain:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define State
class State(TypedDict):
    topic: str
    outline: str
    report: str

# 2. Define Nodes
def generate_outline(state: State) -> dict:
    # In real usage: call an LLM here
    outline = f"Outline for: {state['topic']}\n1. Intro\n2. Body\n3. Conclusion"
    return {"outline": outline}

def write_report(state: State) -> dict:
    report = f"Report based on outline:\n{state['outline']}\n\n[Full content here...]"
    return {"report": report}

# 3. Build Graph
graph = StateGraph(State)
graph.add_node("generate_outline", generate_outline)
graph.add_node("write_report", write_report)

# 4. Add Edges
graph.set_entry_point("generate_outline")
graph.add_edge("generate_outline", "write_report")
graph.add_edge("write_report", END)

# 5. Compile
app = graph.compile()

# 6. Run
result = app.invoke({"topic": "Quantum Computing", "outline": "", "report": ""})
print(result["report"])
```

### With Conditional Routing

```python
def route_query(state: State) -> str:
    intent = state["intent"]
    if intent == "refund":
        return "refund_handler"
    elif intent == "technical":
        return "tech_handler"
    else:
        return "general_handler"

graph.add_conditional_edges(
    "router",           # from node
    route_query,        # function that returns next node name
    {                   # mapping of return values to node names
        "refund_handler": "refund_handler",
        "tech_handler": "tech_handler",
        "general_handler": "general_handler",
    }
)
```

### With a Loop (Evaluator-Optimizer)

```python
def should_continue(state: State) -> str:
    if state["score"] >= 8:
        return "done"
    elif state["iterations"] >= 3:
        return "done"   # prevent infinite loops
    else:
        return "revise"

graph.add_conditional_edges(
    "evaluator",
    should_continue,
    {"done": END, "revise": "generator"}
)
```

> **Always add a max iteration guard** in loops to prevent infinite execution.

---

## 9. Checkpointing and Memory

One of LangGraph's most powerful features: **checkpointing** saves graph state after every superstep. This enables:

- **Pause and resume** — stop mid-workflow and continue later
- **Human-in-the-loop** — pause for human approval before continuing
- **Fault tolerance** — resume from last checkpoint on failure
- **Time travel** — inspect or replay any past state

### Adding a Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # in-memory (use SqliteSaver for persistence)
app = graph.compile(checkpointer=checkpointer)

# Run with a thread_id — each thread_id is an independent conversation/session
config = {"configurable": {"thread_id": "session_001"}}
result = app.invoke({"topic": "AI"}, config=config)

# Resume the same thread later — state is preserved
result2 = app.invoke({"topic": "AI Ethics"}, config=config)
```

### Human-in-the-Loop with `interrupt_before`

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"]  # pause before this node
)

# Graph runs until it hits "send_email", then pauses
state = app.invoke(initial_state, config)

# Human reviews and approves
# Resume by calling invoke again with same config
final_state = app.invoke(None, config)
```

---

## 10. Key Terms Glossary

| Term | Definition |
|---|---|
| **LangGraph** | Orchestration framework for building stateful, multi-step LLM workflows as directed graphs |
| **Node** | A Python function representing a single task in the workflow |
| **Edge** | A connection between nodes — defines execution flow (sequential, conditional, loop) |
| **Conditional Edge** | An edge whose target is decided at runtime by a routing function |
| **State** | A TypedDict that acts as shared, mutable memory flowing through all nodes |
| **Reducer** | Logic defining how a node's state update is merged into the existing state |
| **add_messages** | Built-in reducer that appends messages instead of replacing them |
| **Superstep** | One round of execution — may involve multiple parallel nodes |
| **Message Passing** | Propagating updated state via edges from one node to the next |
| **Google Pregel** | Distributed graph processing system that inspired LangGraph's execution model |
| **Checkpointer** | Component that saves state after each superstep, enabling pause/resume |
| **Thread ID** | Identifier for an independent execution session (used with checkpointers) |
| **Human-in-the-Loop** | Pattern where graph pauses for human input/approval before continuing |
| **StateGraph** | The main LangGraph class used to construct a graph |
| **END** | Special sentinel node that terminates graph execution |

---

## Mental Model Summary

```
LangGraph Application
│
├── StateGraph
│   ├── State (TypedDict)          ← what data flows through
│   │   └── Reducers per key       ← how updates are merged
│   │
│   ├── Nodes (Python functions)   ← what work gets done
│   │   └── each reads state, returns partial update
│   │
│   └── Edges                      ← how execution flows
│       ├── Normal edges            (always A → B)
│       ├── Conditional edges       (A → B or C based on state)
│       └── END edges               (terminate)
│
├── compile() → app
│   └── optional: checkpointer, interrupt_before/after
│
└── app.invoke(initial_state, config)
    └── Execution: superstep by superstep until END
```

---

*This document is updated as new concepts are learned. Next topics: tool calling in nodes, multi-agent graphs, LangGraph Studio visualization.*
