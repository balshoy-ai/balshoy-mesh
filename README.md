# balshoy-mesh

**Balshoy Mesh** is an event-driven framework for building multi-agent AI systems where specialized models operate as independent microservices. Instead of a monolithic architecture, we use an asynchronous event bus (NATS JetStream) to orchestrate tasks via DAGs (Directed Acyclic Graphs), ensuring scalability and fault tolerance. The system provides an OpenAI-compatible API, hiding the complexity of distributed computing behind a simple chat interface.

**Status:** Experimental project. At this stage, it serves as a proof-of-concept: the API, component composition, and phases are subject to change and are **not ready for production**. Full specification is available in [ARCHITECTURE.md](./ARCHITECTURE.md).

### Non-Goals (What the system does NOT do)

*   It does not replace general search or agent frameworks (LangChain/LangGraph) — it focuses solely on routing to specialized models.
*   It does not train or fine-tune the models themselves.
*   It does not store dialogue indefinitely — only active context with a TTL (Time-To-Live).
*   It is not a RAG encyclopedia: vector memory is used for dialogue context, not as a knowledge base.
*   It does not include a UI chat client: we provide an OpenAI-compatible API; the frontend is client-side.

## Roadmap

### Phase 0 — Core Backbone ✅
*   **In-memory implementation:** `Planner → Plan(DAG) → Executor → mock-agents`.
*   Topological sorting and threaded dependency execution proven.
*   Here, "Executor" is an in-memory task loader; in Phase 3, it becomes the Dispatcher (delivery to NATS).
*   **Exit Criteria:** `python src/main.py` returns RESULTS in the correct order.

### Phase 1 — LLM Decomposition ← *Current Risk*
*   Implementation of `OpenAIPlanner using real prompts.
*   Handling of malformed JSON, cycles, and dangling `depends_on` references (via retry / Pydantic validation).
*   **Exit Criteria:** For 10 different requests, the planner consistently outputs a valid `Plan that is semantically useful: standard decomposition matches expert evaluation (no random/redundant tasks, meaningful dependencies). If this fails, the architecture is questionable, and we will not proceed to infrastructure build-out.

### Phase 2 — Real Agents (In-Memory)
*   Replacement of mocks with actual model calls (`code_review` / `doc_gen` / `translate`).
*   Same contract: `(instruction, ctx) -> result`.
*   Development of the **Finalizer**: assembling a coherent answer from parts (N→1, branching dependencies, merge without loss/duplication) and fixing the stub for the OpenAI response contract (see Phase 6).
*   **Exit Criteria:** A real prompt yields a real result through the DAG; the final answer is assembled correctly.

### Phase 3 — Transport (NATS / JetStream)
*   **Transport layer:** FastStream (NATS/JetStream): declarative `subscriber`/`publisher`, Pydantic contracts, built-in DLQ/retry.
*   Orchestrator and agents become separate processes; the executor communicates via the broker (`tasks.*` / `results`). The previously deferred skeleton moves here.
*   Inside the Orchestrator, **independent modules** are allocated: `Planner → Router → Dispatcher → Finalizer` (decomposition / model selection / delivery to NATS with retry+breaker / assembly).
*   **Router:** Default method is **Catalog-driven** (Agent Catalog + JSON-schema, see ARCH §2.3); embedding classifier is a fallback.
*   Large result artifacts are handled via `StoreService` (NATS Object Store → MinIO/S3 for >1 GB); only `result_ref` travels over the bus.
*   Basic subject hierarchy: `tasks.{type}`, `results.{type}`.
*   **Exit Criteria:** Phase 2 works in a distributed manner, not in a single process; latency/backpressure is measured — distributed execution does not degrade compared to in-memory (otherwise §7/"do not do for 1000 RPS" remains a risk).

### Phase 4 — Resilience
*   DLQ + TTL, idempotency by `task_id+type` (KV), moving plan state to KV (orchestrator recovery), circuit breaker, fallback model for unknown `task_type` / low Router confidence (default model).
*   **Exit Criteria:** A killed agent/orchestrator does not lose progress; every request always returns a response.

### Phase 5 — Vector Memory
*   Qdrant: dialogue context is loaded into the planner/agents before execution.
*   This phase is independent and does not gate Phases 0–4; if agent correctness already requires history, it can be pulled in earlier (to Phase 2/3).
*   **Exit Criteria:** Agents see history between calls.

### Phase 6 — Ingress Gateway
*   FastAPI: tokens, launching `Orchestrator.run()` on HTTP request, OpenAI-format response (contract already fixed as a stub in Phase 2).
*   Direct proxy for simple requests (`< 50` tokens, no verbs like "find/do/write") to the default model without decomposition — no waiting for DAG.
*   **Exit Criteria:** An external user hits a single endpoint; simple requests respond without delay from the planner.

### Phase 7 — Scale / Observability
*   N instances of agents (queue groups), metrics, event traces.
*   Subject hierarchy with priorities: `tasks.{type}.{priority}` (high → fast models, low → night batches), queue monitoring by type, subscription to specific `task_id` for debugging.
*   Priority queues are defined by a separate static config; DLQ for items failed after all retries moves here as well.
