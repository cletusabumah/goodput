# Vision

## Why this exists

At cluster scale, individually reliable GPUs become routinely unreliable. A single H100-class GPU may have an MTBF on the order of ~50,000 hours; at 16,384 GPUs that becomes an interruption roughly every few hours. Meta’s Llama 3 405B training logged hundreds of unexpected hardware interruptions over ~54 days — and automated recovery is what made that train feasible.

**Goodput** is the fraction of total GPU-hours that go toward useful, un-lost training progress (as opposed to failures, restarts, and idle recovery). This repository builds a **scaled-down simulator** of that loop: shard a toy model across workers, checkpoint periodically, kill (or hang, or corrupt) a worker mid-run, and measure goodput with and without fast incremental checkpointing.

## Who it serves

| Persona | What they need |
|---------|----------------|
| **Cletus (builder / interviewee)** | A crisp before/after goodput chart, reproducible experiments, and a portfolio narrative that sounds like ML infra — not “I trained a toy model.” |
| **ML infra interviewer** | Evidence you understand synchronized training failure modes, checkpoint trade-offs, and how to quantify dollar impact of a few goodput points. |
| **Future self / collaborator** | A repo that clones in <30 minutes, has Done-when tickets, CI, and no tribal knowledge. |

## What success looks like

1. A chart of **goodput vs injected failure rate**, with vs without fast incremental checkpointing.
2. **Checkpoint/restore latency** vs simulated worker count.
3. A **rough dollar model**: simulated cluster size × public GPU $/hr × measured goodput delta.

## What this is not

- Not a production training framework (DeepSpeed / Megatron competitor).
- Not a SaaS dashboard or multi-tenant product.
- Not a research paper substitute — it is an engineering artifact with metrics.
