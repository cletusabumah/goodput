# Roadmap

High-level view. Details and Done-when criteria live in [`master-plan.md`](master-plan.md).

| Phase | Theme | Approx weeks | Demo-able outcome |
|-------|-------|--------------|-------------------|
| **0** | Foundation | 1 | Docs + CI + importable package |
| **1** | Kill → restore → goodput | 2–3 | 2-worker crash recovery with a number |
| **2** | Fast ckpt A/B | 4–5 | Goodput vs failure-rate chart |
| **3** | Harder faults + $ story | 6–7 | Hang/bitflip + dollar delta |
| **4** | Stretch scale | 8+ | Workers×goodput / Colab / tracker |

```mermaid
gantt
  title Goodput 6–8 week plan
  dateFormat  YYYY-MM-DD
  section Foundation
  Phase 0 docs+CI           :p0, 2026-07-23, 7d
  section Core
  Phase 1 trainer+kill      :p1, after p0, 14d
  Phase 2 fast ckpt+plots   :p2, after p1, 14d
  section Narrative
  Phase 3 faults+$          :p3, after p2, 14d
  Phase 4 stretch           :p4, after p3, 7d
```

Dates are relative to Phase 0 start; adjust `what_i_learned.md` weekly.
