# Naive vs `torch.distributed.checkpoint` (ticket 4.4)

**Done-when:** a side-by-side latency note. This is **not** a rewrite of training onto DCP.

## What each path is for

| | **This repo (naive / incremental)** | **`torch.distributed.checkpoint` (DCP)** |
|--|----------------------------------|-----------------------------------------------|
| API | `torch.save` / `torch.load` of one blob | `dcp.save` / `dcp.load` into a **directory** |
| Layout | `step_000001.pt` | shards + metadata under a checkpoint id |
| Rank model | Rank 0 dumps the full unsharded state | Coordinated save across ranks (FSDP / HSDP) |
| Optimizer | Naive copies it every time; incremental often skips it | Planner-aware; can shard optimizer state |
| CI | Always available | Importable on recent torch; some builds need a process group |

Our incremental path is a **serialize-less-optimizer** trick on the same pickle store. DCP is a **different storage protocol** for distributed sharded models. Comparing wall time on a 1-rank toy MLP is valid as a **sanity check**, not as “DCP is slower so we should not use it at 16k GPUs.”

## How to regenerate the numbers

```bash
goodput-run --dcp-compare experiments/dcp-compare.yaml
```

Writes `artifacts/sweeps/dcp-compare/table.md` (gitignored). The markdown repeats this framing plus median save/restore seconds and byte sizes.

## How to read the table

- **Naive** should always have finite save/restore — that is the 1.5 path.
- **Incremental** times the *model-only* save after a full base. Restore may load two blobs (delta + base), so restore is not always cheaper.
- **DCP** may show `available: no` if this torch build cannot save without `init_process_group`. That is a documented skip, not a CI failure.

## What we did not do

- Did not switch `LocalFsCheckpointStore` to DCP.
- Did not run multi-node FSDP.
- Did not claim a winner from millisecond CPU timings.
