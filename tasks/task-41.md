# task-41 — Trainer shared memory for full-size training

**Depends on:** task-40
**Commit message:** `Give the trainer enough shared memory for full-size data loading`

## Description

The trainer's `/dev/shm` was 1 GB. DataLoader workers hand batches to the trainer through it; LR-ASPP at 798×532
(8 workers × 2 prefetched batches of 16, ~1.4 GB) exhausted it, the workers died with "unable to allocate shared
memory", and the run hung without failing. Raise `shm_size` to 16 GB (used only while batches are in flight).

## Testing

- The hung LR-ASPP run reproduced the error; relaunched with 16 GB it trains (see the run).
