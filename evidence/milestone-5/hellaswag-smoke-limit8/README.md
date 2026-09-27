# Milestone 5 HellaSwag Smoke Evidence

This directory preserves the first successful real HellaSwag smoke evaluation for Milestone 5.

Status: PASSED
Authoritative: false
Benchmark claim allowed: false
Task: hellaswag
Limit: 8
Sample count: 8

Source checkpoint:
622229feb876e2ee3c487c58b5da791193af9139

Frozen model revision:
7ae557604adf67be50417f59c2c2f167def9a775

Frozen dataset revision:
218ec52e09a7e7462a5400043bb9a69a41d06b76

Observed smoke metrics:
- acc: 0.25
- acc_norm: 0.0

These metrics are smoke-only evidence and are not the authoritative quality baseline.

Observed warnings:
- The instruct/chat checkpoint was evaluated with apply_chat_template=false, as required by the frozen Milestone 5 protocol.
- Hugging Face Hub requests were unauthenticated.
- Dataset materialization fetched train, test, and validation artifacts although the evaluation task uses validation.

This directory is not a canonical Huyawo Evidence Bundle.
