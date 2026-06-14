# Failure analysis: small SochDB Agent Memory QA sanity benchmark

## Summary

At k=3, the SochDB-style retriever answers all 16 questions correctly, while the vector-style lexical retriever misses 2 questions.

The remaining vector baseline failures are:

- q4: temporal priority update failure
- q12: exact multi-fact benchmark-spec retrieval failure

These failures are useful because they expose limitations of topic-similarity retrieval for agent-memory workloads.

## q4: Temporal priority update failure

Question:

> What is the highest-priority benchmark after the priority update?

Gold answer:

> Modular Baseline vs SochDB.

Expected evidence turn:

- turn 8

Vector retrieved:

- turns 6, 19, 17

Interpretation:

The vector-style baseline retrieved semantically adjacent benchmark-planning context, including a statement about the highest-priority benchmark and later paper-integrity/Section 11 notes. However, it missed the actual priority-update record where the plan was corrected:

> Originally I thought Agent Memory QA should be the first benchmark, but revise that. Modular Baseline vs SochDB should be P0.1, Context-Artifact Consistency Race should be P0.2, and Agent Memory QA should be P0.3.

This is a temporal-update failure: the retriever found related context but did not recover the memory that changed the current state.

## q12: Exact benchmark-spec retrieval failure

Question:

> Which strategies should Token Budget vs Answer Quality compare?

Gold answer:

> Top-k concatenation, BM25 concatenation, hybrid concatenation, planner, TOON, and planner plus TOON.

Expected evidence turn:

- turn 14

Vector retrieved:

- turns 4, 17, 13

Interpretation:

The vector-style baseline retrieved the general topic of Token Budget vs Answer Quality, but missed the exact benchmark configuration record containing the strategy list.

This is a multi-fact specification failure: broad semantic relevance was not enough to recover the precise configuration needed to answer correctly.

## Takeaway

The sanity benchmark distinguishes generic topic retrieval from SochDB-style memory retrieval. The vector-style baseline performs well on broad conceptual questions but fails on:

1. Later priority corrections.
2. Exact benchmark configuration records.

These are representative of agent-memory workloads where the system must retrieve current, specific, and provenance-grounded memory rather than merely semantically similar context.
