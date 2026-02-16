# AI Tooling

## Tools Used

### Claude (Anthropic) — Primary Development Assistant
**How we used it:** Claude was our primary AI coding assistant throughout the project. We used it to:
- Review and audit existing code for bugs (e.g., identifying the chromadb 0.3.x / Python 3.11 incompatibility and the incorrect refusal threshold logic)
- Generate the updated `ingest.py`, `rag.py`, `app_streamlit.py`, `requirements.txt`, `render.yaml`, and `ci.yml` files
- Write the complete `evaluate.py` script including the 20-question evaluation set and all three metric implementations
- Draft `design-and-evaluation.md` and `ai-tooling.md`
- Debug import errors and explain the Chroma API migration from 0.3 → 0.4

**What worked well:** Claude was excellent at reviewing code holistically — it caught subtle issues like the `st.experimental_rerun()` deprecation, the `pydantic==1.10` / `chromadb>=0.4` conflict, and the `DATA_DIR` double-assignment bug. It also generated complete, runnable files rather than snippets, which saved significant time. The back-and-forth review workflow (share code → get annotated critique → get fixed version) was highly effective.

**What didn't work as well:** Claude occasionally over-explained things we already understood. For very long files, it sometimes needed a reminder to keep all existing functionality intact while making targeted fixes.

### GitHub Copilot — In-IDE Autocomplete
**How we used it:** Used within VS Code for autocomplete while writing the evaluation question set, filling in boilerplate, and writing test assertions.

**What worked well:** Fast for repetitive patterns (e.g., adding another eval question in the same format, writing similar test cases).

**What didn't work as well:** Less reliable for project-specific logic like the Chroma client initialization — it suggested the old 0.3.x API pattern which we had just fixed.

## Summary
AI tooling accelerated the project significantly, particularly for code review, dependency conflict resolution, and generating the evaluation harness. We estimate AI assistance reduced development time by roughly 60–70% compared to writing everything from scratch. All generated code was reviewed, understood, and tested by the team before being committed.