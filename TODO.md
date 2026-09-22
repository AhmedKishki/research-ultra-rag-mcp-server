# TODO

The open work on this server. Everything listed here is unimplemented. Current behaviour is in `README.md`, current facts and limits are in `MEASUREMENTS.md`, capabilities are in `FEATURES.md`, and speculative product ideas are in `ROADMAP.md`. Finished work lives in git history, not in this file.

## Retrieval quality

- **Pooled relevance judgments.** The current judged set is known-item — one designated passage per query, judged by a single annotator — so a passage that makes the same point scores as a miss and true recall is not claimed. Collect every candidate from every mode and judge the pool.
- **Keep the measurement current and wider.** The published numbers come from one corpus and one generation; re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and extend the set to a second corpus and to filtered queries.
- **Investigate the paraphrase gap.** Paraphrase queries are the weakest class for every mode, and depth recovers part of the loss. Either those judgments need revisiting, or the embedding model, the 0.72 cosine gate, and the fusion ordering do.
- **Fusion weights stay fixed** until a larger judged set shows a repeatable benefit: the hybrid lead over BM25 at rank 1 is currently about two queries out of 32, which cannot separate a real weight effect from noise.

## Ingestion lifecycle

- **Prune retained generations.** `status` lists each one with its size; removal is manual today and deletes data, so it needs a reviewed design: which generation, an explicit confirmation step, and never the current one.
- **Deliberate rollback** to a retained generation, instead of only ever moving forward.
- **Check free disk space before a build starts.**
- **An `ingest` dry run** that reports what would change, and what would be reused, without writing anything.
- **Structured activation failures** instead of one all-or-nothing message.
- **One meaning for `stale`** in the no-generation branch, where it currently doubles as "ready to build".
- **Narrow the two broad `except Exception` handlers** at durability boundaries so a real storage fault cannot be swallowed.

## Code structure

- **Split the resumable ingestion loop into per-phase handlers** and enable `C901` with a documented threshold. `service.py`'s `ingest` is roughly 1,100 lines, which makes reviewing a change to it risky; no complexity check is enabled today.

## Tool surface

- **Re-measure the `search` row of the tool-answer table** after the next reference build; `MEASUREMENTS.md` currently carries the pre-trim figure and explains why.
- **Decide how `list_sources` should expose the keyword vocabulary.** `reviewed_metadata_sources` echoes every saved override and is now the largest lean answer, but it is the only place an agent can discover which keywords exist. Either report keyword counts in `status` beside categories and projects, or reduce the review list to handles and leave vocabulary discovery to the filters.

## Shared UI

- **Surface the retained-generation inventory in the UI.** `status` reports `generations` and `retained_generation_bytes`; the pinned status view renders none of them.
- **Warn about a runtime-root mismatch in the standalone UI launcher,** which starts its own server and does not read an MCP client's configuration, so it can silently show a different generation than the agent. The server-hosted UI (`--ui-port`) is unaffected.
- **The standalone UI launcher does not take its private server down with it.** Killing `research-ultra-rag-ui` with `SIGTERM` releases the port but leaves the private stdio server it started running with `ppid 1`, together with a defunct gateway beneath it, so every stop leaks a server, a gateway and their UltraRAG children until they are reaped by hand. Reproduce by starting the launcher, recording `pgrep -P <pid>` for its child, killing the launcher, and confirming `ps -o pid,ppid,stat -p <child>` still shows it alive. The generated per-project launcher only works around it by starting the UI in its own session and stopping the whole process group; a direct `research-ultra-rag-ui` invocation still leaks, and the launcher cannot reap a private server started by a UI it did not launch. The server-hosted UI (`--ui-port`) is unaffected.
- **`RESEARCH_ULTRARAG_UI_PORT` cascades into nested servers.** The variable is the argparse default for `--ui-port`, and the private stdio client a UI starts inherits it, so that nested server hosts a UI of its own and starts the next client in turn: one exported variable turns a single UI into a self-feeding process cascade, measured at fifteen gateway starts a minute with orphaned server chains and repeated `BrokenPipeError` in `research-ui-mcp-stderr.log`. Reproduce with `RESEARCH_ULTRARAG_UI_PORT=<free port> research-ultra-rag-ui …` while watching `pgrep -f 'python3 -m research_ultra_rag_mcp'` and that log's start count. A launcher can avoid it by stripping the variable from the UI child's environment, and child transports should not inherit it at all.
- **Rebind or clear a lost `--ui-port` claim.** `EmbeddedUi.start` probes the port and then hands it to uvicorn, so two simultaneous server starts can both pass the probe: on a lost race uvicorn 0.53.0 exits the UI task through `sys.exit(STARTUP_FAILURE)` on the bind error, which leaves `ui_ready: false` with `ui_error: null`, and the documented reason is lost. A failed probe is never re-evaluated either, so once the winning instance exits the port stays unserved while `ui_error` still reports it as taken. Reproduce by starting two server instances simultaneously, comparing `status`'s `ui_ready` and `ui_error` against `ss -ltnp | grep <port>`, and then terminating the winner while the loser is still running.

## Verification

```bash
uv run pytest -q                        # unit and integration suite
uv run ruff check .                     # lint
uv run ruff format --check .            # formatting
uv run research-ultra-rag-verify /path/to/project --query "your question"
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
uv run python scripts/measure_tool_payloads.py --project /path/to/project --offline
```
