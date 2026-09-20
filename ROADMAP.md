# Research server roadmap

Ideas for later, not commitments. Nothing here is scheduled, and several items would need a decision before they were built. Current capabilities live in `README.md`; the plan behind the recent work and the measurable targets live in `PLAN.md`.

The server deliberately targets English-primary born-digital PDF and EPUB sources with 5,000–50,000 chunks on CPU. Anything outside that envelope — other-language corpora, scanned or OCR'd material, formula-heavy documents — is a different product rather than a roadmap step, so it is not listed here.

## Retrieval quality

- Write a representative set of evaluation queries with relevance judgments, so "does search work well here?" has an answer.
- Measure BM25, dense, hybrid, and reranked quality against it.
- Make fusion weights configurable *only* if that measurement shows a repeatable benefit; guessing at weights without it would be worse than the current constants.

## Ingestion lifecycle

- List, inspect, and prune retained generations. They are roughly 101 MB each and currently accumulate with no built-in way to clean up.
- Roll back to an earlier generation deliberately, instead of only ever moving forward.
- Check available disk space before starting a build.

## Citations and quotation

- Preserve character offsets within extraction units, so a hit can point at a span rather than a whole unit.
- Distinguish a PDF's physical page from its printed page label.
- Add an exact-quote verification tool, which is the missing piece between "cleaned semantic text" and "safe to quote".
- Export citations in common bibliographic styles without inventing any metadata.

## Other

- Optional backup profiles that exclude originals, for users who already have their PDFs safely stored elsewhere.
- Optional cross-project search that keeps every project's boundary explicit, rather than merging indexes.
