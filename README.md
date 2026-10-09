## l2_chunking

### Overview

`l2_chunking` is the second pipeline layer. `src/rag.py` runs it right after `l1_ingestion` and passes it the ingested documents in memory, as a dict of `doc_id` to `Document`, together with an output folder `{--output_path}/{dataset_name}`, where `dataset_name` is the last part of `--input_path` (for example `outputs/pdfa-eng-wds`). The chunking settings travel as a `ChunkingArgs`: `--framework`, `--model_name`, `--chunk_type` (`fixed`, `sentence` or `semantic`), `--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap` and `--semantic_threshold`. `Chunking.compute_chunks()` first validates these settings, then splits each document into `Chunk` records (`id`, `text`, `size` in tokens, `type`). Token counts come from the tokenizer of the SentenceTransformer model named by `--model_name`; `semantic` chunking also uses that model to embed every sentence. All chunks of one document are saved together as a JSON list in `{output folder}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{doc_id}.json`. The file is written to a temporary file first and then renamed into place, so it is either complete or absent. A document whose file already exists is loaded from it instead of being chunked again, and for `semantic` chunking only the documents that are not cached have their sentences embedded. `compute_chunks()` returns all chunks as a dict of `chunk_id` to `Chunk` and prints the chunk folder and how many documents and chunks came from cache or were newly computed.

With `--framework ""` the built-in chunkers are used. With `--framework langchain`, `compute_chunks()` hands the documents to the `Langchain` class instead, which chunks with LangChain's splitters for all three chunk types, using the same settings and the same per-document cache, saved under `{output folder}/chunks/langchain/...` so the two never share a cache. LangChain fixed and sentence chunking follow the same rules as the built-in ones; LangChain semantic chunking uses LangChain's own `SemanticChunker` rule and does not use `--semantic_threshold` or `--min_chunk_size`. The SentenceTransformer model is loaded by `Chunking` only for the built-in chunkers, because the `Langchain` class loads its own copy. LangChain chunking needs the `langchain-text-splitters` and `langchain-experimental` packages.

### Architecture

- **`chunking.py`** — `Chunking`, the orchestrator: validates the settings and, for `--framework langchain`, hands the work to `Langchain`. For the built-in chunkers it loads the SentenceTransformer model, loads cached chunks, embeds sentences for semantic chunking, chunks the remaining documents, saves them and prints the run summary.
- **`langchain.py`** — `Langchain`: loads cached chunks, chunks the remaining documents with one of `fixed_chunking` (LangChain's `SentenceTransformersTokenTextSplitter`), `sentence_chunking` (`NLTKTextSplitter`) or `semantic_chunking` (`SemanticChunker` from `langchain-experimental`), saves them and prints the run summary. Sentence and semantic pieces longer than `max_chunk_size` are cut with LangChain's `split_text_on_tokens`. Also holds `SentenceTransformerEmbeddings`, which lets `SemanticChunker` embed with the already-loaded model.
- **`chunker.py`** — The abstract `Chunker` base class shared by all built-in strategies (chunk id and size handling, and a sliding token-window splitter that stops once a window reaches the last token), and `get_chunker()`, the registry that maps a `chunk_type` to its strategy.
- **`chunkers/fixed.py`** — `FixedChunking`: splits a document's tokens into windows of `max_chunk_size` tokens that overlap by `fixed_chunk_overlap` tokens.
- **`chunkers/sentence.py`** — `SentenceChunking`: packs whole sentences into chunks of up to `max_chunk_size` tokens; a sentence longer than that is split with the fixed window.
- **`chunkers/semantic.py`** — `SemanticChunking`: like sentence chunking, but also starts a new chunk where the cosine similarity between neighbouring sentence embeddings falls below `semantic_threshold`, then merges chunks smaller than `min_chunk_size` into the previous chunk when they fit.
- **`cache.py`** — Saves and loads one JSON file per document in the chunk folder.
- **`validate.py`** — Checks the `ChunkingArgs` values before any chunking starts.
- **`schema.py`** — Defines `Chunk`, the layer's output record.

### Run results

All runs below use `pixparse/pdfa-eng-wds` with `--model_name all-MiniLM-L6-v2 --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.2`.

**5000 documents, built-in vs LangChain** (from `log/l2_chunking/log_5k_builtin_{fixed,sentence,semantic}.txt`, `log/l2_chunking/log_5k_langchain_{fixed,sentence,semantic}.txt`, and the chunk statistics in `log/l2_chunking/chunk_stats_5k_builtin_vs_langchain.txt`). All six runs chunk the same 5000 documents, loaded from the `l1_ingestion` cache, with nothing in the chunk cache. The times are the `Chunking time` printed by `src/rag.py`, which covers the whole `Chunking(...).compute_chunks(...)` call, including loading the model.

| | built-in fixed | LangChain fixed | built-in sentence | LangChain sentence | built-in semantic | LangChain semantic |
|---|---|---|---|---|---|---|
| Chunks | 91670 | 91670 | 94053 | 94053 | 104900 | 112218 |
| Chunks per document | 18.3 | 18.3 | 18.8 | 18.8 | 21.0 | 22.4 |
| Mean / median size (tokens) | 249.8 / 256.0 | 250.0 / 256.0 | 218.2 / 239 | 218.3 / 239 | 195.6 / 220.0 | 199.5 / 256.0 |
| Min / max size (tokens) | 17 / 256 | 17 / 260 | 1 / 256 | 1 / 260 | 1 / 256 | 0 / 260 |
| Chunks below 128 tokens | 2166 (2.4%) | 2163 (2.4%) | 8253 (8.8%) | 8247 (8.8%) | 17368 (16.6%) | 26434 (23.6%) |
| Chunks of exactly 256 tokens | 86701 (94.6%) | 77822 (84.9%) | 15015 (16.0%) | 13951 (14.8%) | 13909 (13.3%) | 64290 (57.3%) |
| Chunks counted above 256 tokens | 0 | 8873 | 0 | 1067 | 0 | 5811 |
| Chunks whose text contains `[UNK]` | 0 | 4252 | 0 | 1048 | 0 | 3998 |
| Total tokens in all chunks | 22900712 | 22920513 | 20523464 | 20527069 | 20523464 | 22382959 |
| Chunking time | 41.90636873245239s | 67.74775099754333s | 46.04506301879883s | 123.76853036880493s | 325.39638090133667s | 567.8647112846375s |

The built-in semantic run spends 267.174s embedding 639475 sentences (`Embedded 639475 sentences in 267.174s` in its log).

**Fixed.** Both cut each document into windows of 256 tokens from the `all-MiniLM-L6-v2` tokenizer, moving 224 tokens forward each time (256 minus the 32-token overlap) and stopping once a window reaches the last token, so both produce the same 91670 chunks. Almost every window is a full 256 tokens; only the last window of each document is shorter. The fixed runs have about 2.4 million more tokens in total than the sentence runs because neighbouring windows repeat 32 tokens.

**Sentence.** Both cut the text into sentences with NLTK's `sent_tokenize` and pack whole sentences into a chunk while it stays within 256 tokens, with no overlap; a single sentence longer than 256 tokens is cut into overlapping 256-token windows. LangChain's `NLTKTextSplitter` keeps such a sentence whole on its own, so `Langchain` cuts it afterwards with LangChain's `split_text_on_tokens`. Both produce the same 94053 chunks, and they cover the text exactly once, so built-in sentence and semantic chunking have the same total token count (20523464).

**Where built-in and LangChain fixed/sentence chunks differ.** The chunks are cut at the same places; the differences come from turning token ids back into text and counting them:

- LangChain keeps the text `[UNK]`, the tokenizer's placeholder for a character it does not know, in the chunk text; the built-in chunkers decode with `skip_special_tokens=True`, which leaves it out. This happens only where token ids are decoded back into text: in every fixed chunk, and in sentence chunking only in windows cut from long sentences.
- LangChain's splitters only return text, so `Langchain` gets each chunk's `size` by tokenizing its text again, while the built-in chunkers take the size from the window or from the sentences' token counts. Tokenizing decoded text does not always give back the same number of tokens, which is why some LangChain chunks are counted at 257 to 260 tokens although they were cut from 256-token windows.

LangChain is slower for the same chunks because of that second tokenizing pass over every chunk; LangChain sentence also tokenizes every sentence while packing, which is why its gap to the built-in chunker is larger (123.76853036880493s vs 46.04506301879883s) than for fixed (67.74775099754333s vs 41.90636873245239s).

**Semantic.** The two use different rules, so their chunks are not expected to match. The built-in chunker starts a new chunk where the cosine similarity between neighbouring sentences falls below 0.2, never lets a chunk grow beyond 256 tokens, and merges chunks smaller than 128 tokens into the previous one. LangChain's `SemanticChunker` (run with `buffer_size=0`, so each sentence is embedded on its own) splits sentences after `.`, `?` or `!` and starts a new chunk only where the distance between neighbouring sentences is in the top 5% for that document (the 95th percentile), with no size limit. That leaves long pieces, which `Langchain` then cuts into overlapping 256-token windows: 57.3% of LangChain semantic chunks are exactly 256 tokens, its median is 256, and the window overlap adds about 1.9 million tokens compared with the built-in semantic chunks. It also leaves more chunks below 128 tokens (23.6% vs 16.6%), because it has no merge step. One LangChain semantic chunk has a size of 0: its text is not blank but contains nothing the tokenizer turns into a token.

Built-in semantic chunking is dominated by sentence embedding: 267.174s of its 325.39638090133667s go to running the model over all 639475 sentences in one batch, and the rest is the same sentence splitting and tokenizing work as sentence chunking. LangChain's `SemanticChunker` embeds one document at a time and `Langchain` tokenizes every chunk again for its size, which together make it slower (567.8647112846375s).

**100 documents, chunk-by-chunk comparison** (from `log/l2_chunking/log_100_fixed_vs_langchain.txt`, `log_100_sentence_vs_langchain.txt` and `log_100_semantic_vs_langchain.txt`). Each run chunks the same 100 documents with the built-in chunker and with LangChain and compares the chunks of each document; the times cover creating the `Chunking` object (including model loading) and the `compute_chunks()` call:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks, built-in / LangChain | 1543 / 1543 | 1569 / 1569 | 1763 / 1841 |
| Documents with identical chunks | 83 | 98 | 8 |
| Chunks with different text | 40 (38 only by `[UNK]`) | 4 (all only by `[UNK]`) | – |
| Chunks with identical text but different size | 126 | 28 | – |
| Time, built-in / LangChain | 3.25s / 4.57s | 4.41s / 6.15s | 7.84s / 13.4s |

For fixed and sentence, every chunk whose text differs contains `[UNK]`, and the size differences are 1 to 3 tokens, for the reasons above. For semantic, 86 of the 100 documents get a different number of chunks, as expected from the different rules.

**Earlier runs, before the built-in chunkers stopped at the last token** (from `log/l2_chunking/log_100_fixed.txt`, `log_5k_fixed.txt`, `log_5k_sentence.txt`, `log_5k_semantic.txt` and `chunk_stats_5k.txt`). The built-in window splitter used to start one more window after a window had already reached the last token; that window held only the last tokens of the document, which were already inside the previous window. On 100 documents fixed chunking produced 1555 chunks instead of 1543, and on 5000 documents fixed, sentence and semantic produced 92366, 94473 and 105050 chunks instead of 91670, 94053 and 104900; in sentence and semantic chunking the extra windows came from sentences longer than 256 tokens. These extra windows were also the 1-token chunks in fixed chunking: the smallest fixed chunk was 1 token then and is 17 tokens now.

The log line `Token indices sequence length is longer than the specified maximum sequence length for this model` is printed by the tokenizer when a whole document is tokenized for counting; the model is not run on those tokens.
