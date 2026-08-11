# Mini RAG

A framework-light, local-first Retrieval-Augmented Generation pipeline.
Built from scratch (no LangChain / LlamaIndex orchestration) to understand every
stage of a modern RAG system: loading, chunking, embedding, ANN search, prompt
construction, and local LLM generation.

Runs entirely on a local machine — no external API calls, no cloud vector DB.

## Hardware target

- GPU: RTX 4060 (8 GB VRAM)
- RAM: 16 GB
- CPU: Intel i7-12650H

## Project layout

```text
mini-rag/
├── data/            # source documents (.md / .txt), one subfolder per collection
├── cache/            # huggingface model cache
├── db/               # persistent Chroma store + bm25 index + metadata
├── notebooks/         # 01_indexing.ipynb (add/maintain the vector DB), 02_query.ipynb (ask questions, inspect retrieval)
├── prompts/           # system_prompt.txt — system instruction, kept out of Python code
├── scripts/          # entrypoints: build_index.py, query.py
├── src/               # pipeline modules (loader, splitter, embeddings, vectordb, retriever, prompt, generator, rag)
├── config.py          # single source of truth for all tunable parameters
└── requirements.txt
```

## Setup

```bash
conda create -n mini-rag python=3.11 -y
conda activate mini-rag

# Install torch with CUDA support first (check your CUDA version):
# https://pytorch.org/get-started/locally/
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

To remove the env later: `conda deactivate && conda env remove -n mini-rag`

## Status

Scaffold only — pipeline modules are being implemented phase by phase.

- [ ] Phase 1 — Indexing (loader → splitter → embeddings → ChromaDB)
- [ ] Phase 2 — Retrieval (dense top-K search)
- [ ] Phase 3 — Generation (prompt builder + local LLM + citations)
- [ ] Phase 4 — Enhancements (hybrid retrieval, reranking, semantic chunking, etc.)

## Notebooks

`notebooks/` contains two notebooks that mirror `scripts/build_index.py` and `scripts/query.py`,
useful for interactive exploration:

- **01_indexing.ipynb** — add new documents to `data/` and push them into ChromaDB, inspect the resulting collection
- **02_query.ipynb** — ask questions, and separately inspect retrieval (chunks + distances) and the final prompt before generation

To register the kernel so it matches the notebooks' kernelspec:

```bash
conda activate mini-rag
python -m ipykernel install --user --name mini-rag --display-name "mini-rag"
jupyter notebook notebooks/
```
