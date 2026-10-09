# Naive RAG — Movie Script Q&A

A command-line Retrieval-Augmented Generation (RAG) pipeline over movie scripts from
[IMSDb](https://imsdb.com). Pick a movie, ask a question about it, and get an answer
generated from the actual script text.

## Pipeline

1. **Load** — Scrapes the full movie list from IMSDb (`WebBaseLoader` + BeautifulSoup),
   then loads the chosen movie's script (`IMSDbLoader`).
2. **Split** — Cleans whitespace and splits the script into scenes on `INT.` markers
   (`RecursiveCharacterTextSplitter`, chunk size 500 / overlap 10).
3. **Embed** — Embeds each scene (`OpenAIEmbeddings`, `text-embedding-3-small`,
   1536-dim, cosine distance) and stores them in a Qdrant collection named after the
   movie (spaces replaced with hyphens, e.g. `Mission-Impossible`). Skips re-embedding
   if a valid collection already exists.
4. **Retrieve** — Rewrites the user's question into a keyword-focused search query
   (`ChatOpenAI`), then runs similarity search against the vector store to retrieve
   the top 5 relevant scenes.
5. **Generate** — Feeds the retrieved scenes as context to a chat model, which answers
   the user's original question in detail.

## Requirements

- Python 3.10+
- A running [Qdrant](https://qdrant.tech) instance on `localhost:6333`:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
  ```
- An OpenAI API key with access to chat and embedding models.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this directory (never commit it — see `.gitignore`):

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
python main.py
```

The program prints the full list of movies available on IMSDb, then prompts for:

1. A **movie title** (must match a title from the printed list exactly).
2. A **question** about that movie's script.

### Example

```
> Mission Impossible

Loaded script for Mission Impossible from https://imsdb.com/scripts/Mission-Impossible.html.

Found 129 scenes in the script for Mission Impossible.

Embedded script for Mission Impossible.

> Only scenes involving trains
Rewritten query to: train scenes

Final Answer:
...
```

If the movie title isn't found in the IMSDb list, the program reports it and exits:

```
> HyperMovie
Script for 'HyperMovie' wasn't found in the list of movie scripts.
```
