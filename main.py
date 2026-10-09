"""Naive RAG CLI: answers questions about an IMSDb movie script via Qdrant + OpenAI."""

import os
import re

from dotenv import load_dotenv

os.environ.setdefault("USER_AGENT", "Naive-RAG-Task/1.0")
load_dotenv()

from langchain_community.document_loaders import IMSDbLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

ALL_SCRIPTS_URL = "https://imsdb.com/all-scripts.html"
SCRIPT_URL_TEMPLATE = "https://imsdb.com/scripts/{}.html"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
RETRIEVAL_TOP_K = 5

REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions about a movie script into a short search "
            "query that emphasizes the key terms needed to retrieve relevant "
            "scenes from a vector store of scene text. Respond with only the "
            "rewritten query, nothing else.",
        ),
        ("human", "{query}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert on movie scripts. Use the following scenes from "
            "the movie script as context to answer the user's request in "
            "detail.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def load_movie_titles():
    """Scrape the deduplicated, ordered list of movie titles from IMSDb's A-Z index."""
    soup = WebBaseLoader(ALL_SCRIPTS_URL).scrape()
    heading = soup.find("h1", string=lambda text: text and "All Movie Scripts" in text)

    titles = []
    seen = set()
    for link in heading.find_all_next("a", href=True):
        if not link["href"].startswith("/Movie Scripts/"):
            continue
        title = link.get_text(strip=True)
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def build_script_url(title):
    """Build the IMSDb script URL for a movie title."""
    return SCRIPT_URL_TEMPLATE.format(title.replace(" ", "-"))


def load_script(title):
    """Load the full script text and source URL for a movie title."""
    url = build_script_url(title)
    document = IMSDbLoader(url).load()[0]
    return document.page_content, url


def split_into_scenes(script):
    """Collapse whitespace and split the script into scene-sized chunks at `INT.`."""
    cleaned = re.sub(r"\s+", " ", script).strip()
    splitter = RecursiveCharacterTextSplitter(
        separators=["INT."], chunk_size=500, chunk_overlap=10, keep_separator="start"
    )
    return splitter.split_text(cleaned)


def collection_name_for(title):
    """Derive the Qdrant collection name for a movie title."""
    return title.replace(" ", "-")


def _has_valid_vector_config(client, collection_name):
    """Check the collection uses a bare (unnamed) vector config, not a named-vector dict."""
    vectors = client.get_collection(collection_name).config.params.vectors
    return (
        isinstance(vectors, VectorParams)
        and vectors.size == EMBEDDING_DIM
        and vectors.distance == Distance.COSINE
    )


def get_vector_store(collection_name, client=None):
    """Build a LangChain vector store bound to an existing Qdrant collection."""
    client = client or QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )


def embed_scenes(title, scenes):
    """Embed and store scenes in Qdrant, recreating the collection if malformed or absent."""
    collection_name = collection_name_for(title)
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(collection_name) and not _has_valid_vector_config(
        client, collection_name
    ):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        get_vector_store(collection_name, client).add_texts(scenes)


def rewrite_query(search_query):
    """Rewrite a free-form user question into a keyword-focused retrieval query."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = REWRITE_QUERY_PROMPT | llm
    response = chain.invoke({"query": search_query})
    return response.content.strip()


def retrieve_scenes(title, search_query):
    """Return the top-K scene texts most similar to the search query."""
    vector_store = get_vector_store(collection_name_for(title))
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    return [doc.page_content for doc in retriever.invoke(search_query)]


def generate_answer(user_query, scenes):
    """Generate a detailed answer to the user's original question using retrieved scenes as context."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"context": "\n\n".join(scenes), "question": user_query})
    return response.content.strip()


def main():
    """Run the CLI: load a movie's script, embed it, then answer a question about it."""
    titles = load_movie_titles()
    for index, title in enumerate(titles, start=1):
        print(f"{index}. {title}")

    movie_title = input().strip()

    if movie_title not in titles:
        print(f"Script for '{movie_title}' wasn't found in the list of movie scripts.")
        return

    script, url = load_script(movie_title)
    print(f"\nLoaded script for {movie_title} from {url}.\n")

    scenes = split_into_scenes(script)
    print(f"Found {len(scenes)} scenes in the script for {movie_title}.\n")

    embed_scenes(movie_title, scenes)
    print(f"Embedded script for {movie_title}.\n")

    search_query = input().strip()
    rewritten_query = rewrite_query(search_query)
    print(f"Rewritten query to: {rewritten_query}\n")

    scenes_context = retrieve_scenes(movie_title, rewritten_query)
    answer = generate_answer(search_query, scenes_context)
    print(f"Final Answer:\n{answer}")


if __name__ == "__main__":
    main()
