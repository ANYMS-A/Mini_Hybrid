import chromadb
from typing import Callable


def query_from_chroma(
        query_texts: str,
        client: chromadb.HttpClient,
        collection_name: str,
        embedding_fn: Callable,
        top_k: int = 3,
):

    collection: chromadb.Collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    results = collection.query(
        query_texts=query_texts,
        n_results=top_k,
    )

    return results

