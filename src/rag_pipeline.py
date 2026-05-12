import chromadb


chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="sec_filings"
)


def chunk_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


def store_filing_text(
    ticker,
    filing_text
):

    chunks = chunk_text(
        filing_text
    )

    ids = [
        f"{ticker}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "ticker": ticker,
            "source": "SEC 10-K"
        }
        for _ in chunks
    ]

    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)


def retrieve_context(
    query,
    n_results=3
):

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    retrieved_context = "\n\n".join(
        results["documents"][0]
    )

    return retrieved_context