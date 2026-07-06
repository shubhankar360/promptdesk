"""Retriever tests - pure Python, no API keys needed."""

from app.rag import Retriever, chunk_markdown, tokenize

DOCS = [
    ("Shipping Policy", "# Shipping\n\n## Standard shipping\nStandard shipping takes 5-7 business days and costs $4.99.\n\n## Express shipping\nExpress takes 1-2 business days."),
    ("Returns Policy", "# Returns\n\n## Return window\nItems can be returned within 30 days of delivery for a full refund."),
]


def make_retriever():
    r = Retriever()
    r.index_texts(DOCS)
    return r


def test_tokenize_removes_stopwords():
    assert "the" not in tokenize("How long is the shipping?")
    assert "shipping" in tokenize("How long is the shipping?")


def test_chunk_markdown_splits_on_headings():
    chunks = chunk_markdown(DOCS[0][1], "Shipping Policy")
    assert len(chunks) == 2
    assert all(c["title"] == "Shipping Policy" for c in chunks)


def test_search_finds_relevant_chunk():
    r = make_retriever()
    results = r.search("how long does standard shipping take")
    assert results
    assert results[0]["title"] == "Shipping Policy"
    assert "5-7" in results[0]["text"]


def test_search_ranks_returns_doc_for_refund_query():
    r = make_retriever()
    results = r.search("can I get a refund on my order")
    assert results
    assert results[0]["title"] == "Returns Policy"


def test_search_empty_query_returns_nothing():
    r = make_retriever()
    assert r.search("") == []


def test_irrelevant_query_scores_below_threshold():
    r = make_retriever()
    assert r.search("quantum entanglement propulsion") == []
