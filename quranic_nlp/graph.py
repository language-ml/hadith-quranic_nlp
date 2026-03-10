"""
quranic_nlp.graph
~~~~~~~~~~~~~~~~~
Build verse-similarity graphs for a surah and extract central verses.

Usage
-----
::

    from quranic_nlp import language, graph

    nlp = language.Pipeline('pos,root,lem', 'fa#1')

    # Get all verse docs for a surah
    docs = language.surah_docs(nlp, 'فاتحه')   # or surah_docs(nlp, 1)

    # Build a similarity graph (TF-IDF over surface + lemma + root)
    G = graph.build_graph(docs, rep='tfidf')

    # Or with a sentence-embedding model (any model with .encode())
    # G = graph.build_graph(docs, rep='embedding', model=my_model)

    # Find the most central verse
    doc, scores = graph.central_verse(G, docs, method='pagerank')
    print(doc._.surah, doc._.ayah, doc._.text)

    # Maximum Spanning Tree
    T = graph.mst(G)
    doc, scores = graph.central_verse(T, docs, method='degree')
"""

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _verse_tokens(doc):
    """Return a string of surface + lemma + root tokens for TF-IDF input."""
    tokens = []
    for token in doc:
        tokens.append(str(token))
        if token.lemma_:
            tokens.append(token.lemma_)
        if token._.root:
            tokens.append(str(token._.root))
    return ' '.join(tokens)


def _tfidf_sim(docs):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError(
            "scikit-learn is required for rep='tfidf'. "
            "Install it with: pip install scikit-learn"
        )
    texts = [_verse_tokens(doc) for doc in docs]
    X = TfidfVectorizer().fit_transform(texts)
    return cosine_similarity(X)


def _embedding_sim(docs, model):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError(
            "scikit-learn is required for cosine similarity. "
            "Install it with: pip install scikit-learn"
        )
    texts = [doc._.text or str(doc) for doc in docs]
    embeddings = model.encode(texts)
    return cosine_similarity(np.array(embeddings))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(docs, rep='tfidf', model=None, threshold=0.0):
    """
    Build a verse-similarity graph from a list of processed docs.

    Parameters
    ----------
    docs : list[Doc]
        Verse docs, e.g. from ``language.surah_docs(nlp, surah)``.
    rep : str
        Representation method:

        * ``'tfidf'``  — TF-IDF over concatenated surface + lemma + root tokens.
          Requires ``pos``, ``lem``, ``root`` pipelines and ``scikit-learn``.
        * ``'embedding'`` — cosine similarity of sentence embeddings via ``model``.
    model : optional
        Any model with a ``.encode(list[str]) -> np.ndarray`` method
        (e.g. ``sentence_transformers.SentenceTransformer``).
        Required when ``rep='embedding'``.
    threshold : float
        Minimum similarity score to create an edge. Default ``0.0`` (all edges).

    Returns
    -------
    networkx.Graph
        Nodes are integer indices (0-based, matching ``docs`` order).
        Each node has attributes ``ayah``, ``surah``, ``text``.
        Edges carry ``weight`` = cosine similarity.
    """
    if rep == 'tfidf':
        sim = _tfidf_sim(docs)
    elif rep == 'embedding':
        if model is None:
            raise ValueError("model must be provided when rep='embedding'")
        sim = _embedding_sim(docs, model)
    else:
        raise ValueError(f"Unknown rep={rep!r}. Choose 'tfidf' or 'embedding'.")

    G = nx.Graph()
    for i, doc in enumerate(docs):
        G.add_node(i, ayah=doc._.ayah, surah=doc._.surah, text=doc._.text or str(doc))

    n = len(docs)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w > threshold:
                G.add_edge(i, j, weight=w)

    return G


def mst(G):
    """
    Return the Maximum Spanning Tree of a similarity graph.

    Internally negates edge weights and calls
    ``networkx.minimum_spanning_tree``.

    Parameters
    ----------
    G : networkx.Graph
        Weighted similarity graph from ``build_graph``.

    Returns
    -------
    networkx.Graph
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, weight=-d.get('weight', 1.0))
    return nx.minimum_spanning_tree(H)


def central_verse(G, docs, method='pagerank'):
    """
    Return the most central verse in the graph.

    Parameters
    ----------
    G : networkx.Graph
        Weighted similarity graph (from ``build_graph`` or ``mst``).
    docs : list[Doc]
        Verse docs in the same order as graph nodes.
    method : str
        Centrality measure:

        * ``'pagerank'``    — weighted PageRank (default)
        * ``'degree'``      — weighted degree centrality
        * ``'betweenness'`` — weighted betweenness centrality
        * ``'eigenvector'`` — weighted eigenvector centrality
        * ``'mst'``         — degree on the Maximum Spanning Tree

    Returns
    -------
    tuple[Doc, dict]
        ``(most_central_doc, {node_index: score, ...})``

    Example
    -------
    ::

        G = graph.build_graph(docs)
        doc, scores = graph.central_verse(G, docs, method='pagerank')
        print(doc._.surah, doc._.ayah, doc._.text)
    """
    if method == 'pagerank':
        scores = nx.pagerank(G, weight='weight')
    elif method == 'degree':
        scores = dict(G.degree(weight='weight'))
    elif method == 'betweenness':
        scores = nx.betweenness_centrality(G, weight='weight')
    elif method == 'eigenvector':
        scores = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    elif method == 'mst':
        T = mst(G)
        scores = dict(T.degree(weight='weight'))
    else:
        raise ValueError(
            f"Unknown method={method!r}. "
            "Choose 'pagerank', 'degree', 'betweenness', 'eigenvector', or 'mst'."
        )

    central_idx = max(scores, key=scores.get)
    return docs[central_idx], scores
