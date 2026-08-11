import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import embeddings
import vectordb


#--------------------------------------------------------------<Dense Retrieval>---------||
'''
PHASE 1 STRATEGY: EMBED THE QUERY, RUN COSINE SIMILARITY SEARCH IN CHROMA
STRAIGHT PASS-THROUGH TO vectordb.search(), NO RE-RANKING YET
'''
def denseRetrieve(queryText, topK):
    queryEmbedding = embeddings.embedQuery(queryText)
    return vectordb.search(queryEmbedding, topK=topK)




#--------------------------------------------------------------<Hybrid Retrieval>--------||
'''
PHASE 2 (FUTURE): MERGE DENSE RESULTS (EMBEDDINGS) WITH SPARSE RESULTS (BM25)
BEFORE RANKING, PER THE DESIGN DOC. NOT IMPLEMENTED YET, KEPT AS A CLEAR
EXTENSION POINT SO SWITCHING config.retrievalStrategy = "hybrid" HAS SOMEWHERE
TO ROUTE TO WITHOUT CHANGING retrieve() ITSELF
'''
def hybridRetrieve(queryText, topK):
    raise NotImplementedError("Hybrid retrieval (dense + BM25) is a Phase 4 enhancement, not implemented yet.")




#--------------------------------------------------------------<Retrieve>----------------||
'''
SINGLE ENTRY POINT FOR THE REST OF THE APP (rag.py, prompt.py, ETC)
HIDES WHICH STRATEGY IS ACTUALLY RUNNING BEHIND config.retrievalStrategy
SO SWAPPING dense -> hybrid LATER DOESN'T TOUCH ANY CALLING CODE
'''
def retrieve(queryText, topK=None):
    if topK is None:
        topK = config.topK

    if config.retrievalStrategy == "dense":
        return denseRetrieve(queryText, topK)
    elif config.retrievalStrategy == "hybrid":
        return hybridRetrieve(queryText, topK)
    else:
        raise ValueError(f"Unknown retrievalStrategy in config: {config.retrievalStrategy}")




#--------------------------------------------------------------<Manual Test>-------------||
'''
RUN "python src/retriever.py" DIRECTLY TO SANITY-CHECK RETRIEVAL
AGAINST WHATEVER IS CURRENTLY IN THE CHROMA COLLECTION
(RUN scripts/build_index.py FIRST IF THE COLLECTION IS EMPTY)
'''
if __name__ == "__main__":
    testQuery = "What is attention in transformers?"
    retrievedChunks = retrieve(testQuery, topK=3)

    print(f"Query: {testQuery}")
    print(f"Retrieved {len(retrievedChunks)} chunk(s):\n")

    for rank, chunk in enumerate(retrievedChunks, start=1):
        preview = chunk["text"][:70].replace("\n", " ")
        print(f"  {rank}. {chunk['metadata']['source']} (dist={chunk['distance']:.4f}) -> {preview}...")
