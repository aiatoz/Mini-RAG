import os
import sys

import chromadb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#------------------------------------------------------<Client & Collection>------------||
'''
PersistentClient WRITES STRAIGHT TO DISK AT config.chromaDir, NO SEPARATE
persist() CALL IS ACTUALLY REQUIRED BY CHROMA ITSELF (UNLIKE OLDER VERSIONS)
BUT WE STILL EXPOSE persist() BELOW TO MATCH THE INTERFACE FROM THE DESIGN DOC
AND TO KEEP vectordb.py SWAPPABLE WITH A FAISS BACKEND LATER

get_or_create_collection SO RE-RUNNING build_index.py DOESN'T CRASH ON A
COLLECTION THAT ALREADY EXISTS
'''
chromaClient = chromadb.PersistentClient(path=config.chromaDir)

collection = chromaClient.get_or_create_collection(
    name=config.collectionName,
    metadata={"hnsw:space": config.distanceMetric},
)




#------------------------------------------------------<Add>----------------------------||
'''
STORES CHUNKS + THEIR EMBEDDINGS + METADATA TOGETHER
chunkList AND embeddingList MUST BE THE SAME LENGTH AND SAME ORDER
(GUARANTEED BY embeddings.embedDocuments() RETURNING VECTORS IN INPUT ORDER)

idList IS BUILT AS "documentId_chunkId" SO IT'S UNIQUE ACROSS THE WHOLE
CORPUS, NOT JUST WITHIN ONE DOCUMENT
'''
def add(chunkList, embeddingList):
    idList = [f"{chunk.documentId}_{chunk.chunkId}" for chunk in chunkList]
    documentTextList = [chunk.text for chunk in chunkList]
    metadataList = [chunk.toMetadata() for chunk in chunkList]

    collection.add(
        ids=idList,
        embeddings=embeddingList.tolist(),
        documents=documentTextList,
        metadatas=metadataList,
    )



#------------------------------------------------------<Search>--------------------------||
'''
TAKES A SINGLE QUERY EMBEDDING, RETURNS THE topK NEAREST CHUNKS
RETURNS A LIST OF DICTS INSTEAD OF THE RAW CHROMA RESPONSE SO retriever.py
DOESN'T NEED TO KNOW ANYTHING ABOUT CHROMA'S RESPONSE SHAPE
(KEEPS CHROMA SWAPPABLE FOR FAISS LATER, PER THE DESIGN DOC)
'''
def search(queryEmbedding, topK=None):
    if topK is None:
        topK = config.topK

    rawResults = collection.query(
        query_embeddings=[queryEmbedding.tolist()],
        n_results=topK,
    )

    resultList = []
    for i in range(len(rawResults["ids"][0])):
        resultList.append({
            "id": rawResults["ids"][0][i],
            "text": rawResults["documents"][0][i],
            "metadata": rawResults["metadatas"][0][i],
            "distance": rawResults["distances"][0][i],
        })

    return resultList





#------------------------------------------------------<Persist>------------------------||
'''
NO-OP FOR CHROMA'S PersistentClient (IT WRITES ON EVERY add() ALREADY)
KEPT AS A REAL FUNCTION SO build_index.py CAN CALL vectordb.persist()
WITHOUT CARING WHICH BACKEND IS ACTUALLY IN USE
'''
def persist():
    pass





#------------------------------------------------------<Manual Test>--------------------||
'''
RUN "python src/vectordb.py" DIRECTLY TO VERIFY THE FULL
loader -> splitter -> embeddings -> vectordb -> search ROUND TRIP
'''
if __name__ == "__main__":
    import embeddings
    import loader
    import splitter

    loadedDocuments = loader.loadDocuments()
    allChunks = splitter.splitDocuments(loadedDocuments)
    documentEmbeddings = embeddings.embedDocuments(allChunks)

    add(allChunks, documentEmbeddings)
    persist()
    print(f"Added {len(allChunks)} chunk(s) to collection '{config.collectionName}'")

    sampleQueryEmbedding = embeddings.embedQuery("What is attention in transformers?")
    searchResults = search(sampleQueryEmbedding, topK=3)

    print(f"\nTop {len(searchResults)} result(s):")
    for result in searchResults:
        preview = result["text"][:60].replace("\n", " ")
        print(f"  - {result['metadata']['source']} (dist={result['distance']:.4f}) -> {preview}...")
