import os
import sys

from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#-------------------------------------------------------------<Model Loading>-----------||
'''
LOADED ONCE AT MODULE IMPORT TIME, NOT INSIDE EVERY FUNCTION CALL
cacheDir KEEPS THE DOWNLOADED WEIGHTS OUT OF THE HF DEFAULT ~/.cache
SO EVERYTHING STAYS INSIDE THE PROJECT FOLDER
'''
embeddingModel = SentenceTransformer(
    config.embeddingModel,
    cache_folder=config.cacheDir,
    device=config.device,
)




#-------------------------------------------------------------<Embed Documents>---------||
'''
TAKES A LIST OF Chunk OBJECTS (FROM splitter.py), RETURNS A LIST OF EMBEDDING VECTORS
IN THE SAME ORDER AS THE INPUT CHUNKS SO vectordb.py CAN ZIP THEM BACK TOGETHER
BGE MODELS RECOMMEND A "passage: " PREFIX FOR STORED TEXT, "query: " FOR SEARCH TEXT
'''
def embedDocuments(chunkList):
    textList = [chunk.text for chunk in chunkList]

    embeddingList = embeddingModel.encode(
        textList,
        batch_size=config.embeddingBatchSize,
        normalize_embeddings=config.normalizeEmbeddings,
        show_progress_bar=True,
    )

    return embeddingList


#-------------------------------------------------------------<Embed Query>-------------||
'''
SINGLE STRING IN, SINGLE VECTOR OUT
KEPT SEPARATE FROM embedDocuments() SO CALLERS DON'T HAVE TO WRAP A QUERY
IN A LIST EVERY TIME retriever.py NEEDS TO EMBED ONE
'''
def embedQuery(queryText):
    queryEmbedding = embeddingModel.encode(
        queryText,
        normalize_embeddings=config.normalizeEmbeddings,
    )

    return queryEmbedding


#-------------------------------------------------------------<Manual Test>------------||
'''
RUN "python src/embeddings.py" DIRECTLY TO VERIFY THE MODEL LOADS
AND PRODUCES VECTORS OF THE EXPECTED DIMENSION (384 FOR bge-small)
'''
if __name__ == "__main__":
    import loader
    import splitter

    loadedDocuments = loader.loadDocuments()
    allChunks = splitter.splitDocuments(loadedDocuments)

    documentEmbeddings = embedDocuments(allChunks)
    print(f"{len(allChunks)} chunk(s) -> embeddings shape {documentEmbeddings.shape}")

    sampleQueryEmbedding = embedQuery("What is attention in transformers?")
    print(f"Query embedding shape -> {sampleQueryEmbedding.shape}")
