import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import embeddings
import loader
import splitter
import vectordb


#---------------------------------------------------------<Build Index>---------------------||
'''
OFFLINE PIPELINE FROM THE DESIGN DOC:
Documents -> Loader -> Chunking -> Embedding Model -> ChromaDB -> Persistent Vector Store

RUN THIS WHENEVER data/ CHANGES (NEW FILES ADDED / EXISTING FILES EDITED)
NOT PART OF THE ONLINE QUERY PATH, THAT LIVES IN query.py / rag.py
'''
def buildIndex():
    print("Loading documents...")
    loadedDocuments = loader.loadDocuments()
    print(f"  -> {len(loadedDocuments)} document(s) loaded")

    if len(loadedDocuments) == 0:
        print("No documents found in data/. Add .md or .txt files and re-run.")
        return

    print("Splitting into chunks...")
    allChunks = splitter.splitDocuments(loadedDocuments)
    print(f"  -> {len(allChunks)} chunk(s) created")

    print("Generating embeddings...")
    documentEmbeddings = embeddings.embedDocuments(allChunks)

    print("Writing to ChromaDB...")
    vectordb.add(allChunks, documentEmbeddings)
    vectordb.persist()

    print(f"Done. Collection '{vectordb.collection.name}' now has {vectordb.collection.count()} chunk(s) total.")

#---------------------------------------------------------<Entrypoint>----------------------||
if __name__ == "__main__":
    buildIndex()
