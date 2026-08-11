import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#------------------------------------------------------<Chunk Object>-------------||
'''
ONE PIECE OF A Document AFTER SPLITTING
CARRIES source / documentId FORWARD SO CITATIONS WORK LATER WITHOUT
GOING BACK TO THE ORIGINAL FILE
'''
class Chunk:
    def __init__(self, text, source, documentId, chunkId):
        self.text = text
        self.source = source
        self.documentId = documentId
        self.chunkId = chunkId


    '''
    MATCHES THE METADATA SHAPE FROM THE DESIGN DOC
    vectordb.py WILL STORE THIS ALONGSIDE THE EMBEDDING
    '''
    def toMetadata(self):
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunkId,
            "document_id": self.documentId,
        }


#------------------------------------------------------<Splitter Instance>--------||
'''
BUILT ONCE FROM config VALUES SO chunkSize / chunkOverlap NEVER GET
HARDCODED HERE. SEPARATORS FALL BACK IN ORDER: PARAGRAPH -> LINE -> SENTENCE -> WORD
'''
recursiveSplitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunkSize,
    chunk_overlap=config.chunkOverlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


#-------------------------------------------------------<Split Single Document>----||
'''
RUNS ONE Document THROUGH THE SPLITTER AND WRAPS EACH PIECE INTO A Chunk
chunkId IS LOCAL TO THE DOCUMENT (0, 1, 2, ...), NOT GLOBAL ACROSS THE WHOLE CORPUS
'''
def splitDocument(document):
    rawPieces = recursiveSplitter.split_text(document.content)

    chunkList = []
    for chunkIndex, piece in enumerate(rawPieces):
        chunk = Chunk(
            text=piece,
            source=document.source,
            documentId=document.documentId,
            chunkId=chunkIndex,
        )
        chunkList.append(chunk)

    return chunkList


#------------------------------------------------------<Split Multiple Documents>-||

'''
CONVENIENCE WRAPPER FOR THE INDEXING PIPELINE (build_index.py)
FLATTENS CHUNKS FROM ALL DOCUMENTS INTO ONE LIST
'''
def splitDocuments(documentList):
    allChunks = []

    for document in documentList:
        allChunks.extend(splitDocument(document))

    return allChunks




#------------------------------------------------------<Manual Test>--------------||
'''
RUN "python src/splitter.py" DIRECTLY TO VERIFY CHUNKING BEHAVIOUR
BEFORE WIRING INTO THE EMBEDDING STEP
'''
if __name__ == "__main__":
    import loader

    loadedDocuments = loader.loadDocuments()
    allChunks = splitDocuments(loadedDocuments)

    print(f"Loaded {len(loadedDocuments)} document(s) -> {len(allChunks)} chunk(s)")

    for chunk in allChunks:
        preview = chunk.text[:60].replace("\n", " ")
        print(f"  - {chunk.source} chunk#{chunk.chunkId} -> {preview}...")
