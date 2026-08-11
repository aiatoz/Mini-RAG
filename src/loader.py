import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#------------------------------------------------<Document Object>---------------------||
'''
SIMPLE CONTAINER FOR A LOADED FILE. ONE Document PER FILE, NOT PER CHUNK
splitter.py WILL LATER BREAK content INTO Chunks WHILE CARRYING THIS METADATA FORWARD
'''
class Document:
    def __init__(self, content, source, documentId):
        self.content = content            # raw text of the whole file
        self.source = source              # filename, used later for citations
        self.documentId = documentId      # filename without extension, unique per doc


#-----------------------------------------------<Read Single File>---------------------||
'''
READS ONE FILE FROM DISK AND WRAPS IT INTO A Document
SKIPS EMPTY FILES SO THEY DON'T POLLUTE THE INDEX LATER
'''
def readFile(filePath):
    with open(filePath, "r", encoding="utf-8") as fileHandle:
        content = fileHandle.read().strip()

    if content == "":
        return None

    fileName = os.path.basename(filePath)
    documentId = os.path.splitext(fileName)[0]

    return Document(content=content, source=fileName, documentId=documentId)


#------------------------------------------------<Scan Directory>----------------------||
'''
WALKS folderPath RECURSIVELY (data/ml_notes, data/rag, ETC ALL GET PICKED UP)
ONLY FILES WITH AN EXTENSION LISTED IN config.supportedExtensions ARE LOADED
'''
def loadDocuments(folderPath=None):
    if folderPath is None:
        folderPath = config.dataDir

    documentList = []

    for rootPath, dirNames, fileNames in os.walk(folderPath):
        for fileName in fileNames:
            fileExtension = os.path.splitext(fileName)[1].lower()

            if fileExtension not in config.supportedExtensions:
                continue

            filePath = os.path.join(rootPath, fileName)
            document = readFile(filePath)

            if document is not None:
                documentList.append(document)

    return documentList



#------------------------------------------------<Manual Test>-------------------------||
'''
QUICK SANITY CHECK. RUN "python src/loader.py" DIRECTLY TO VERIFY
DOCUMENTS ARE BEING PICKED UP FROM data/ BEFORE WIRING INTO THE REST OF THE PIPELINE
'''
if __name__ == "__main__":
    loadedDocuments = loadDocuments()

    print(f"Loaded {len(loadedDocuments)} document(s)")

    for document in loadedDocuments:
        preview = document.content[:80].replace("\n", " ")
        print(f"  - {document.source} (id={document.documentId}) -> {preview}...")
