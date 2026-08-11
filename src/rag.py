import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generator
import prompt as promptModule
import retriever


#--------------------------------------------------<Format Sources>---------------------||
'''
TURNS retrievedChunks INTO A BULLET LIST OF SOURCES FOR THE FINAL OUTPUT,
MATCHING THE "Sources" LAYOUT FROM THE DESIGN DOC:
    • transformers.md (Chunk 8)
    • attention.md (Chunk 3)

DEDUPES SAME source+chunk_id PAIRS IN CASE THE SAME CHUNK SOMEHOW SHOWS UP
TWICE (SHOULDN'T HAPPEN WITH DENSE RETRIEVAL, BUT HYBRID MERGING LATER COULD)
'''
def formatSources(retrievedChunks):
    seenPairs = set()
    sourceLines = []

    for chunk in retrievedChunks:
        source = chunk["metadata"]["source"]
        chunkId = chunk["metadata"]["chunk_id"]
        pairKey = (source, chunkId)

        if pairKey in seenPairs:
            continue

        seenPairs.add(pairKey)
        sourceLines.append(f"• {source} (Chunk {chunkId})")

    return "\n".join(sourceLines)


#--------------------------------------------------<Ask>--------------------------------||
'''
FULL ONLINE PIPELINE FROM THE DESIGN DOC:
Question -> Retriever -> Prompt Builder -> Generator -> Response

THIS IS THE ONE FUNCTION THE REST OF THE APP (query.py, A FUTURE WEB UI, ETC)
SHOULD CALL. NO BUSINESS LOGIC HERE BEYOND COORDINATING THE OTHER MODULES

RETURNS A DICT INSTEAD OF A PRINTED STRING SO CALLERS CAN DECIDE HOW TO
DISPLAY IT (CLI, WEB UI, JSON API, ETC)
'''
def ask(queryText, topK=None):
    retrievedChunks = retriever.retrieve(queryText, topK=topK)
    finalPrompt = promptModule.buildPrompt(queryText, retrievedChunks)
    answerText = generator.generate(finalPrompt)
    sourcesText = formatSources(retrievedChunks)

    return {
        "question": queryText,
        "answer": answerText,
        "sources": sourcesText,
        "chunks": retrievedChunks,
    }


#--------------------------------------------------<Manual Test>------------------------||
'''
RUN "python src/rag.py" DIRECTLY FOR A FULL END-TO-END SANITY CHECK
(RUN scripts/build_index.py FIRST IF THE COLLECTION IS EMPTY)
'''
if __name__ == "__main__":
    testQuery = "What is attention in transformers?"
    result = ask(testQuery)

    print(f"Question: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources:\n{result['sources']}")
