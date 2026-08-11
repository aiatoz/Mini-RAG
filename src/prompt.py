import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#-----------------------------------------------------------<Format Context>------------||
'''
TURNS THE LIST OF RETRIEVED CHUNKS (FROM retriever.retrieve()) INTO A
NUMBERED BLOCK OF TEXT, MATCHING THE "Retrieved Chunk 1 / 2 / 3" LAYOUT
FROM THE DESIGN DOC. KEPT SEPARATE FROM buildPrompt() SO IT CAN BE REUSED
IF WE EVER WANT TO SHOW THE RAW CONTEXT SOMEWHERE ELSE (e.g. A DEBUG UI)
'''
def formatContext(retrievedChunks):
    contextBlockList = []

    for rank, chunk in enumerate(retrievedChunks, start=1):
        contextBlockList.append(f"Retrieved Chunk {rank}:\n{chunk['text']}")

    return "\n\n".join(contextBlockList)


#-----------------------------------------------------------<Build Prompt>--------------||
'''
ASSEMBLES THE FINAL PROMPT STRING SENT TO THE LLM:
System Instructions -> Context (numbered chunks) -> Question

KEPT INDEPENDENT FROM retriever.py AND generator.py, PER THE DESIGN DOC,
SO PROMPT FORMAT CAN BE TWEAKED WITHOUT TOUCHING EITHER OF THOSE
'''
def buildPrompt(queryText, retrievedChunks):
    contextBlock = formatContext(retrievedChunks)

    promptText = (
        f"{config.systemInstruction}\n\n"
        f"Context:\n{contextBlock}\n\n"
        f"Question:\n{queryText}"
    )

    return promptText

#-----------------------------------------------------------<Manual Test>---------------||
'''
RUN "python src/prompt.py" DIRECTLY TO VERIFY PROMPT ASSEMBLY
AGAINST WHATEVER IS CURRENTLY IN THE CHROMA COLLECTION
(RUN scripts/build_index.py FIRST IF THE COLLECTION IS EMPTY)
'''
if __name__ == "__main__":
    import retriever

    testQuery = "What is attention in transformers?"
    retrievedChunks = retriever.retrieve(testQuery, topK=2)

    finalPrompt = buildPrompt(testQuery, retrievedChunks)

    print("---- FINAL PROMPT ----")
    print(finalPrompt)
