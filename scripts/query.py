import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rag


#---------------------------------------<Print Result>-------------------------||
'''
FORMATS ONE rag.ask() RESULT FOR THE TERMINAL
KEPT SEPARATE FROM rag.ask() SO THE CLI'S DISPLAY FORMAT DOESN'T LEAK
INTO THE CORE PIPELINE FUNCTION
'''
def printResult(result):
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Sources:\n{result['sources']}\n")


#---------------------------------------<Interactive Loop>---------------------||
'''
SIMPLE REPL: TYPE A QUESTION, GET AN ANSWER, REPEAT
TYPE "exit" OR "quit" TO STOP, OR Ctrl+C
'''
def runInteractive():
    print("Mini RAG — ask a question (type 'exit' to quit)\n")

    while True:
        queryText = input("> ").strip()

        if queryText.lower() in ("exit", "quit"):
            break

        if queryText == "":
            continue

        result = rag.ask(queryText)
        printResult(result)


#---------------------------------------<Entrypoint>---------------------------||
'''
SUPPORTS TWO MODES:
    python scripts/query.py                          -> INTERACTIVE LOOP
    python scripts/query.py "What is attention?"      -> SINGLE ONE-OFF QUESTION
'''
if __name__ == "__main__":
    if len(sys.argv) > 1:
        oneOffQuery = " ".join(sys.argv[1:])
        result = rag.ask(oneOffQuery)
        printResult(result)
    else:
        runInteractive()
