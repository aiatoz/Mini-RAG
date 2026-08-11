import os
import torch

#--------------------------------------------------------------<Base Paths>-----------||
'''
ROOT OF THE PROJECT. EVERY OTHER PATH IS BUILT RELATIVE TO THIS
SO THE PROJECT STAYS PORTABLE ACROSS MACHINES
'''
baseDir = os.path.dirname(os.path.abspath(__file__))

dataDir = os.path.join(baseDir, "data")
cacheDir = os.path.join(baseDir, "cache", "huggingface")
dbDir = os.path.join(baseDir, "db")
chromaDir = os.path.join(dbDir, "chroma")
bm25Path = os.path.join(dbDir, "bm25.pkl")
metadataPath = os.path.join(dbDir, "metadata.json")



#--------------------------------------------------------------<Device>---------------||
'''
AUTO DETECT CUDA. FALLS BACK TO CPU IF GPU IS NOT AVAILABLE
(e.g. RUNNING ON A DIFFERENT MACHINE)
'''
device = "cuda" if torch.cuda.is_available() else "cpu"





#--------------------------------------------------------------<Document Loader>------||
supportedExtensions = [".md", ".txt"]





#--------------------------------------------------------------<Chunking>-------------||
'''
RECURSIVE CHARACTER SPLITTING FOR PHASE 1
chunkSize / chunkOverlap ARE IN CHARACTERS, NOT TOKENS
'''
chunkingStrategy = "recursive"          # recursive | semantic (future)
chunkSize = 800
chunkOverlap = 120



#--------------------------------------------------------------<Embedding Model>------||
embeddingModel = "BAAI/bge-small-en-v1.5"
embeddingBatchSize = 32
normalizeEmbeddings = True



#--------------------------------------------------------------<Vector DB>------------||
collectionName = "miniRagCollection"
distanceMetric = "cosine"



#--------------------------------------------------------------<Retrieval>------------||
retrievalStrategy = "dense"             # dense | hybrid (future)
topK = 5



#--------------------------------------------------------------<Local LLM>------------||
'''
QUANTIZED 4-BIT LOAD KEEPS THIS UNDER THE 8GB VRAM BUDGET ON THE 4060
SWAP llmModel TO SWITCH BETWEEN CANDIDATES WITHOUT TOUCHING generator.py
'''
llmModel = "microsoft/Phi-3.5-mini-instruct"
loadIn4Bit = True
temperature = 0.3
maxNewTokens = 512




#--------------------------------------------------------------<Prompt>---------------||
'''
SYSTEM INSTRUCTION IS KEPT OUT OF THIS FILE ENTIRELY, LIVES IN
prompts/system_prompt.txt SO IT CAN BE EDITED / VERSIONED / SWAPPED
WITHOUT TOUCHING PYTHON CODE. READ ONCE AT IMPORT TIME, SAME AS
EVERY OTHER MODULE THAT LOADS SOMETHING ONCE (embeddings.py, generator.py)
'''
promptsDir = os.path.join(baseDir, "prompts")
systemPromptPath = os.path.join(promptsDir, "system_prompt.txt")

with open(systemPromptPath, "r", encoding="utf-8") as systemPromptFile:
    systemInstruction = systemPromptFile.read().strip()
