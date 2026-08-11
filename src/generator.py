import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#-----------------------------------------------------------<Quantization Config>----------||
'''
4-BIT NF4 QUANTIZATION SO llmModel FITS ALONGSIDE bge-small IN THE 4060's
8GB VRAM BUDGET. ONLY BUILT WHEN config.loadIn4Bit IS True AND WE'RE ON CUDA,
CPU FALLBACK LOADS FULL PRECISION SINCE bitsandbytes NEEDS A GPU
'''
quantizationConfig = None
if config.loadIn4Bit and config.device == "cuda":
    quantizationConfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


#------------------------------------------------------------<Model & Tokenizer Loading>----||
'''
LOADED ONCE AT MODULE IMPORT TIME, SAME PATTERN AS embeddings.py
device_map="auto" LETS accelerate PLACE LAYERS ON GPU AUTOMATICALLY
cache_dir KEEPS WEIGHTS INSIDE THE PROJECT FOLDER, NOT ~/.cache

trust_remote_code=False: CURRENT transformers VERSIONS HAVE Phi-3 SUPPORT
BUILT IN NATIVELY, SO REMOTE CODE ISN'T ACTUALLY NEEDED. SETTING THIS
EXPLICITLY (RATHER THAN LEAVING IT AMBIGUOUS) SKIPS THE INTERACTIVE
CONFIRMATION PROMPT ENTIRELY -- WHICH MATTERS ON WINDOWS SINCE THAT
PROMPT'S FALLBACK RELIES ON signal.SIGALRM, WHICH WINDOWS DOESN'T HAVE --
AND ALSO AVOIDS A KNOWN BUG WHERE THE DOWNLOADED REMOTE MODULE PATH
BREAKS ON THE LITERAL "." IN "Phi-3.5-mini-instruct"
'''
tokenizer = AutoTokenizer.from_pretrained(
    config.llmModel,
    cache_dir=config.cacheDir,
    trust_remote_code=False,
)

llmModel = AutoModelForCausalLM.from_pretrained(
    config.llmModel,
    cache_dir=config.cacheDir,
    quantization_config=quantizationConfig,
    device_map="auto" if config.device == "cuda" else None,
    dtype=torch.float16 if config.device == "cuda" else torch.float32,
    trust_remote_code=False,
)

#------------------------------------------------------------<Generate>---------------------||
'''
TAKES THE FINAL PROMPT STRING (FROM prompt.buildPrompt()), RETURNS THE
MODEL'S ANSWER AS PLAIN TEXT. USES THE MODEL'S CHAT TEMPLATE SO INSTRUCT
MODELS (Phi / Qwen) FOLLOW THE PROMPT PROPERLY INSTEAD OF JUST
CONTINUING IT LIKE A BASE MODEL WOULD

promptText IS SENT AS A SINGLE "user" TURN, SINCE system INSTRUCTIONS ARE
ALREADY BAKED INTO promptText BY prompt.py
'''
def generate(promptText):
    chatMessages = [{"role": "user", "content": promptText}]

    #-- return_dict=True: NEWER transformers RETURNS A BatchEncoding (dict-like)
    #-- INSTEAD OF A RAW TENSOR. MUST BE UNPACKED WITH ** INTO generate(),
    #-- NOT PASSED POSITIONALLY, OR generate() TRIES TO READ .shape ON A DICT
    modelInputs = tokenizer.apply_chat_template(
        chatMessages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(llmModel.device)

    inputLength = modelInputs["input_ids"].shape[-1]

    outputIds = llmModel.generate(
        **modelInputs,
        max_new_tokens=config.maxNewTokens,
        temperature=config.temperature,
        do_sample=config.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generatedIds = outputIds[0][inputLength:]
    answerText = tokenizer.decode(generatedIds, skip_special_tokens=True)

    return answerText.strip()

#------------------------------------------------------------<Manual Test>------------------||
'''
RUN "python src/generator.py" DIRECTLY TO VERIFY THE MODEL LOADS AND
GENERATES A REASONABLE ANSWER FROM RETRIEVED CONTEXT
(RUN scripts/build_index.py FIRST IF THE COLLECTION IS EMPTY)
'''
if __name__ == "__main__":
    import prompt as promptModule
    import retriever

    testQuery = "What is attention in transformers?"
    retrievedChunks = retriever.retrieve(testQuery, topK=2)
    finalPrompt = promptModule.buildPrompt(testQuery, retrievedChunks)

    print("Generating answer...")
    answerText = generate(finalPrompt)

    print(f"\nQuestion: {testQuery}")
    print(f"Answer: {answerText}")
