from transformers import AutoTokenizer
import transformers
import torch
from langchain.prompts import PromptTemplate
import sys


model = "meta-llama/Llama-2-7b-chat-hf"


files = ['pixel-2-100417.txt', 'pixel-3xl-100918.txt', 'pixel-4xl-102419.txt', 'pixel-6a-072022.txt', 'pixel-8-102023.txt',
'pixel-2xl-100417.txt', 'pixel-4-102419.txt', 'pixel-5-102020.txt', 'pixel-7-102022.txt', 'pixel-8-pro-102023.txt',
'pixel-3-100918.txt', 'pixel-4a-102020.txt', 'pixel-5a-with-5g-082021.txt', 'pixel-7a-052023.txt', 'pixel-fold-062023.txt',
'pixel-3a-100918.txt', 'pixel-4a-5g-102020.txt', 'pixel-6-102021.txt', 'pixel-7-pro-102022.txt']

files = files[:3]

for filename in files:
    print('filename - ',filename)

    tokenizer = None
    pipeline = None
    template = None
    prompt = None
    poem_prompt = None
    sequences = None

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )


    template = """
    {text}

    Categories:
    1. Energy Efficiency
    2. Recycled and Renewable Content
    3. Longevity and Durability
    4. Emission Reduction Initiatives
    5. Material Use
    6. Packaging
    7. Ethical Sourcing and Supplier Code of Conduct

    Using the information from the text, write a detailed summary for each of the seven categories above. Be sure to include specific data and statistics where available.

    """

    prompt = PromptTemplate(input_variables=["text"], template=template)

    text = None
    with open(filename, 'r') as file:
        text = file.read()

    poem_prompt = prompt.format(text=text)

    sequences = pipeline(
        poem_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    ending = "Using the information from the text, write a detailed summary for each of the seven categories above. Be sure to include specific data and statistics where available."

    for seq in sequences:
        if ending in seq['generated_text']:
            result = (seq['generated_text'].split(ending))[1]
            writefilename = 'output2/' + filename.split('.')[0] + '-res.txt' 
            print(result)
            print(writefilename)
            with open(writefilename, 'w') as file:
                file.write(result)
        else:
            print(f"Result: {seq['generated_text']}")
