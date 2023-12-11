from transformers import AutoTokenizer
import transformers
import torch
from langchain.prompts import PromptTemplate
import sys


model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


template = """
{text}

Given the text above, create a summary focusing on the following seven categories:

1. Energy Efficiency
2. Recycled and Renewable Content
3. Longevity and Durability
4. Emission Reduction Initiatives
5. Material Use
6. Packaging
7. Ethical Sourcing and Supplier Code of Conduct


"""

prompt = PromptTemplate(input_variables=["text"], template=template)

# with open('iPhone_15_Pro_and_iPhone_15_Pro_Max_Sept2023.txt', 'r') as poem:
#     text = poem.read()

text = "this product is totally recyclable"

poem_prompt = prompt.format(text=text)

sequences = pipeline(
    poem_prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
