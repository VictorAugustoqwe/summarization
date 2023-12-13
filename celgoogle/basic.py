import transformers
import torch
from langchain.prompts import PromptTemplate
import os

def runPromptsAndSaveResultsInDirectory(inputDirectoryPath, outputDirectoryPath):
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    template = """
    {environmentalReport}

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

    prompt = PromptTemplate(input_variables=["environmentalReport"], template=template)


    files = os.listdir(inputDirectoryPath)

    for filename in files:
        filepath = inputDirectoryPath + "/" + filename
        print('filepath - ',filepath)

        environmentalReport =  None
        with open(filepath, 'r') as file:
            environmentalReport = file.read()

        environmentalReportPrompt = prompt.format(environmentalReport=environmentalReport)

        sequences = pipeline(
            environmentalReportPrompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

        ending = "Using the information from the text, write a detailed summary for each of the seven categories above. Be sure to include specific data and statistics where available."

        for sequence in sequences:
            if ending in sequence['generated_text']:
                result = (sequence['generated_text'].split(ending))[1]
                writefilename = outputDirectoryPath + '/' + filename.split('.')[0] + '-res.txt' 
                print(result)
                print(writefilename)
                with open(writefilename, 'w') as file:
                    file.write(result)

runPromptsAndSaveResultsInDirectory('txts','output3')
