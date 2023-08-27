import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "tiiuae/falcon-7b-instruct"
#repo_id = "gpt-3.5-turbo"
#repo_id = 'tiiuae/falcon-40b'

llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """
Exmaple Prompt:

My name is Abdullah. I am making the niyat for Isha Salah. The salah unit is Sunnah and total rakkah are 4.

Output:

personid = Abdullah
Namaz = Isha
Salah Unit = Sunnah
Entered_Rakah = 4 

{question}

"""

'''
@cl.langchain_factory
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    return llm_chain'''
    
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"].lower()).send()
