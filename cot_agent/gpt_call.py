import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
load_dotenv()


async def openai_gpt4o_async(prompt, max_tokens=999):
    """use GPT-4o with Json output!
    """
    model_name = "gpt-4o"
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    msg = [
        {"role": "user", "content": prompt},
    ]
    
    response = await client.chat.completions.create(
        model=model_name,  
        messages=msg,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    res = response.choices[0].message.content.strip()
    return res


async def openai_gpt4o_mini_async(prompt, max_tokens=9999):
    """use GPT-4o mini with Json output!
    """
    model_name = "gpt-4o-mini"
    client = AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    print(client)

    
    sys_prompt="You are an Assistant."
    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await client.chat.completions.create(
        model=model_name,  
        messages=msg,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    res = response.choices[0].message.content.strip()

    return res


