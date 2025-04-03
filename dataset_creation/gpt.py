import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
load_dotenv()


async def openai_gpt4o_async(prompt, max_tokens=999, output_json=True, sys_prompt="You are an Assistant to Output in a JSON with keys 'response'."):
    """use GPT-4o with Json output!
    """
    model_name = "gpt-4o"
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    if output_json:
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            response_format={ "type": "json_object" }, # use json mode
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    res = response.choices[0].message.content.strip()
    return res


async def openai_gpt4o_mini_async(prompt, max_tokens=9999, output_json=True, ):
    model_name = "gpt-4o-mini"
    client = AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    print(client)

    if output_json:
        sys_prompt="You are an Assistant to Output in a JSON with keys 'response'."
        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            response_format={ "type": "json_object" }, # use json mode
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
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
    print(res)
    return res


async def openai_gpt4o_async_token(prompt, max_tokens=999, output_json=True, sys_prompt="You are an Assistant to Output in a JSON with keys 'response'."):
    """use GPT-4o with Json output!
    """
    model_name = "gpt-4o"
    client = AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    if output_json:
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            response_format={ "type": "json_object" }, # use json mode
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    
    res = response.choices[0].message.content.strip()
    if response.usage:
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    else:
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Default if usage is missing

    return res, tokens


async def openai_gpt4o_mini_async_token(prompt, max_tokens=9999, output_json=True, ):
    """use GPT-4o mini with Json output!
    """
    model_name = "gpt-4o-mini"
    client = AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY_US2"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_US2"), 
        api_version="2023-09-01-preview"
    )
    print(client)

    if output_json:
        sys_prompt="You are an Assistant to Output in a JSON with keys 'response'."
        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await client.chat.completions.create(
            model=model_name,  
            messages=msg,
            response_format={ "type": "json_object" }, # use json mode
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
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
    if response.usage:
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    else:
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Default if usage is missing

    return res, tokens