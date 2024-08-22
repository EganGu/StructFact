import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def get_client():
    return AsyncOpenAI(
        api_key="your_api",
        base_url='your_base_url',
    )

async def fetch_response(args, client, system, prompt, semaphore):
    async with semaphore:

        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=args.model_name_or_path,
            temperature=args.temperature,
            logprobs=getattr(args, 'logprobs', None),
            n = getattr(args, 'n', 1)
        )
        return response

async def generate(args, client, sys_prompt, prompts, return_raw=False):

    '''  
    # 异步读入
    async with aiofiles.open('examples.txt', 'r') as f:
        queries = [line.strip() for line in await f.readlines()]
    '''

    # Semaphore to limit concurrent requests to 10
    semaphore = asyncio.Semaphore(40)

    # Create tasks for all queries with semaphore
    tasks = [fetch_response(args, client, sys_prompt, prompt, semaphore) for prompt in tqdm_asyncio(prompts)]

    # Gather responses in order
    responses = await tqdm_asyncio.gather(*tasks, desc="Processing", total=len(tasks))

    # Ensure all responses are valid
    ensure_response = []
    for i, response in enumerate(responses):
        if response.choices is not None:
            ensure_response.append(response)
        else:
            tasks = [fetch_response(args, client, sys_prompt, prompts[i], semaphore)]
            _responses = await asyncio.gather(*tasks)
            while _responses[0].choices is None:
                _responses = await asyncio.gather(*tasks)
            ensure_response.append(_responses[0])

    if return_raw:
        return ensure_response

    return [response.choices[0].message.content for response in responses]

