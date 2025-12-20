import aiohttp

from secret import T2I_API_KEY, T2I_URL, T2I_MODEL


async def generate(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {T2I_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": T2I_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "size": "768*768"
        }
    }
    client = aiohttp.ClientSession()
    response = await client.post(T2I_URL, headers=headers, json=body)
    response_json = await response.json()
    await client.close()
    if "output" not in response_json:
        return ""
    return response_json["output"]["choices"][0]["message"]["content"][0]["image"]