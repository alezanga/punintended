import json
import logging
import sys
from typing import Optional

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, after_log, \
    retry_if_not_exception_type, wait_random_exponential

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


@retry(
    retry=(retry_if_not_exception_type(openai.RateLimitError) |
           retry_if_exception_type(json.JSONDecodeError) |
           retry_if_exception_type(openai.InternalServerError)),
    wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
    after=after_log(logger, logging.WARNING)
)
def call_api(client, system_prompt, user_prompt, model, max_tokens: Optional[int], temperature: float,
             return_body: bool = False):
    # NOTE: here require manual edit. I changed it for t5 and GPT
    body = {
        # "response_format": {
        #     "type": "json_object"
        # },
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "model": model,
        "max_completion_tokens": max_tokens,
        "logprobs": True
    }
    if model != "o3-mini":
        body["temperature"] = temperature

    if return_body:
        return body

    chat_completion = client.chat.completions.create(**body)

    out = json.loads(chat_completion.choices[0].message.content)
    return out
