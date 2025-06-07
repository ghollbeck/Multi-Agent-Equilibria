import os
import json
import asyncio
import requests
from dotenv import load_dotenv
import re  # for JSON parsing

# Load environment variables for API key
load_dotenv()

# Default LLM model name
MODEL_NAME: str = "gpt-4o-mini"

# Utility for robust JSON extraction from LLM responses
def safe_parse_json(response_str: str) -> dict:
    """Extract the first JSON object in the LLM response string and parse it, handling markdown fences and incomplete JSON."""
    s = response_str.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    start = s.find('{')
    if start == -1:
        try:
            return json.loads(s)
        except Exception as e:
            print(f"❌ [safe_parse_json] Could not find '{{' in response. Error: {e}. Response: {s}")
            raise
    substring = s[start:]
    brace_count = 0
    end_index = None
    for idx, ch in enumerate(substring):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = idx + 1
                break
    if end_index is not None:
        json_text = substring[:end_index]
    else:
        json_text = substring.rstrip(', \n\r\t')
        json_text += '}' * brace_count
    try:
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ [safe_parse_json] Failed to parse JSON. Error: {e}. Extracted: {json_text}")
        raise


def parse_json_with_default(response_str: str, default: dict, context: str) -> dict:
    """Parse JSON and return default value on failure, logging the error context."""
    try:
        return safe_parse_json(response_str)
    except Exception as e:
        print(f"❌ [parse_json_with_default] Error parsing JSON in {context}: {e}. Response was: {response_str!r}")
        m = re.search(r'"order_quantity"\s*:\s*(\d+)', response_str)
        if m:
            salvaged = default.copy()
            salvaged['order_quantity'] = int(m.group(1))
            print(f"❌ [parse_json_with_default] Salvaged order_quantity={m.group(1)} in {context}.")
            return salvaged
        return default


class LiteLLMClient:
    def __init__(self, logger=None):
        self.api_key = os.getenv("LITELLM_API_KEY")
        self.endpoint = "https://litellm.sph-prod.ethz.ch/chat/completions"
        self._semaphores_by_loop = {}
        self.logger = logger

    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str,
                              temperature: float = 0.7, max_tokens: int = 450):
        print(f"[LLM SYSTEM PROMPT]: {system_prompt}")
        print(f"[LLM USER PROMPT]: {user_prompt}")
        if self.logger:
            self.logger.log(f"[LLM SYSTEM PROMPT]: {system_prompt}")
            self.logger.log(f"[LLM USER PROMPT]: {user_prompt}")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        max_retries = 5
        delay = 2
        attempt = 0
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores_by_loop.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(2)
            self._semaphores_by_loop[loop] = semaphore
        while True:
            async with semaphore:
                response = await asyncio.to_thread(requests.post, self.endpoint, json=payload, headers=headers)
            if response.ok:
                break
            elif response.status_code == 429:
                attempt += 1
                if attempt >= max_retries:
                    raise Exception(f"LiteLLM API error: {response.status_code} {response.text}")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"LiteLLM API error: {response.status_code} {response.text}")
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
        print(f"[LLM RAW RESPONSE]: {content}")
        if self.logger:
            self.logger.log(f"[LLM RAW RESPONSE]: {content}")
        return content


lite_client = LiteLLMClient() 