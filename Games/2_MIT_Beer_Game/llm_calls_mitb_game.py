import os
import json
import asyncio
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import re  # for JSON parsing

# Load environment variables for API key
load_dotenv()

# Default LLM model name
MODEL_NAME: str = "gpt-4o"

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
        self.semaphore = asyncio.Semaphore(2)
        self.logger = logger
        self.total_calls = 0
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_inference_time = 0.0

    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str,
                              temperature: float = 0.7, max_tokens: int = 450):
        # Start timing
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
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
        while True:
            async with self.semaphore:
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
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
        
        # Debug: Print response structure to see if cost is in body
        print(f"🔍 [DEBUG] Response Body Keys: {list(data.keys())}")
        if "usage" in data:
            print(f"🔍 [DEBUG] Usage Keys: {list(data['usage'].keys())}")
        
        # Extract token usage and cost information
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # Extract cost from multiple possible sources
        cost = 0.0
        cost_source = "unknown"
        
        # 1. Check response headers for cost
        response_cost_header = response.headers.get("x-litellm-response-cost")
        if response_cost_header:
            try:
                cost = float(response_cost_header)
                cost_source = "response_header"
                print(f"💰 [COST] Found cost in headers: ${cost:.6f}")
            except (ValueError, TypeError):
                print(f"⚠️  [COST] Invalid cost format in headers: {response_cost_header}")
        
        # 2. Check if cost is in usage section of response body
        if cost == 0.0 and "usage" in data:
            usage_cost = usage.get("cost") or usage.get("response_cost") or usage.get("total_cost")
            if usage_cost:
                try:
                    cost = float(usage_cost)
                    cost_source = "usage_body"
                    print(f"💰 [COST] Found cost in usage body: ${cost:.6f}")
                except (ValueError, TypeError):
                    print(f"⚠️  [COST] Invalid cost format in usage: {usage_cost}")
        
        # 3. Check if cost is in top-level response body
        if cost == 0.0:
            body_cost = data.get("cost") or data.get("response_cost") or data.get("total_cost")
            if body_cost:
                try:
                    cost = float(body_cost)
                    cost_source = "response_body"
                    print(f"💰 [COST] Found cost in response body: ${cost:.6f}")
                except (ValueError, TypeError):
                    print(f"⚠️  [COST] Invalid cost format in body: {body_cost}")
        
        # 4. Fallback cost calculation if no cost found
        if cost == 0.0:
            # GPT-4o: $5.00 per 1M input tokens, $15.00 per 1M output tokens
            cost = (prompt_tokens * 5.00 / 1_000_000) + (completion_tokens * 15.00 / 1_000_000)
            cost_source = "calculated_fallback"
            print(f"💰 [COST] Using fallback calculation: ${cost:.6f} (${5.00}/1M input + ${15.00}/1M output)")
        
        # Debug: Print all response headers to see what's available
        print(f"🔍 [DEBUG] Response Headers: {dict(response.headers)}")
        print(f"🔍 [DEBUG] Cost Source: {cost_source}")
        
        # Update running totals
        self.total_calls += 1
        self.total_cost += cost
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_inference_time += inference_time
        
        # Create comprehensive metrics
        metrics = {
            "timestamp": timestamp,
            "model": model,
            "inference_time_seconds": round(inference_time, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "cost_source": cost_source,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "call_number": self.total_calls,
            "cumulative_cost": round(self.total_cost, 6),
            "cumulative_input_tokens": self.total_input_tokens,
            "cumulative_output_tokens": self.total_output_tokens,
            "cumulative_inference_time": round(self.total_inference_time, 3)
        }
        
        # Log to terminal
        print(f"\n🔍 [LLM INFERENCE METRICS]")
        print(f"   📊 Call #{self.total_calls} | Model: {model}")
        print(f"   ⏱️  Inference Time: {inference_time:.3f}s")
        print(f"   📝 Tokens: {prompt_tokens} in → {completion_tokens} out → {total_tokens} total")
        print(f"   💰 Cost: ${cost:.6f} (Total: ${self.total_cost:.6f})")
        print(f"   🎯 Temperature: {temperature} | Max Tokens: {max_tokens}")
        
        # Log to file logger if available
        if self.logger:
            self.logger.log(f"[LLM INFERENCE METRICS] {json.dumps(metrics, indent=2)}")
        
        # Save metrics to JSON file
        self._save_metrics_to_file(metrics)
        
        print(f"[LLM RAW RESPONSE]: {content}")
        if self.logger:
            self.logger.log(f"[LLM RAW RESPONSE]: {content}")
        
        return content
    
    def _save_metrics_to_file(self, metrics):
        """Save metrics to a JSON file for analysis"""
        try:
            metrics_file = "llm_inference_metrics.json"
            
            # Read existing metrics
            existing_metrics = []
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        existing_metrics = json.load(f)
                except json.JSONDecodeError:
                    existing_metrics = []
            
            # Append new metrics
            existing_metrics.append(metrics)
            
            # Write back to file
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save metrics to file: {e}")
    
    def get_session_summary(self):
        """Get a summary of all LLM calls in this session"""
        if self.total_calls == 0:
            return "No LLM calls made in this session."
        
        avg_inference_time = self.total_inference_time / self.total_calls
        avg_cost_per_call = self.total_cost / self.total_calls
        
        summary = {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_inference_time_seconds": round(self.total_inference_time, 3),
            "average_inference_time_seconds": round(avg_inference_time, 3),
            "average_cost_per_call_usd": round(avg_cost_per_call, 6)
        }
        
        return summary


lite_client = LiteLLMClient() 