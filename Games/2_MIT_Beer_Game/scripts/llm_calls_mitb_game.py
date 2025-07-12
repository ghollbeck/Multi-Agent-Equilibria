import os
import json
import asyncio
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import re  # for JSON parsing

# Force disable LangSmith to avoid rate limit issues
LANGSMITH_AVAILABLE = False

def traceable(name=None, **kwargs):
    def decorator(func):
        return func
    return decorator

# Commented out LangSmith import to prevent any tracing
# try:
#     from langsmith import traceable
#     LANGSMITH_AVAILABLE = True
#     # print("✓ LangSmith successfully imported and available for tracing")  # Commented out
# except (ImportError, TypeError, Exception) as e:
#     # print(f"Warning: LangSmith not available ({type(e).__name__}: {str(e)[:100]}...), running without tracing")  # Commented out
#     LANGSMITH_AVAILABLE = False
#     def traceable(name=None, **kwargs):
#         def decorator(func):
#             return func
#         return decorator

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
            # print(f"❌ [safe_parse_json] Could not find '{{' in response. Error: {e}. Response: {s}")  # Commented out
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
        # print(f"❌ [safe_parse_json] Failed to parse JSON. Error: {e}. Extracted: {json_text}")  # Commented out
        raise


def parse_json_with_default(response_str: str, default: dict, context: str) -> dict:
    """Parse JSON and return default value on failure, logging the error context."""
    try:
        return safe_parse_json(response_str)
    except Exception as e:
        # print(f"❌ [parse_json_with_default] Error parsing JSON in {context}: {e}. Response was: {response_str!r}")  # Commented out
        m = re.search(r'"order_quantity"\s*:\s*(\d+)', response_str)
        if m:
            salvaged = default.copy()
            salvaged['order_quantity'] = int(m.group(1))
            # print(f"❌ [parse_json_with_default] Salvaged order_quantity={m.group(1)} in {context}.")  # Commented out
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
        
        # Configure LangSmith project if available
        if LANGSMITH_AVAILABLE:
            os.environ.setdefault("LANGSMITH_PROJECT", "MIT_beer_game_Langsmith")
            # print("✓ LangSmith project configured as MIT_beer_game_Langsmith")  # Commented out

    @traceable(name="llm_chat_completion")
    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str,
                              temperature: float = 0.7, max_tokens: int = 800,
                              agent_role: str = None, round_index: int = None, 
                              decision_type: str = None):
        # Start timing
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Minimal logging - only show action being performed
        if agent_role and decision_type:
            if decision_type == "strategy_initialization":
                print(f"🔥 [{agent_role}] {decision_type}")
            else:
                print(f"✅ [{agent_role}] {decision_type}")
        
        # print(f"[LLM SYSTEM PROMPT]: {system_prompt}")  # Commented out
        # print(f"[LLM USER PROMPT]: {user_prompt}")  # Commented out
        # if self.logger:
        #     self.logger.log(f"[LLM SYSTEM PROMPT]: {system_prompt}")
        #     self.logger.log(f"[LLM USER PROMPT]: {user_prompt}")
        
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
        # print(f"🔍 [DEBUG] Response Body Keys: {list(data.keys())}")  # Commented out
        # if "usage" in data:  # Commented out
        #     print(f"🔍 [DEBUG] Usage Keys: {list(data['usage'].keys())}")  # Commented out
        
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
                # print(f"💰 [COST] Found cost in headers: ${cost:.6f}")  # Commented out
            except (ValueError, TypeError):
                pass  # print(f"⚠️  [COST] Invalid cost format in headers: {response_cost_header}")  # Commented out
        
        # 2. Check if cost is in usage section of response body
        if cost == 0.0 and "usage" in data:
            usage_cost = usage.get("cost") or usage.get("response_cost") or usage.get("total_cost")
            if usage_cost:
                try:
                    cost = float(usage_cost)
                    cost_source = "usage_body"
                    # print(f"💰 [COST] Found cost in usage body: ${cost:.6f}")  # Commented out
                except (ValueError, TypeError):
                    pass  # print(f"⚠️  [COST] Invalid cost format in usage: {usage_cost}")  # Commented out
        
        # 3. Check if cost is in top-level response body
        if cost == 0.0:
            body_cost = data.get("cost") or data.get("response_cost") or data.get("total_cost")
            if body_cost:
                try:
                    cost = float(body_cost)
                    cost_source = "response_body"
                    # print(f"💰 [COST] Found cost in response body: ${cost:.6f}")  # Commented out
                except (ValueError, TypeError):
                    pass  # print(f"⚠️  [COST] Invalid cost format in body: {body_cost}")  # Commented out
        
        # 4. Fallback cost calculation if no cost found
        if cost == 0.0:
            # GPT-4o: $5.00 per 1M input tokens, $15.00 per 1M output tokens
            cost = (prompt_tokens * 5.00 / 1_000_000) + (completion_tokens * 15.00 / 1_000_000)
            cost_source = "calculated_fallback"
            # print(f"💰 [COST] Using fallback calculation: ${cost:.6f} (${5.00}/1M input + ${15.00}/1M output)")  # Commented out
        
        # Debug: Print all response headers to see what's available
        # print(f"🔍 [DEBUG] Response Headers: {dict(response.headers)}")  # Commented out
        # print(f"🔍 [DEBUG] Cost Source: {cost_source}")  # Commented out
        
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
            "cumulative_inference_time": round(self.total_inference_time, 3),
            "agent_role": agent_role,
            "round_index": round_index,
            "decision_type": decision_type
        }
        
        # Log to terminal - simplified version
        # print(f"\n🔍 [LLM INFERENCE METRICS]")  # Commented out
        # print(f"   📊 Call #{self.total_calls} | Model: {model}")  # Commented out
        # print(f"   ⏱️  Inference Time: {inference_time:.3f}s")  # Commented out
        # print(f"   📝 Tokens: {prompt_tokens} in → {completion_tokens} out → {total_tokens} total")  # Commented out
        # print(f"   💰 Cost: ${cost:.6f} (Total: ${self.total_cost:.6f})")  # Commented out
        # print(f"   🎯 Temperature: {temperature} | Max Tokens: {max_tokens}")  # Commented out
        
        # Log to file logger if available
        if self.logger:
            self.logger.log(f"[LLM INFERENCE METRICS] {json.dumps(metrics, indent=2)}")
        
        # Save metrics to JSON file
        self._save_metrics_to_file(metrics)
        
        # print(f"[LLM RAW RESPONSE]: {content}")  # Commented out
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
            pass  # print(f"⚠️  Warning: Could not save metrics to file: {e}")  # Commented out
    
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