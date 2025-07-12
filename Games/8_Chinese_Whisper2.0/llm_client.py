import os
import json
import asyncio
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# Load .env from project root (two levels up from this script)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except (ImportError, TypeError, Exception):
    LANGSMITH_AVAILABLE = False
    def traceable(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

# Configure LangSmith
if LANGSMITH_AVAILABLE:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "chinese_whispers_sql"

# Default model
DEFAULT_MODEL = "gpt-4o-mini"

class LiteLLMClient:
    """Centralized LiteLLM client for Chinese Whispers SQL project"""
    
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
        
        if not self.api_key:
            print(f"⚠️  Warning: LITELLM_API_KEY not found in environment")
            print(f"   Checked: {env_path}")
            print(f"   Available env vars: {[k for k in os.environ.keys() if 'API' in k or 'KEY' in k]}")

    @traceable(name="llm_chat_completion")
    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str,
                              temperature: float = 0.7, max_tokens: int = 800,
                              task_type: str = None, step: int = None):
        """Make an LLM API call with comprehensive error handling and metrics"""
        
        # Start timing
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Minimal logging
        if task_type and step is not None:
            print(f"📍 [{task_type}] Step {step}")
        elif task_type:
            print(f"📍 [{task_type}]")
        
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
                try:
                    response = await asyncio.to_thread(
                        requests.post, 
                        self.endpoint, 
                        json=payload, 
                        headers=headers
                    )
                except Exception as e:
                    raise Exception(f"Network error: {e}")
                    
            if response.ok:
                break
            elif response.status_code == 429:
                attempt += 1
                if attempt >= max_retries:
                    raise Exception(f"LiteLLM API error after {max_retries} retries: {response.status_code} {response.text}")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                # For authentication errors, provide more specific feedback
                if response.status_code == 401:
                    error_text = response.text
                    api_key_preview = f"{self.api_key[:5]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 9 else "Invalid/Missing"
                    raise Exception(f"Authentication failed (401). API key: {api_key_preview}. Error: {error_text}")
                else:
                    raise Exception(f"LiteLLM API error: {response.status_code} {response.text}")
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {response.text}")
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
        
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
            except (ValueError, TypeError):
                pass
        
        # 2. Check if cost is in usage section of response body
        if cost == 0.0 and "usage" in data:
            usage_cost = usage.get("cost") or usage.get("response_cost") or usage.get("total_cost")
            if usage_cost:
                try:
                    cost = float(usage_cost)
                    cost_source = "usage_body"
                except (ValueError, TypeError):
                    pass
        
        # 3. Check if cost is in top-level response body
        if cost == 0.0:
            body_cost = data.get("cost") or data.get("response_cost") or data.get("total_cost")
            if body_cost:
                try:
                    cost = float(body_cost)
                    cost_source = "response_body"
                except (ValueError, TypeError):
                    pass
        
        # 4. Fallback cost calculation if no cost found
        if cost == 0.0:
            # GPT-4o: $5.00 per 1M input tokens, $15.00 per 1M output tokens
            cost = (prompt_tokens * 5.00 / 1_000_000) + (completion_tokens * 15.00 / 1_000_000)
            cost_source = "calculated_fallback"
        
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
            "task_type": task_type,
            "step": step
        }
        
        # Log to file logger if available
        if self.logger:
            self.logger.log(f"[LLM INFERENCE METRICS] {json.dumps(metrics, indent=2)}")
        
        # Save metrics to JSON file
        self._save_metrics_to_file(metrics)
        
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
            pass  # Silently fail to avoid disrupting main workflow
    
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


# Global client instance
lite_client = LiteLLMClient() 