import boto3
import json
import sys
import time
from botocore.exceptions import ClientError
import concurrent.futures
from datetime import datetime
import threading

# --- Configuration ---
REGION_NAME = "us-west-2"
# Using Claude Sonnet 4.5 V1 for higher rate limits (500k tokens/min)
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Pricing for Claude Sonnet 4.5 V1 (approximate, per 1,000 tokens)
PRICE_INPUT_PER_1K = 0.003
PRICE_OUTPUT_PER_1K = 0.015

# Initialize Bedrock Client
bedrock = boto3.client("bedrock-runtime", region_name=REGION_NAME)

# Rate limiter to stay within 500k tokens/min quota
class RateLimiter:
    def __init__(self, tokens_per_minute=500000, buffer_percent=0.8):
        """
        Args:
            tokens_per_minute: Max tokens per minute allowed
            buffer_percent: Use only this fraction of quota (0.8 = 80% to be safe)
        """
        self.max_tokens_per_min = int(tokens_per_minute * buffer_percent)
        self.tokens_used = 0
        self.window_start = time.time()
        self.lock = threading.Lock()

    def wait_if_needed(self, estimated_tokens):
        """Wait if we would exceed rate limit"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.window_start

            # Reset window after 60 seconds
            if elapsed >= 60:
                self.tokens_used = 0
                self.window_start = current_time
                elapsed = 0

            # Check if adding these tokens would exceed limit
            if self.tokens_used + estimated_tokens > self.max_tokens_per_min:
                # Wait until window resets
                sleep_time = 60 - elapsed + 0.1  # +0.1s buffer
                print(f"Rate limit approaching, sleeping {sleep_time:.1f}s...", flush=True)
                time.sleep(sleep_time)
                # Reset after sleep
                self.tokens_used = 0
                self.window_start = time.time()

            self.tokens_used += estimated_tokens

# Global rate limiter (80% of 500k = 400k tokens/min to be safe)
rate_limiter = RateLimiter(tokens_per_minute=500000, buffer_percent=0.8)

def calculate_cost(input_tokens, output_tokens):
    """Calculates the cost of the request based on token usage."""
    input_cost = (input_tokens / 1000) * PRICE_INPUT_PER_1K
    output_cost = (output_tokens / 1000) * PRICE_OUTPUT_PER_1K
    return input_cost + output_cost

def run_agent_request(agent_id, prompt):
    """
    Sends a request to Bedrock and returns metadata and response.
    """
    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # Allow for detailed mathematical explanations
        "temperature": 0.2
    }

    # Estimate tokens (rough: ~4 chars per token)
    estimated_tokens = (len(prompt) + body["max_tokens"]) // 4

    # Wait if rate limit would be exceeded
    rate_limiter.wait_if_needed(estimated_tokens)

    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    attempt = 0
    max_retries = 3

    try:
        while True:
            try:
                response = bedrock.invoke_model(
                    modelId=MODEL_ID,
                    body=json.dumps(body)
                )
                break
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == 'ThrottlingException' and attempt < max_retries:
                    attempt += 1
                    backoff = 2 ** attempt
                    print(f"Agent {agent_id} throttled, retry {attempt}/{max_retries} after {backoff}s", flush=True)
                    time.sleep(backoff)
                    continue
                else:
                    raise e

        # Read stream/body (now outside the while loop)
        response_body = json.loads(response["body"].read())
        end_time = time.time()
        end_dt = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Extract data
        content = response_body["content"][0]["text"].strip().replace("\n", " ")
        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Calculate metrics
        duration = end_time - start_time
        cost = calculate_cost(input_tokens, output_tokens)

        return {
            "Agent ID": agent_id,
            "Status": "Success",
            "Send Time": start_dt,
            "Arrival Time": end_dt,
            "Duration (s)": f"{duration:.2f}",
            "Input Tok": input_tokens,
            "Output Tok": output_tokens,
            "Cost ($)": f"${cost:.6f}",
            "Response": content  # Return full content
        }

    except Exception as e:
        end_time = time.time()
        end_dt = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        duration = end_time - start_time

        return {
            "Agent ID": agent_id,
            "Status": "Error",
            "Send Time": start_dt,
            "Arrival Time": end_dt,
            "Duration (s)": f"{duration:.2f}",
            "Input Tok": 0,
            "Output Tok": 0,
            "Cost ($)": "$0.000000",
            "Response": str(e)
        }



def print_table(data):
    """Prints a formatted table of the results."""
    if not data:
        print("No data to display.")
        return

    headers = ["Agent ID", "Status", "Send Time", "Arrival Time", "Duration (s)", "Input Tok", "Output Tok", "Cost ($)", "Response"]
    
    # Calculate column widths (max 50 for Response)
    widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(h, ""))
            if h == "Response" and len(val) > 50:
                widths[h] = 53 # 50 + "..."
            else:
                widths[h] = max(widths[h], len(val))
    
    # Create format string
    fmt = " | ".join([f"{{:<{widths[h]}}}" for h in headers])
    separator = "-+-".join(["-" * widths[h] for h in headers])
    
    print("\n" + fmt.format(*headers))
    print(separator)
    for row in data:
        row_display = row.copy()
        # Truncate Response for table display
        if len(str(row_display["Response"])) > 50:
            row_display["Response"] = str(row_display["Response"])[:50] + "..."
            
        print(fmt.format(*[str(row_display.get(h, "")) for h in headers]))

if __name__ == "__main__":
    # Complex mathematical problem requiring detailed explanation (~5000 tokens input)
    PROMPT = """You are a mathematical expert. Please solve the following complex problem with detailed explanations at each step.

PROBLEM: Consider a dynamical system in game theory with multiple agents playing a repeated Prisoner's Dilemma game.

Setup:
- There are N=100 agents in the system
- Each agent can choose to Cooperate (C) or Defect (D) in each round
- Payoff matrix for pairwise interactions:
  * Both Cooperate: (3, 3)
  * Both Defect: (1, 1)
  * One Cooperates, other Defects: (0, 5) - cooperator gets 0, defector gets 5
- Agents update strategies using reinforcement learning with discount factor γ = 0.95
- The system runs for T=1000 time steps
- Each agent plays against 10 randomly selected opponents per round
- Agents use epsilon-greedy strategy with ε = 0.1

Additional complexity:
1. Some agents (20%) are "altruistic" and have a bias toward cooperation with probability p_alt = 0.7
2. Some agents (20%) are "selfish" and have a bias toward defection with probability p_self = 0.8
3. The remaining 60% are "rational" agents that learn purely from payoffs
4. There is a network structure: agents are arranged on a small-world graph with rewiring probability p_rewire = 0.1
5. Learning rate α varies by agent type: α_alt = 0.05, α_self = 0.15, α_rat = 0.1

Mathematical Questions to Answer:

Q1. Nash Equilibrium Analysis:
   a) What are the pure strategy Nash equilibria for this system?
   b) What are the mixed strategy Nash equilibria?
   c) How does the network structure affect equilibrium convergence?
   d) Derive the conditions for evolutionary stable strategies (ESS)

Q2. Convergence Analysis:
   a) Under what conditions does the system converge to mutual cooperation?
   b) What is the expected convergence time T_conv as a function of N, γ, α, and ε?
   c) Prove or disprove: The system will converge to a stable equilibrium with probability > 0.95
   d) How does the small-world network topology affect convergence compared to random graphs?

Q3. Payoff Dynamics:
   a) Calculate the expected average payoff per agent at equilibrium
   b) Derive the variance in payoffs across different agent types
   c) What is the social welfare (sum of all payoffs) at different equilibria?
   d) Compare Pareto efficiency of different equilibria

Q4. Learning Dynamics:
   a) Model the Q-learning update equations for each agent type
   b) What is the rate of convergence of Q-values to optimal values?
   c) How does epsilon-greedy exploration affect the final equilibrium?
   d) Derive the Bellman equations for this multi-agent system

Q5. Stochastic Analysis:
   a) Model the system as a Markov chain - what is the state space size?
   b) Calculate the stationary distribution of the Markov chain
   c) What is the mixing time of the chain?
   d) Prove convergence to stationary distribution using coupling arguments

Q6. Information Theoretic Analysis:
   a) Calculate the mutual information between agent types and strategies
   b) What is the entropy of the strategy distribution at equilibrium?
   c) How much information does an agent gain from observing neighbors?
   d) Derive bounds on the KL-divergence between learned and optimal policies

Q7. Network Effects:
   a) How does clustering coefficient affect cooperation rates?
   b) Calculate the impact of network centrality on agent payoffs
   c) What role do hub nodes play in strategy propagation?
   d) Model the diffusion of cooperation as an epidemic process

Q8. Robustness Analysis:
   a) How robust is cooperation to invasion by defectors?
   b) What fraction of defectors can the system tolerate before collapse?
   c) Analyze bifurcation points in the parameter space (γ, α, ε, N)
   d) Derive stability conditions using Lyapunov analysis

Q9. Computational Complexity:
   a) What is the computational complexity of finding Nash equilibria?
   b) Is the equilibrium selection problem NP-hard for this system?
   c) Provide approximation algorithms with provable guarantees
   d) Analyze the sample complexity of learning optimal strategies

Q10. Extensions:
   a) How would the analysis change with continuous action spaces?
   b) What if agents have incomplete information about opponent types?
   c) Add temporal discounting - how does this affect cooperation sustainability?
   d) Model reputation systems - derive the reputation dynamics equations

For each question, provide:
1. Detailed mathematical derivations with all steps shown
2. Proofs where applicable (using contradiction, induction, or direct proof)
3. Numerical examples with concrete calculations
4. Intuitive explanations of the mathematical results
5. Discussion of assumptions and their validity
6. Connections to relevant theorems (e.g., Folk Theorem, Fixed Point Theorems)
7. Comparison with empirical results from agent-based simulations

Please be extremely thorough and rigorous in your mathematical exposition. Use proper mathematical notation, cite relevant theorems, and provide complete proofs for all claims."""

    NUM_AGENTS = 40
    MAX_TOKENS = 2000

    # Calculate token usage and throttling predictions
    prompt_chars = len(PROMPT)
    estimated_input_tokens = prompt_chars // 4  # Rough estimate: 4 chars per token
    estimated_output_tokens = MAX_TOKENS  # Assume full output
    tokens_per_request = estimated_input_tokens + estimated_output_tokens

    total_tokens_needed = NUM_AGENTS * tokens_per_request
    rate_limit_tokens_per_min = 500000
    effective_rate_limit = int(rate_limit_tokens_per_min * 0.8)  # 80% buffer

    # Calculate how many agents can run without throttling
    agents_that_fit_in_quota = effective_rate_limit // tokens_per_request
    agents_to_throttle = max(0, NUM_AGENTS - agents_that_fit_in_quota)

    # Calculate expected wait time
    if total_tokens_needed > effective_rate_limit:
        extra_tokens = total_tokens_needed - effective_rate_limit
        extra_time_needed = (extra_tokens / effective_rate_limit) * 60  # seconds
    else:
        extra_time_needed = 0

    print("=" * 80)
    print("THROTTLING CALCULATION & PREDICTION")
    print("=" * 80)
    print(f"\n📊 Token Usage Estimation:")
    print(f"  Prompt length: {prompt_chars:,} characters")
    print(f"  Estimated input tokens per request: {estimated_input_tokens:,}")
    print(f"  Max output tokens per request: {estimated_output_tokens:,}")
    print(f"  Total tokens per request: {tokens_per_request:,}")
    print(f"\n🔢 Total Requirements:")
    print(f"  Number of agents: {NUM_AGENTS}")
    print(f"  Total tokens needed: {total_tokens_needed:,}")
    print(f"\n⚡ Rate Limit Analysis:")
    print(f"  AWS quota: {rate_limit_tokens_per_min:,} tokens/min")
    print(f"  Effective limit (80% buffer): {effective_rate_limit:,} tokens/min")
    print(f"  Tokens per request: {tokens_per_request:,}")
    print(f"\n🎯 Throttling Prediction:")
    print(f"  Agents that fit in 1-min quota: {agents_that_fit_in_quota}")
    print(f"  Agents that will be throttled: {agents_to_throttle}")

    if agents_to_throttle > 0:
        print(f"\n⏰ Expected Behavior:")
        print(f"  ✅ Agents 1-{agents_that_fit_in_quota}: Will complete immediately")
        print(f"  ⏸️  Agents {agents_that_fit_in_quota + 1}-{NUM_AGENTS}: Will wait for rate limit reset")
        print(f"  ⌛ Extra time needed: ~{extra_time_needed:.1f} seconds")
        print(f"  📈 Total estimated time: ~{60 + extra_time_needed:.1f} seconds")
    else:
        print(f"\n✅ All {NUM_AGENTS} agents fit within rate limit!")
        print(f"  Expected completion: ~30-60 seconds (no throttling needed)")

    # Calculate cost
    input_cost = (total_tokens_needed * 0.5) * (PRICE_INPUT_PER_1K / 1000)  # 50% of tokens are input
    output_cost = (total_tokens_needed * 0.5) * (PRICE_OUTPUT_PER_1K / 1000)  # 50% are output
    total_estimated_cost = input_cost + output_cost

    print(f"\n💰 Cost Estimation:")
    print(f"  Estimated total cost: ${total_estimated_cost:.6f}")
    print(f"  Cost per agent: ${total_estimated_cost/NUM_AGENTS:.6f}")

    print("\n" + "=" * 80)
    print(f"Starting {NUM_AGENTS} concurrent agents...")
    print(f"Model: {MODEL_ID}")
    print("=" * 80 + "\n")

    results = []
    # Use more workers for faster parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all tasks
        future_to_agent = {executor.submit(run_agent_request, i+1, PROMPT): i for i in range(NUM_AGENTS)}
        
        # Wait for completion
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_id = future_to_agent[future]
            try:
                data = future.result()
                results.append(data)
                
                # Print error immediately if one occurred
                if data["Status"] == "Error":
                    curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{curr_time}] !! Agent {agent_id} Error: {data['Response']}", flush=True)
                    
            except Exception as exc:
                print(f"Agent {agent_id} generated an exception: {exc}")

    # Sort results by Agent ID for cleaner output
    results.sort(key=lambda x: x["Agent ID"])

    print_table(results)

    # Calculate statistics
    successful = [r for r in results if r["Status"] == "Success"]
    errors = [r for r in results if r["Status"] == "Error"]

    total_cost = sum(float(r["Cost ($)"].replace("$", "")) for r in results)
    total_input_tokens = sum(r["Input Tok"] for r in successful)
    total_output_tokens = sum(r["Output Tok"] for r in successful)

    print(f"\n=== Summary ===")
    print(f"Total Agents: {NUM_AGENTS}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_input_tokens + total_output_tokens}")
    print(f"Total Cost: ${total_cost:.6f}")

    if successful:
        avg_duration = sum(float(r["Duration (s)"]) for r in successful) / len(successful)
        print(f"Avg Duration: {avg_duration:.2f}s")

    # Print Error Summary
    if errors:
        print(f"\n--- Error Summary ({len(errors)} errors) ---")
        error_types = {}
        for r in errors:
            error_msg = r['Response']
            # Extract error type
            if "ThrottlingException" in error_msg:
                error_type = "ThrottlingException"
            elif "ValidationException" in error_msg:
                error_type = "ValidationException"
            else:
                error_type = "Other"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
