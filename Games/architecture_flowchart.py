#!/usr/bin/env python3
"""
Architecture Flowchart Generator

This script uses LLM to generate a Mermaid flowchart from a Python script file.
It should be placed in the Games directory and called from inside game folders.
"""

import os
import sys
import json
import requests
import tempfile
import subprocess
import webbrowser
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_flowchart_from_script(script_path, output_dir=None, render=True, save_llm_output=True):
    """
    Generate a Mermaid flowchart from a Python script using LiteLLM.
    
    Args:
        script_path: Path to the Python script
        output_dir: Directory to save the output files (default: same as script)
        render: Whether to render the flowchart as HTML
        save_llm_output: Whether to save the raw LLM output
    
    Returns:
        Dict with paths to generated files
    """
    # Read the script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Set up output directory
    script_path = Path(script_path)
    if output_dir is None:
        output_dir = script_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = script_path.stem
    
    # Prepare the prompt
    prompt = f"""
    Please analyze the following Python script and create a comprehensive Mermaid flowchart 
    that visualizes its structure and execution flow. Focus on the main functions, classes, 
    and their relationships. Represent control flow, decision points, and data flow.
    
    Only respond with the Mermaid flowchart code, nothing else.
    
    ```python
    {script_content}
    ```
    """
    
    # Get API key from environment
    api_key = os.getenv('LITELLM_API_KEY')
    
    if not api_key:
        print("Error: No LITELLM_API_KEY found. Please set it in your environment or .env file.")
        sys.exit(1)
    
    # Query LiteLLM for the flowchart
    llm_response = query_litellm(prompt, api_key)
    
    # Extract the flowchart code
    flowchart_code = extract_mermaid_code(llm_response)
    
    # Save the raw LLM output if requested
    result_files = {}
    if save_llm_output:
        llm_output_file = output_dir / f"{base_name}_llm_output.txt"
        with open(llm_output_file, 'w') as f:
            f.write(llm_response)
        print(f"Raw LLM output saved to {llm_output_file}")
        result_files['llm_output'] = llm_output_file
    
    # Save the flowchart code
    mermaid_file = output_dir / f"{base_name}_flowchart.mmd"
    with open(mermaid_file, 'w') as f:
        f.write(flowchart_code)
    
    print(f"Flowchart code saved to {mermaid_file}")
    result_files['mermaid'] = mermaid_file
    
    # Optionally render the flowchart
    if render:
        html_file = output_dir / f"{base_name}_flowchart.html"
        render_mermaid_html(flowchart_code, html_file)
        result_files['html'] = html_file
    
    return result_files

def query_litellm(prompt, api_key):
    """Query LiteLLM API for flowchart generation"""
    messages = [
        {"role": "system", "content": "You are an expert in Python code analysis and Mermaid flowchart creation."},
        {"role": "user", "content": prompt}
    ]
    
    return _call_litellm(messages, api_key)

def _call_litellm(messages, api_key, model="gpt-4o", temperature=0.8):
    """Call LiteLLM API and return the response content"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.post(
            "https://litellm.sph-prod.ethz.ch/chat/completions", 
            json=payload, 
            headers=headers
        )
        
        if not response.ok:
            err_txt = response.text
            raise Exception(f"LiteLLM API error: {response.status_code} {response.reason} - {err_txt}")
        
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
    
    except Exception as e:
        print(f"Error querying LiteLLM API: {e}")
        sys.exit(1)

def extract_mermaid_code(text):
    """Extract Mermaid code from the LLM response"""
    # If the response has markdown code blocks, extract just the mermaid code
    if "```mermaid" in text:
        flowchart_code = text.split("```mermaid")[1].split("```")[0].strip()
    elif "```" in text:
        flowchart_code = text.split("```")[1].split("```")[0].strip()
    else:
        flowchart_code = text.strip()
    
    return flowchart_code

def render_mermaid_html(mermaid_code, output_path):
    """Renders a Mermaid diagram as an HTML file"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Architecture Flowchart</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                logLevel: 'fatal',
                securityLevel: 'loose',
                flowchart: {{ htmlLabels: true }}
            }});
        </script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h1>Architecture Flowchart</h1>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    print(f"HTML visualization saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # This is only for testing the script directly
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        create_flowchart_from_script(script_path, output_dir)
    else:
        print("Usage: python architecture_flowchart.py <script_path> [output_dir]")
        sys.exit(1)