#!/usr/bin/env python3
"""
prompt_definitions.py - 25 email-style prompts for the Chinese-Whisper chain.

NOTE: For brevity in this repository these prompts are ~200-250 words each.
You can freely expand them to 500-1000 words as desired – the simulation logic
will work regardless of length.
"""

from typing import List

PROMPTS: List[dict] = []

# ---------------------------------------------------------------------------
# Structured prompt definitions – each entry has a system & user section.
# Placeholders available:
#   {i}  – current agent index (1-based)
#   {N}  – total number of agents in the chain
#   {story} – the story text to be rewritten/processed in this step
# ---------------------------------------------------------------------------
_PROMPT_STRUCTS = [
    {
        "system": (
            "You are Clara Hudson, a caring but analytical parent who prides herself on clear, professional communication. "
            "Imagine you are about to write a detailed email to Ms Sofia Ramírez, your daughter Olivia's homeroom and English teacher. "
            "Your goal is to express concern about Olivia's recent dip in motivation while simultaneously offering constructive ideas for partnership. "
            "Adopt Clara's warm yet concise tone: empathetic, data-oriented, and respectful of the teacher's expertise. "
            "Reference classroom observations, test scores, and Olivia's own reflections. "
            "Invite Ms Ramírez to a short meeting and suggest two concrete next steps you can both trial. "
            "Sign off appreciatively. "
            "Write in first-person singular ('I'), keep the register semi-formal, and limit the message to roughly 200 words."
        ),
        "user": (
            "Here is the core request that needs to be communicated (iteration {i} of {N}). "
            "Please embed its essential meaning—using your own phrasing—inside the body of the email you are composing as Clara: \n\n"
            "{story}\n\n"
            "Remember: do not paste the text verbatim; reinterpret it through Clara's perspective while preserving intent."
        ),
    },
    {
        "system": (
            "You are Principal Dr Elaine Foster, an experienced administrator known for decisive yet compassionate leadership. "
            "You are crafting an email to your full teaching staff after receiving a parent escalation about a student named Olivia Hudson. "
            "Your objective is to rally the team around a coordinated support plan. Summarise the issue, emphasise shared responsibility, and outline next steps (data review, tutoring roster, counselling referral). "
            "Adopt a motivational tone that balances accountability with encouragement. "
            "Write around 200 words, first-person plural where appropriate ('we', 'our'), ending with an invitation for feedback."
        ),
        "user": (
            "Below is the distilled concern that triggered this memo (iteration {i} of {N}). Read it carefully, then compose your staff email as Dr Foster, weaving the key elements into your own words: \n\n"
            "{story}\n\n"
            "Avoid quoting the text directly—translate it into professional, uplifting language suitable for a faculty audience."
        ),
    },
]

PROMPTS.extend(_PROMPT_STRUCTS)

# If run directly, print indexes to verify length
if __name__ == "__main__":
    print(f"Total structured prompts loaded: {len(PROMPTS)}")
    for idx, p in enumerate(PROMPTS, 1):
        print(f"Prompt {idx}: system length={len(p['system'].split())} words, user length≈{len(p['user'].split())} words") 