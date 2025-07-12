#!/usr/bin/env python3
"""
story_definitions.py - Single story definition for Chinese Whispers SQL simulation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class StoryDefinition:
    name: str
    description: str
    initial_story: str
    tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# ---------------------------------------------------------------------------
# ONLY ONE STORY – Parent writes a very long email to teacher.
# Expand the body as desired (≈800-900 words placeholder below).
# ---------------------------------------------------------------------------
PARENT_TEACHER_ESCALATION = StoryDefinition(
    name="parent_teacher_escalation",
    description="Long escalation chain beginning with a parent email about grades, well-being, and test scores.",
    initial_story=(
        """Subject: Request for Insight on Olivia Hudson's Recent Challenges\n\n"
        "Dear Ms. Ramirez,\n\n"
        "I hope you had a restful weekend and that the school's preparations for next month's Spring Fair are progressing smoothly. Olivia has been very excited about the robotics booth and keeps talking about the new 3-D printer the PTA funded, although she insists the cafeteria still serves the best macaroni on Wednesdays!  \n\n"
        "I am writing, however, because her latest progress report shows unexpected difficulties in mathematics and science. According to the portal she scored 61% on the algebra quiz and 68% on the life-cycle lab, whereas earlier this term she was comfortably in the high seventies. Strangely, her reading score climbed to 92% and she wrote a beautiful book review for the library newsletter.  \n\n"
        "At home nothing major has changed except that she has started the after-school chess club (she says the room is very cold, perhaps that's distracting?) and we switched our internet provider which briefly interrupted her online homework tracker. We also had a family reunion last weekend—lots of cousins, late nights.  \n\n"
        "Could you please share any observations you might have: missed assignments, attentiveness in class, friendship dynamics, anything small or large that may explain these dips? I would love to arrange a short meeting and brainstorm two concrete actions to help her regain confidence.\n\n"
        "Thank you for your dedication and for navigating all the bus reroutes while the new parking lot is paved.\n\n"
        "Warm regards,\n"
        "Clara Hudson"""
    ),
    tags=["parent", "teacher", "escalation"]
)


# Export helper --------------------------------------------------------------

STORIES: Dict[str, StoryDefinition] = {PARENT_TEACHER_ESCALATION.name: PARENT_TEACHER_ESCALATION}


def get_story(name: str = "parent_teacher_escalation") -> StoryDefinition:
    if name not in STORIES:
        raise ValueError(f"Story '{name}' not found. Available: {list(STORIES)}")
    return STORIES[name]


def list_stories() -> None:
    print("Available story: parent_teacher_escalation – a long parent email.")


# Compatibility helpers ------------------------------------------------------

def get_story_by_name(name: str):
    return get_story(name)


def get_default_story():
    return get_story()


def get_story_for_config(story_name: str = "parent_teacher_escalation") -> Dict[str, Any]:
    s = get_story(story_name)
    return {
        "initial_story": s.initial_story,
        "story_name": s.name,
        "story_description": s.description,
        "story_tags": s.tags,
    }


if __name__ == "__main__":
    list_stories()
    s = get_story()
    print("\nPreview (first 400 chars):\n")
    print(s.initial_story[:400] + "…") 