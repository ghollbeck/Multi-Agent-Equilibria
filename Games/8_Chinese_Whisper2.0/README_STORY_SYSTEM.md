# Chinese Whispers SQL - Story Definition System

This document explains the new story definition system that allows you to define stories in Python code rather than configuration files.

## Overview

The story definition system provides:
- **Centralized story management** in `story_definitions.py`
- **Multiple predefined stories** with different difficulty levels
- **Metadata support** (tags, difficulty, expected SQL patterns)
- **Easy story selection** via command line arguments
- **Backward compatibility** with existing config system

## Story Library

### Available Stories

The system includes 8 predefined stories with varying difficulty levels:

#### Easy Stories (Simple AND conditions)
- **`high_achievers`**: Find students who passed all subjects
- **`spanish_speakers`**: Find Spanish speakers with good grades
- **`english_proficiency`**: Find English-proficient active students

#### Medium Stories (Multiple conditions)
- **`inactive_students`**: Find inactive students with specific criteria (default)
- **`dropout_risk`**: Identify students at risk of dropping out
- **`multilingual_achievers`**: Find high-performing bilingual students

#### Hard Stories (Complex logic)
- **`language_learners`**: Find bilingual students with mixed performance (XOR logic)
- **`struggling_students`**: Find students failing multiple subjects (arithmetic conditions)

## Usage

### Command Line Interface

```bash
# List all available stories
python simulate_chain.py --list-stories
python run_complete_pipeline.py --list-stories

# Use a specific story
python simulate_chain.py --story high_achievers
python run_complete_pipeline.py --story language_learners

# Run with different stories
python run_complete_pipeline.py --story spanish_speakers
python run_complete_pipeline.py --story struggling_students
```

### Python API

```python
from story_definitions import StoryLibrary, get_story_by_name, get_default_story

# Get the default story
default_story = get_default_story()
print(default_story.initial_story)

# Get a specific story
story = get_story_by_name("high_achievers")
print(f"Difficulty: {story.difficulty}")
print(f"Tags: {story.tags}")

# Get all stories
all_stories = StoryLibrary.get_all_stories()

# Filter by difficulty
easy_stories = StoryLibrary.get_stories_by_difficulty("easy")
hard_stories = StoryLibrary.get_stories_by_difficulty("hard")

# Filter by tag
bilingual_stories = StoryLibrary.get_stories_by_tag("bilingual")
```

## Story Structure

Each story is defined using the `StoryDefinition` dataclass:

```python
@dataclass
class StoryDefinition:
    name: str                    # Unique identifier
    description: str             # Human-readable description
    initial_story: str          # The story text for the simulation
    expected_sql_pattern: str   # Expected SQL pattern (for validation)
    difficulty: str             # "easy", "medium", or "hard"
    tags: List[str]             # Categorization tags
    notes: str                  # Additional notes
```

## Adding New Stories

To add a new story to the library:

1. **Edit `story_definitions.py`**
2. **Add a new `StoryDefinition`** to the `StoryLibrary` class:

```python
NEW_STORY = StoryDefinition(
    name="new_story",
    description="Brief description of what this story tests",
    initial_story="""Your story text here. This will be the starting point for the Chinese Whispers chain.""",
    expected_sql_pattern="SELECT * FROM students WHERE condition = 1",
    difficulty="medium",
    tags=["tag1", "tag2", "category"],
    notes="Additional notes about this story"
)
```

3. **Test the new story**:
```bash
python story_definitions.py  # Should show your new story
python simulate_chain.py --story new_story
```

## Story Categories

### By Difficulty
- **Easy**: Simple AND conditions, basic filtering
- **Medium**: Multiple conditions, moderate complexity
- **Hard**: Complex logic (OR, XOR, arithmetic), nested conditions

### By Tags
- **Language**: `language`, `bilingual`, `language_proficiency`
- **Academic**: `academic_excellence`, `academic_failure`, `academic_support`
- **Logic**: `boolean_logic`, `complex_logic`, `exclusive_or`, `simple_conditions`
- **Purpose**: `risk_assessment`, `status_tracking`, `filtering`

## File Structure

```
Games/8_Chinese_Whisper2.0/
├── story_definitions.py          # Story library and definitions
├── simulate_chain.py             # Modified to support story selection
├── run_complete_pipeline.py      # Modified to support story selection
├── config.yaml                   # Still used for other configurations
└── results/
    └── run_YYYY-MM-DD_HH-MM-SS/
        ├── run_info.json         # Includes story metadata
        └── ...
```

## Integration with Results

When using a story from the library:
- **Story metadata** is saved in `run_info.json`
- **Expected SQL pattern** can be used for validation
- **Tags and difficulty** help categorize results
- **Run folder** contains complete story information

## Examples

### Run Different Difficulty Levels
```bash
# Easy story
python run_complete_pipeline.py --story high_achievers

# Medium story  
python run_complete_pipeline.py --story inactive_students

# Hard story
python run_complete_pipeline.py --story language_learners
```

### Compare Stories
```bash
# Run multiple stories and compare results
python run_complete_pipeline.py --story spanish_speakers
python run_complete_pipeline.py --story english_proficiency
python run_complete_pipeline.py --story multilingual_achievers

# Check results folder to compare outputs
ls results/
```

## Benefits

1. **Centralized Management**: All stories in one place
2. **Rich Metadata**: Tags, difficulty, expected patterns
3. **Easy Selection**: Command line story selection
4. **Reproducibility**: Stories are versioned with code
5. **Extensibility**: Easy to add new stories
6. **Validation**: Expected SQL patterns for testing
7. **Categorization**: Filter stories by tags or difficulty

## Backward Compatibility

The system maintains backward compatibility:
- **Config file** still works as before
- **`--story` argument** overrides config story
- **Default behavior** unchanged if no story specified
- **Existing scripts** continue to work without modification 