# CURSOR RULES - Systematic Development Practices

## 🎯 PRIMARY DIRECTIVE
**ALWAYS follow this systematic approach for ANY coding task to ensure quality, traceability, and maintainability.**

---

## 📋 PRE-EXECUTION CHECKLIST

### 1. 🔍 HISTORICAL AWARENESS CHECK
**Before starting ANY task, ALWAYS:**
- [ ] Search for and read relevant `Readme_running.md` files in the project
- [ ] Check existing implementations and what worked/didn't work
- [ ] Review user memories for relevant past experiences
- [ ] Understand the broader project context and architecture

**Commands to run:**
```bash
# Find all readme running files
find . -name "*Readme_running*" -o -name "*readme_running*" -type f

# Search for similar implementations
grep -r "similar_feature_name" . --include="*.py" --include="*.md"
```

### 2. 📝 CHAIN OF THOUGHT PLANNING
**Create explicit planning documentation BEFORE coding:**
- [ ] **Goal Definition**: What exactly needs to be implemented/fixed?
- [ ] **Current State Analysis**: What exists now?
- [ ] **Required Changes**: List specific files and modifications needed
- [ ] **Dependencies**: What other systems/files will be affected?
- [ ] **Testing Strategy**: How will we verify it works?
- [ ] **Risk Assessment**: What could go wrong?

**Planning Template:**
```markdown
## TASK PLANNING - [Task Name]
**Date**: [Current Date]
**Goal**: [Clear objective]

### Current State Analysis
- Existing files: [list relevant files]
- Current functionality: [describe what works now]
- Known issues: [list problems to address]

### Implementation Plan
1. [Step 1 with specific files]
2. [Step 2 with specific files]
3. [Testing approach]
4. [Documentation updates]

### Risk Mitigation
- Potential issues: [list risks]
- Backup strategy: [how to revert if needed]
- Testing checkpoints: [verification steps]
```

---

## 🛠️ IMPLEMENTATION WORKFLOW

### 3. 🧪 TEST-DRIVEN DEVELOPMENT
**For EVERY script modification:**

#### A. Create/Update Unit Tests FIRST
- [ ] **Test File Location**: `tests/test_[module_name].py` or inline tests
- [ ] **Test Coverage**: Test both success and failure cases
- [ ] **Mock External Dependencies**: APIs, file I/O, user input
- [ ] **Edge Cases**: Empty inputs, boundary conditions, error states

**Unit Test Template:**
```python
import unittest
import logging
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Test[ModuleName](unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        logging.disable(logging.CRITICAL)  # Suppress logs during testing
    
    def tearDown(self):
        """Clean up after each test method."""
        logging.disable(logging.NOTSET)  # Re-enable logs
    
    def test_[function_name]_success(self):
        """Test successful execution of [function_name]."""
        # Arrange
        test_input = "test_data"
        expected_output = "expected_result"
        
        # Act
        result = module_function(test_input)
        
        # Assert
        self.assertEqual(result, expected_output)
    
    def test_[function_name]_error_handling(self):
        """Test error handling in [function_name]."""
        # Test invalid inputs
        with self.assertRaises(ValueError):
            module_function(None)
    
    @patch('module.external_dependency')
    def test_[function_name]_with_mocks(self, mock_dependency):
        """Test [function_name] with mocked dependencies."""
        mock_dependency.return_value = "mocked_result"
        result = module_function("input")
        self.assertEqual(result, "expected_with_mock")

if __name__ == '__main__':
    unittest.main()
```

#### B. Run Tests BEFORE Making Changes
```bash
# Run specific test file
python -m pytest tests/test_[module_name].py -v

# Run all tests with coverage
python -m pytest --cov=[module_name] --cov-report=html

# Run tests and save results
python -m pytest tests/ > test_results_pre_change.txt 2>&1
```

### 4. 📊 COMPREHENSIVE LOGGING IMPLEMENTATION
**Add detailed logging to EVERY function:**

#### A. Logging Setup Template
```python
import logging
import datetime
import functools
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/{__name__}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_function_calls(func):
    """Decorator to log function entry, exit, and errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 ENTERING: {func.__name__} with args={args[:3]}{'...' if len(args) > 3 else ''}, kwargs={list(kwargs.keys())}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ COMPLETED: {func.__name__} -> {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"❌ ERROR in {func.__name__}: {str(e)}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            raise
    return wrapper
```

#### B. Strategic Logging Points
```python
@log_function_calls
def main_function(param1, param2):
    """Main function with comprehensive logging."""
    logger.info(f"🎯 STARTING: main_function with param1={param1}, param2={param2}")
    
    # Input validation logging
    if not param1:
        logger.warning(f"⚠️ VALIDATION: param1 is empty or None")
        return None
    
    # Step-by-step process logging
    logger.info(f"📊 PROCESSING: Step 1 - Data preparation")
    processed_data = preprocess_data(param1)
    logger.info(f"📊 RESULT: Step 1 completed, processed {len(processed_data)} items")
    
    logger.info(f"📊 PROCESSING: Step 2 - Core logic execution")
    result = core_logic(processed_data, param2)
    logger.info(f"📊 RESULT: Step 2 completed with result type: {type(result)}")
    
    # Performance logging
    logger.info(f"⏱️ PERFORMANCE: Function completed in {datetime.datetime.now()}")
    
    return result
```

### 5. 🏃‍♂️ SCRIPT EXECUTION & VERIFICATION
**After making changes, ALWAYS:**

#### A. Run the Modified Script
```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# Run script with logging
python [modified_script].py 2>&1 | tee logs/execution_$(date +%Y%m%d_%H%M%S).log

# Check exit status
echo "Exit status: $?"
```

#### B. Verify Functionality
```bash
# Run unit tests AFTER changes
python -m pytest tests/test_[module_name].py -v > test_results_post_change.txt 2>&1

# Compare test results
diff test_results_pre_change.txt test_results_post_change.txt

# Run integration tests if available
python scripts/integration_test.py

# Check for runtime errors in logs
grep -i "error\|exception\|traceback" logs/execution_*.log
```

---

## 📚 DOCUMENTATION WORKFLOW

### 6. 📝 README_RUNNING CONTINUOUS UPDATES

#### A. Update BEFORE Making Changes
**Add to `Readme_running.md` (create if doesn't exist):**
```markdown
## Latest Changes (Most Recent First)

### [Current Date]: [Task Name] - PLANNING PHASE
- **Objective**: [What we're trying to accomplish]
- **Current State**: [What exists now]
- **Planned Changes**: 
  - File 1: [specific changes planned]
  - File 2: [specific changes planned]
- **Testing Strategy**: [how we'll verify it works]
- **Risk Assessment**: [potential issues and mitigation]
- **Expected Outcome**: [what success looks like]

---
```

#### B. Update DURING Implementation
```markdown
### [Current Date]: [Task Name] - IMPLEMENTATION PROGRESS
- **✅ Completed**: 
  - [Specific change 1] in [file]
  - [Specific change 2] in [file]
- **🔄 In Progress**: 
  - [Current task] in [file]
- **❌ Issues Encountered**: 
  - [Problem 1]: [Solution applied]
  - [Problem 2]: [Workaround used]
- **🧪 Testing Status**: 
  - Unit tests: [status]
  - Integration tests: [status]
  - Manual verification: [status]

---
```

#### C. Update AFTER Completion
```markdown
### [Current Date]: [Task Name] - COMPLETED ✅
- **Final Implementation**: 
  - [Summary of what was actually implemented]
  - Files modified: [list with key changes]
  - Tests added/updated: [list]
- **Verification Results**:
  - Unit tests: [X passed, Y failed]
  - Integration tests: [results]
  - Performance impact: [any changes]
- **Known Issues**: [any remaining problems]
- **Future Improvements**: [suggestions for next iteration]
- **Searchable Keywords**: [relevant terms for LLM search]

---
```

### 7. 🔍 SEARCHABLE DOCUMENTATION
**Include comprehensive keywords in every Readme_running entry:**
```markdown
**Searchable Keywords**: [feature_name], [technology_used], [problem_solved], [files_modified], [testing_approach], [error_types_fixed], [performance_improvements], [integration_points], [dependencies_added], [api_endpoints], [database_changes], [configuration_updates]
```

---

## 🔧 POST-IMPLEMENTATION VERIFICATION

### 8. 🎯 FINAL QUALITY CHECKLIST

#### A. Code Quality
- [ ] **All functions have docstrings** with clear descriptions
- [ ] **All functions have logging** for entry, exit, and errors
- [ ] **Error handling** is comprehensive with meaningful messages
- [ ] **Input validation** prevents invalid data processing
- [ ] **Type hints** are used where appropriate
- [ ] **Constants** are defined instead of magic numbers

#### B. Testing Verification
- [ ] **Unit tests pass** with >80% coverage
- [ ] **Integration tests pass** (if applicable)
- [ ] **Manual testing** confirms expected behavior
- [ ] **Edge cases tested** (empty inputs, boundary conditions)
- [ ] **Error conditions tested** (invalid inputs, network failures)

#### C. Documentation Completeness
- [ ] **Readme_running.md updated** with complete change log
- [ ] **Code comments** explain complex logic
- [ ] **Function docstrings** describe parameters and return values
- [ ] **Searchable keywords** added for future reference
- [ ] **Usage examples** provided where appropriate

#### D. Operational Readiness
- [ ] **Script runs without errors** from command line
- [ ] **Log files generated** and contain useful information
- [ ] **Performance acceptable** (no significant degradation)
- [ ] **Dependencies documented** and installable
- [ ] **Configuration files updated** if needed

---

## 🚨 ERROR HANDLING PROTOCOLS

### 9. 🐛 Bug Detection & Resolution

#### A. When Tests Fail
```bash
# 1. Capture test failure details
python -m pytest tests/ --tb=long > test_failures.log 2>&1

# 2. Analyze failure patterns
grep -B5 -A10 "FAILED\|ERROR" test_failures.log

# 3. Fix issues one by one
# 4. Re-run tests after each fix
# 5. Update Readme_running.md with resolution details
```

#### B. When Script Execution Fails
```bash
# 1. Check logs for error details
tail -50 logs/execution_*.log

# 2. Run with enhanced debugging
python -u [script].py --debug 2>&1 | tee debug_output.log

# 3. Use Python debugger for complex issues
python -m pdb [script].py

# 4. Document resolution in Readme_running.md
```

#### C. Performance Issues
```bash
# 1. Profile script execution
python -m cProfile -o profile_output.prof [script].py

# 2. Analyze performance
python -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"

# 3. Monitor resource usage
top -p $(pgrep -f [script_name])

# 4. Document optimization in Readme_running.md
```

---

## 🎯 PROJECT-SPECIFIC ADAPTATIONS

### 10. 🎮 Multi-Agent Game Simulation Specific Rules

#### A. Game Logic Testing
- [ ] **Simulate complete game rounds** in tests
- [ ] **Test agent decision-making** with mock LLM responses
- [ ] **Verify game state consistency** across rounds
- [ ] **Test equilibrium conditions** where applicable
- [ ] **Validate scoring/metrics calculations**

#### B. LLM Integration Testing
```python
@patch('module.llm_client.call')
def test_agent_decision_making(self, mock_llm):
    """Test agent decision making with mocked LLM responses."""
    mock_llm.return_value = {"decision": "cooperate", "reasoning": "test"}
    
    agent = GameAgent("test_agent")
    decision = agent.make_decision(game_state)
    
    self.assertEqual(decision["decision"], "cooperate")
    mock_llm.assert_called_once()
```

#### C. Simulation Logging
```python
def log_game_round(round_num, game_state, decisions):
    """Log complete game round state and decisions."""
    logger.info(f"🎮 ROUND {round_num} START")
    logger.info(f"📊 Game State: {game_state}")
    
    for agent_id, decision in decisions.items():
        logger.info(f"🤖 Agent {agent_id}: {decision}")
    
    logger.info(f"🎮 ROUND {round_num} END")
```

---

## 📊 METRICS & MONITORING

### 11. 📈 Continuous Improvement Tracking

#### A. Code Quality Metrics
```bash
# Test coverage reporting
python -m pytest --cov=[module] --cov-report=term-missing

# Code complexity analysis
pip install radon
radon cc . --average

# Documentation coverage
pip install interrogate
interrogate -v .
```

#### B. Performance Monitoring
```python
import time
import psutil
import functools

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"⏱️ {func.__name__} executed in {end_time - start_time:.2f}s")
        logger.info(f"💾 Memory delta: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper
```

---

## 🎯 SUMMARY CHECKLIST

### Every Task Must Include:
- [ ] ✅ **Historical awareness check** - Read existing Readme_running.md files
- [ ] ✅ **Chain of thought planning** - Document approach before coding
- [ ] ✅ **Unit tests written/updated** - Test before and after changes
- [ ] ✅ **Comprehensive logging added** - Every function logged
- [ ] ✅ **Script execution verified** - Run and check for errors
- [ ] ✅ **Readme_running.md updated** - Before, during, and after changes
- [ ] ✅ **Quality verification** - All checklist items completed

### Success Criteria:
- **All tests pass** with comprehensive coverage
- **Script runs cleanly** with informative logs
- **Documentation is complete** and searchable
- **Changes are traceable** through commit history and logs
- **Future developers can understand** the implementation and rationale

---

**Remember: Quality over speed. Better to do it right once than to debug it multiple times.** 