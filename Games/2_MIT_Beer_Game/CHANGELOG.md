# MIT Beer Game Changelog

## Logical Fixes (2025-07-30)

### Fixed
1. **Per-round logging accuracy**: Extended RoundData to capture LLM decision snapshots at the time of each round, preventing overwriting of historical data with latest agent state
2. **Workflow total_rounds**: Fixed hardcoded `total_rounds=3` to use actual `num_rounds` parameter
3. **Communication toggle**: Changed `--enable_communication` default from True to False, making it opt-in
4. **Unused parameter**: Removed unused `--profit_per_unit_sold` CLI parameter
5. **Misleading variable**: Removed `retailer_order` variable that incorrectly suggested orders equal shipments
6. **Pipeline fields**: Now populating `orders_in_transit_0/1` and `production_queue_0/1` fields in RoundData

### Testing
- Added comprehensive test suite in `test_logging_and_flags.py` covering all fixes
- Tests use stubbed LLM responses for deterministic behavior
- No network calls required for testing 