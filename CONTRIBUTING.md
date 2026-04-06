# Contributing

## Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Before Submitting a PR
```bash
# Lint
flake8 src/ main.py --max-line-length=100

# Tests must pass
pytest tests/ -v --cov=src

# Check no secrets leaked
grep -r "api_key\|API_KEY" --include="*.py" . | grep -v ".env\|os.getenv\|example"
```

## Adding a New Therapeutic Area
1. Add trial template to `TRIAL_TEMPLATES` in `src/utils/data_generator.py`
2. Add lab checks for the new area in `score_patient_trial()` in `src/models/eligibility_screener.py`
3. Add patient generation logic for the area in `generate_patients()`
4. Add a test case in `tests/test_matching.py`

## Adding a New Agent Node
1. Define the node function in `src/agents/matching_agent.py`
2. Add only serializable fields to `AgentState`
3. Register: `workflow.add_node("node_name", node_fn)`
4. Add edge: `workflow.add_edge("prev_node", "node_name")`
5. Update `CLAUDE.md` node map table
