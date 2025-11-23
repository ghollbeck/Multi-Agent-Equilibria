import os, json, datetime, threading

_LOCK = threading.Lock()

METRICS_FILE = os.path.join(os.path.dirname(__file__), "llm_cost_summary.json")


def _load():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {"all_runs_total_cost": 0.0, "runs": []}


def _save(data):
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def start_run() -> str:
    """Register a new simulation run and return its unique id."""
    with _LOCK:
        data = _load()
        run_id = f"run_{len(data['runs']) + 1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data["runs"].append({"run_id": run_id, "total_cost": 0.0, "timestamp": datetime.datetime.now().isoformat()})
        _save(data)
        return run_id


def add_cost(run_id: str, cost: float):
    """Accumulate cost for the given run and update global totals."""
    with _LOCK:
        data = _load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                r["total_cost"] += cost
                break
        else:
            # If run_id not found, create it (edge-case safety)
            data["runs"].append({"run_id": run_id, "total_cost": cost, "timestamp": datetime.datetime.now().isoformat()})
        data["all_runs_total_cost"] = sum(r["total_cost"] for r in data["runs"])
        _save(data)


def end_run(run_id: str):
    """Currently a no-op placeholder – kept for API symmetry."""
    pass 