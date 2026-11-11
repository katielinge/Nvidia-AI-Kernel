import json
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution
from scripts.run_local import get_trace_set_path
 
 
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    solution = Solution.model_validate_json(pack_solution().read_text())
    ts = TraceSet.from_path(get_trace_set_path())
    definition = ts.definitions[solution.definition]
    workloads = ts.workloads[solution.definition][:n]
 
    bench_ts = TraceSet(
        root=ts.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1, use_isolated_runner=True)
    print("Benchmark log folder:", getattr(config, "log_dir", "unknown"))
    result = Benchmark(bench_ts, config).run_all(dump_traces=False)
 
    for trace in result.traces.get(definition.name, []):
        ev = trace.evaluation
        print("=" * 80)
        print("Workload:", trace.workload.uuid)
        if ev is None:
            print("  (no evaluation record)")
            continue
        print("  Status:", ev.status.value)
        if ev.status.value != "PASSED":
            try:
                dump = ev.model_dump(mode="json")
            except Exception:
                dump = {k: str(v) for k, v in vars(ev).items()}
            print(json.dumps(dump, indent=2, default=str))
    print("=" * 80)
 
 
if __name__ == "__main__":
    main()
