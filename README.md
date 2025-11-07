# Introduction to Artificial Intelligence — Assignment 1: Ring Destroyer

This repository contains a local tester for the Codeforces “Ring Destroyer” assignment. It lets you:
- Run a single map against your agent implementation
- Generate and run batches of random maps
- Collect per-run reports and overall statistics

The tester imitates the interactor on the codeforces and communicates with the agent via stdin/stdout.

## Prerequisites

1. Install Python 3.11.9
2. Install Go 1.25.0
3. Get the solutions from Codeforces and build two executables:
	 - A* solution → save to `main.go`, then build:
		 - Windows (PowerShell):
			 ```powershell
			 go build -o a_star.exe
			 ```
		 - Linux/macOS:
			 ```bash
			 go build -o a_star
			 ```
	 - Backtracking (BFS) solution → replace `main.go` with that code, then build:
		 - Windows (PowerShell):
			 ```powershell
			 go build -o backtrack.exe
			 ```
		 - Linux/macOS:
			 ```bash
			 go build -o backtrack
			 ```

Note: You’re building two different binaries from different versions of `main.go`. Keep both binaries in the repo root.

## Running a single test

1. Create a 13x13 map in `maps/` as a `.txt` file (example: `maps/sample_map.txt`). Use the following tokens:
	 - `G` Gollum (must exist)
	 - `M` Mount Doom (must exist)
	 - `C` Mithril coat (optional)
	 - `O` Orc, `U` Uruk-hai, `N` Nazgûl, `W` Watchtower (enemies)
	 - `-` empty cell
	 Do not include perception zones (`P`); they’re computed by the tester.

2. Run with A*:
	 - Windows (PowerShell):
		 ```powershell
		 python .\tester.py --map maps\sample_map.txt --vision 1 --agent-cmd ".\a_star.exe"
		 ```
	 - Linux/macOS:
		 ```bash
		 python3 tester.py --map maps/sample_map.txt --vision 1 --agent-cmd "./a_star"
		 ```

3. Run with Backtracking (BFS):
	 - Windows (PowerShell):
		 ```powershell
		 python .\tester.py --map maps\sample_map.txt --vision 1 --agent-cmd ".\backtrack.exe"
		 ```
	 - Linux/macOS:
		 ```bash
		 python3 tester.py --map maps/sample_map.txt --vision 1 --agent-cmd "./backtrack"
		 ```

4. Where to find results:
	 - The tester writes a per-run report and transcript under `answers/`:
		 - `answers/<mapname>_v<vision>_report.txt`
		 - `answers/<mapname>_v<vision>_session.log`
	 - The console also prints the paths at the end of the run.

## Running a batch of generated tests

Use `tester_gen.py` to generate random maps and run your agent across them.

Basic usage (Windows PowerShell):

```powershell
# A* across 1000 maps, vision=1, fixed seed
python .\tester_gen.py --count 1000 --vision 1 --agent-cmd ".\a_star.exe" --seed 42 --forbid-start-hazards

# Backtracking across 1000 maps, vision=1, fixed seed
python .\tester_gen.py --count 1000 --vision 1 --agent-cmd ".\backtrack.exe" --seed 42 --forbid-start-hazards
```

Linux/macOS equivalents:

```bash
python3 tester_gen.py --count 1000 --vision 1 --agent-cmd "./a_star" --seed 42 --forbid-start-hazards
python3 tester_gen.py --count 1000 --vision 1 --agent-cmd "./backtrack" --seed 42 --forbid-start-hazards
```

Parameters:
- `--count N` number of maps to generate
- `--vision {1,2,both}` which perception radius to test
- `--agent-cmd` path/command to your binary
- `--seed` integer for reproducible generation (omit for random)
- `--forbid-start-hazards` try to avoid lethal start tile (best-effort)
- `--maps-dir` where to put generated maps (default `maps/generated`)
- `--answers-location {alongside,answers}` where to store reports/logs

Outputs:
- Per-run reports: either next to each generated map or in `answers/` (depends on `--answers-location`).
- Overall summary: `maps/generated/generated_summary.txt` with totals and statistics:
	- successes/failures; solvable vs unsolvable counts; total elapsed time
	- execution time mean/median/mode/stddev; wins/losses series stats

## Troubleshooting

- Map format errors: The map must be 13 rows of 13 characters; include exactly one `G` and one `M`.
- “Program exited without sending a final e <int>”: Your solution must end with `e <len>` or `e -1`.
- Ring toggle failures: The tester will fail the run if you toggle the ring into a lethal state on the current tile.
- More descriptions coming later... Possibly...