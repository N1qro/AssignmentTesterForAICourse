#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple
import time

# Reuse the existing tester logic (Interactor, parsing, hazards, etc.)
import tester
import subprocess


WIDTH = HEIGHT = 13

TOKENS = {
    'EMPTY': '-',
    'ORC': 'O',
    'URUK': 'U',
    'NAZGUL': 'N',
    'WATCH': 'W',
    'GOLLUM': 'G',
    'MOUNT': 'M',
    'COAT': 'C',
}


def make_empty_grid() -> List[List[str]]:
    return [[TOKENS['EMPTY'] for _ in range(WIDTH)] for _ in range(HEIGHT)]


def random_free_cell(occupied: set[Tuple[int,int]]) -> Tuple[int,int]:
    while True:
        x = random.randrange(HEIGHT)
        y = random.randrange(WIDTH)
        if (x, y) not in occupied:
            return x, y


def generate_map(include_coat_prob: float = 0.35,
                 orc_range=(4, 12), uruk_range=(0, 4),
                 nazgul_range=(0, 2), watch_range=(0, 3),
                 forbid_start_hazards: bool = False,
                 enforce_safe_specials: bool = True,
                 seed: int | None = None) -> List[str]:
    """
    Generate a single 13x13 map with at least one Gollum 'G' and Mount Doom 'M'.
    May optionally include a Coat 'C' and a variety of enemies.

    - Start at (0,0) is always empty.
    - Ensures no duplicate placements.
    - If forbid_start_hazards=True, avoid placing an enemy at (0,0) and try to
      avoid immediate lethal coverage at (0,0) for common enemies (best-effort).
    """
    if seed is not None:
        random.seed(seed)

    grid = make_empty_grid()
    occupied: set[Tuple[int,int]] = {(0, 0)}  # start is reserved empty

    # Place Gollum and Mount Doom
    gx, gy = random_free_cell(occupied)
    occupied.add((gx, gy))
    grid[gx][gy] = TOKENS['GOLLUM']

    mx, my = random_free_cell(occupied)
    occupied.add((mx, my))
    grid[mx][my] = TOKENS['MOUNT']

    # Optionally place Coat
    if random.random() < include_coat_prob:
        cx, cy = random_free_cell(occupied)
        occupied.add((cx, cy))
        grid[cx][cy] = TOKENS['COAT']

    # Enemy counts
    num_orc = random.randint(*orc_range)
    num_uruk = random.randint(*uruk_range)
    num_naz = random.randint(*nazgul_range)
    num_watch = random.randint(*watch_range)

    # Place enemies
    def place_many(count: int, token: str):
        for _ in range(count):
            x, y = random_free_cell(occupied)
            occupied.add((x, y))
            grid[x][y] = token

    place_many(num_orc, TOKENS['ORC'])
    place_many(num_uruk, TOKENS['URUK'])
    place_many(num_naz, TOKENS['NAZGUL'])
    place_many(num_watch, TOKENS['WATCH'])

    # Best-effort: ensure start is not immediately lethal without ring/coat if requested
    if forbid_start_hazards:
        hazards = tester.compute_hazards([''.join(row) for row in grid], ring_on=False, coat_on=False)
        if (0, 0) in hazards:
            # Try to relocate one nearby enemy causing the hazard; if can't, accept as-is
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx, ny = 0 + dx, 0 + dy
                if 0 <= nx < HEIGHT and 0 <= ny < WIDTH and grid[nx][ny] in (TOKENS['ORC'], TOKENS['URUK'], TOKENS['NAZGUL'], TOKENS['WATCH']):
                    # move it to a free place
                    grid[nx][ny] = TOKENS['EMPTY']
                    occupied.remove((nx, ny))
                    rx, ry = random_free_cell(occupied)
                    occupied.add((rx, ry))
                    grid[rx][ry] = random.choice([TOKENS['ORC'], TOKENS['URUK'], TOKENS['NAZGUL'], TOKENS['WATCH']])
                    break

    # Enforce that special cells (start, G, M, C if present) are safe with ring on/off
    if enforce_safe_specials:
        specials = {(0, 0), (gx, gy), (mx, my)}
        # Find coat position, if any
        c_pos = None
        for x in range(HEIGHT):
            for y in range(WIDTH):
                if grid[x][y] == TOKENS['COAT']:
                    c_pos = (x, y)
                    break
            if c_pos is not None:
                break
        if c_pos is not None:
            specials.add(c_pos)

        def specials_safe() -> bool:
            lines = [''.join(row) for row in grid]
            haz_no_ring = tester.compute_hazards(lines, ring_on=False, coat_on=False)
            haz_ring = tester.compute_hazards(lines, ring_on=True, coat_on=False)
            for s in specials:
                if s in haz_no_ring or s in haz_ring:
                    return False
            return True

        def nearest_enemy_to(offender: tuple[int,int]) -> tuple[int,int] | None:
            ox, oy = offender
            best = None
            bestd = 10**9
            for x in range(HEIGHT):
                for y in range(WIDTH):
                    if grid[x][y] in (TOKENS['ORC'], TOKENS['URUK'], TOKENS['NAZGUL'], TOKENS['WATCH']):
                        d = abs(x-ox) + abs(y-oy)
                        if d < bestd:
                            bestd = d
                            best = (x, y)
            return best

        tries = 0
        max_tries = 2000
        while not specials_safe() and tries < max_tries:
            tries += 1
            lines = [''.join(row) for row in grid]
            haz_no_ring = tester.compute_hazards(lines, ring_on=False, coat_on=False)
            haz_ring = tester.compute_hazards(lines, ring_on=True, coat_on=False)
            offenders = [s for s in specials if s in haz_no_ring or s in haz_ring]
            if not offenders:
                break
            # Pick the first offender and move the nearest enemy elsewhere
            target = offenders[0]
            epos = nearest_enemy_to(target)
            if epos is None:
                break
            ex, ey = epos
            # Vacate enemy
            grid[ex][ey] = TOKENS['EMPTY']
            occupied.remove((ex, ey))
            # Find a new free slot far-ish from specials by random attempts
            for _ in range(200):
                nx, ny = random_free_cell(occupied)
                if (nx, ny) in specials:
                    continue
                # avoid placing on G/M/C
                if grid[nx][ny] != TOKENS['EMPTY']:
                    continue
                occupied.add((nx, ny))
                # Reuse same enemy type; approximate: pick a random enemy type
                # Alternatively, we could have remembered token at (ex,ey) before clearing
                # Let's track it: set token before clearing above
                break
            else:
                # If failed to find a place, restore and break
                grid[ex][ey] = random.choice([TOKENS['ORC'], TOKENS['URUK'], TOKENS['NAZGUL'], TOKENS['WATCH']])
                occupied.add((ex, ey))
                break
            # Place a random enemy type
            grid[nx][ny] = random.choice([TOKENS['ORC'], TOKENS['URUK'], TOKENS['NAZGUL'], TOKENS['WATCH']])

    return [''.join(row) for row in grid]


def write_map(path: Path, grid_lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for line in grid_lines:
            f.write(line + '\n')


class BulkInteractor:
    """
    A slightly safer interactor for bulk runs, based on tester.Interactor but
    avoids blocking on stderr reads if the agent hasn't exited yet.
    """
    def __init__(self, agent_cmd: List[str], vision:int, grid:List[str], g_pos:Tuple[int,int], m_pos:Tuple[int,int], answers_dir: Path, run_id: str):
        # Reuse tester.Interactor state and helpers by composition
        self._inner = tester.Interactor(agent_cmd, vision, grid, g_pos, m_pos, answers_dir, run_id)

    @property
    def ok(self):
        return self._inner.ok

    @property
    def report_path(self):
        return self._inner.report_path

    def run(self):
        # Inline the control flow, but fix the finally block ordering
        it = self._inner
        it.start_agent()
        try:
            it.initial_handshake()
            while True:
                it.steps += 1
                line = it.read_from_agent()
                if not line:  # EOF
                    it.fail_reason = 'Program exited without sending a final e <int>'
                    break
                line = line.strip()
                if not line:
                    it.fail_reason = 'Wrong answer: empty command'
                    break
                if line.startswith('e '):
                    it.ok = True
                    it.final_e = line
                    break
                try:
                    if line == 'r' or line == 'rr':
                        if not it.handle_toggle(line):
                            break
                        continue
                    parts = line.split()
                    if parts[0] == 'm':
                        if not it.handle_move(parts):
                            break
                        continue
                except RuntimeError:
                    break
                it.fail_reason = f"Wrong answer: unknown command '{line}'"
                break
        finally:
            # Ensure the agent process is terminated first to avoid blocking reads
            try:
                if it.proc and it.proc.poll() is None:
                    it.proc.kill()
            except Exception:
                pass

            # Collect any available stderr without blocking forever
            stderr = ''
            try:
                # Use communicate with timeout to avoid hangs
                if it.proc:
                    try:
                        _, stderr = it.proc.communicate(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        # Best-effort: read non-blocking by setting to empty
                        stderr = ''
            except Exception:
                stderr = ''

            # Report
            with it.report_path.open('w', encoding='utf-8') as rpt:
                if it.ok:
                    final_line = it.final_e.strip() if it.final_e else 'e <int>'
                    rpt.write(f'Result: SUCCESS (accepted {final_line})\n')
                else:
                    rpt.write('Result: FAILURE\n')
                    rpt.write(f'Reason: {it.fail_reason or "(unknown)"}\n')
                rpt.write(f'Steps processed: {it.steps}\n')
                rpt.write(f'Agent position: {it.pos}\n')
                rpt.write(f'Ring on: {it.ring_on}\n')
                rpt.write(f'Coat on: {it.coat_on}\n')
                rpt.write(f'Gollum found: {it.g_found}\n')
                if stderr:
                    rpt.write('\nAgent stderr (if any):\n')
                    rpt.write(stderr)


def run_single(agent_cmd: str, map_path: Path, answers_dir: Path, vision: int) -> tuple[bool, Path, str | None, float]:
    grid, g_pos, m_pos, _ = tester.parse_map(map_path)

    # Prepare interactor
    import shlex
    agent = shlex.split(agent_cmd)
    run_id = f"{map_path.stem}_v{vision}"
    it = BulkInteractor(agent, vision, grid, g_pos, m_pos, answers_dir, run_id)

    t0 = time.time()
    it.run()
    dt = time.time() - t0
    final_e = getattr(it._inner, 'final_e', None)
    final_e = final_e.strip() if isinstance(final_e, str) else None
    return it.ok, it.report_path, final_e, dt


def main():
    parser = argparse.ArgumentParser(description='Bulk map generator and runner for Ring Destroyer assignment')
    parser.add_argument('--count', type=int, default=1000, help='Number of maps to generate')
    parser.add_argument('--maps-dir', type=Path, default=Path('maps/generated'), help='Directory to write generated maps')
    parser.add_argument('--answers-location', choices=['alongside', 'answers'], default='alongside', help='Where to store reports/logs: alongside maps or in answers/')
    parser.add_argument('--vision', choices=['1','2','both'], default='both', help='Which vision(s) to run')
    parser.add_argument('--agent-cmd', type=str, default='go run .', help='Command to run the agent')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation')
    parser.add_argument('--forbid-start-hazards', action='store_true', help='Try to avoid lethal start tile (0,0)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    maps_dir: Path = args.maps_dir
    maps_dir.mkdir(parents=True, exist_ok=True)

    if args.answers_location == 'alongside':
        answers_dir_base = maps_dir
    else:
        answers_dir_base = Path('answers')
        answers_dir_base.mkdir(parents=True, exist_ok=True)

    visions = [1, 2] if args.vision == 'both' else [int(args.vision)]

    total_runs = 0
    failures = 0
    successes = 0
    solvable = 0    # e != -1
    unsolvable = 0  # e == -1

    per_run_times: List[float] = []
    wins_series: List[int] = []
    losses_series: List[int] = []

    summary_lines: List[str] = []
    t0 = time.time()

    for i in range(1, args.count + 1):
        base = f"map_{i:04d}"
        map_path = maps_dir / f"{base}.txt"
        # Generate
        grid = generate_map(forbid_start_hazards=args.forbid_start_hazards, enforce_safe_specials=True)
        write_map(map_path, grid)

        # Run for each vision
        for v in visions:
            answers_dir = answers_dir_base  # one dir per run_id via Interactor
            ok, report_path, final_e, dt_run = run_single(args.agent_cmd, map_path, answers_dir, v)
            total_runs += 1
            per_run_times.append(dt_run)
            wins_series.append(1 if ok else 0)
            losses_series.append(0 if ok else 1)
            if ok:
                successes += 1
                if final_e is not None:
                    try:
                        val = int(final_e.split()[1])
                        if val == -1:
                            unsolvable += 1
                        else:
                            solvable += 1
                    except Exception:
                        pass
            else:
                failures += 1
                # Append minimal failure summary line
                summary_lines.append(f"FAIL	{map_path.name}	vision={v}	see={report_path.name}")

    dt = time.time() - t0

    # Write overall summary next to generated maps
    summary_path = maps_dir / 'generated_summary.txt'
    # Stats helpers
    import statistics
    from collections import Counter

    def safe_mode(seq):
        if not seq:
            return 'n/a'
        try:
            return statistics.mode(seq)
        except statistics.StatisticsError:
            c = Counter(seq)
            maxc = max(c.values())
            modes = [k for k, v in c.items() if v == maxc]
            try:
                return min(modes)
            except TypeError:
                return modes[0]

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    seed_used = str(args.seed) if args.seed is not None else 'random'

    with summary_path.open('w', encoding='utf-8') as f:
        f.write(f"Seed: {seed_used}\n")
        f.write(f"Total maps: {args.count}\n")
        f.write(f"Total runs: {total_runs} (visions={visions})\n")
        f.write(f"Successes: {successes}\n")
        f.write(f"Failures: {failures}\n")
        f.write(f"Solvable (e != -1): {solvable}\n")
        f.write(f"Unsolvable (e == -1): {unsolvable}\n")
        f.write(f"Elapsed (total): {dt:.2f}s\n\n")

        if summary_lines:
            f.write("Failures list:\n")
            for line in summary_lines:
                f.write(line + "\n")

        if per_run_times:
            f.write("\nStatistics (over runs):\n")
            f.write("Execution time (s):\n")
            f.write(f"  mean: {fmt(statistics.mean(per_run_times))}\n")
            f.write(f"  median: {fmt(statistics.median(per_run_times))}\n")
            f.write(f"  mode: {fmt(safe_mode(per_run_times))}\n")
            f.write(f"  stddev: {fmt(statistics.pstdev(per_run_times))}\n")

            f.write("Wins (1 if accepted):\n")
            f.write(f"  mean: {fmt(statistics.mean(wins_series))}\n")
            f.write(f"  median: {fmt(statistics.median(wins_series))}\n")
            f.write(f"  mode: {fmt(safe_mode(wins_series))}\n")
            f.write(f"  stddev: {fmt(statistics.pstdev(wins_series))}\n")

            f.write("Losses (1 if failed):\n")
            f.write(f"  mean: {fmt(statistics.mean(losses_series))}\n")
            f.write(f"  median: {fmt(statistics.median(losses_series))}\n")
            f.write(f"  mode: {fmt(safe_mode(losses_series))}\n")
            f.write(f"  stddev: {fmt(statistics.pstdev(losses_series))}\n")

    print(f"Done. Maps in: {maps_dir}")
    print(f"Reports/logs in: {answers_dir_base}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
