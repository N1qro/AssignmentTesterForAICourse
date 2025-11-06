#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Set

WIDTH = HEIGHT = 13

# Tokens
EMPTY = '-'
PZONE = 'P'    # only in map file as hint; not used directly for hazards
ORC = 'O'
URUK = 'U'
NAZGUL = 'N'
WATCH = 'W'
GOLLUM = 'G'
MOUNT = 'M'
COAT = 'C'

# Helpers
VN_DIRS = [(1,0),(-1,0),(0,1),(0,-1)]


def in_bounds(x:int,y:int)->bool:
    return 0 <= x < HEIGHT and 0 <= y < WIDTH


def vn_within(x:int,y:int,r:int)->List[Tuple[int,int]]:
    """All cells with Manhattan distance <= r (Von Neumann neighborhood)."""
    cells = []
    for dx in range(-r, r+1):
        rem = r - abs(dx)
        for dy in range(-rem, rem+1):
            nx, ny = x+dx, y+dy
            if in_bounds(nx, ny):
                cells.append((nx, ny))
    return cells


def moore_square(x:int,y:int,r:int)->List[Tuple[int,int]]:
    cells = []
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            nx, ny = x+dx, y+dy
            if in_bounds(nx, ny):
                cells.append((nx, ny))
    return cells


def compute_hazards(grid:List[str], ring_on:bool, coat_on:bool) -> Set[Tuple[int,int]]:
    hazards: Set[Tuple[int,int]] = set()
    for x in range(HEIGHT):
        for y in range(WIDTH):
            t = grid[x][y]
            if t == ORC:
                # Base VN r=1; with ring OR coat -> r=0 (only cell)
                if ring_on or coat_on:
                    zones = {(x,y)}
                else:
                    zones = set(vn_within(x,y,1))
                zones.add((x,y))
                hazards.update(zones)
            elif t == URUK:
                # Base VN r=2; with ring OR coat -> r=1
                r = 1 if (ring_on or coat_on) else 2
                zones = set(vn_within(x,y,r))
                zones.add((x,y))
                hazards.update(zones)
            elif t == NAZGUL:
                # Base: Moore r=1 + ears at (±2,±2)
                # With coat: Moore r=1 (no ears)
                # With ring: Moore r=2 + ears at (±3,±3)
                if ring_on:
                    zones = set(moore_square(x,y,2))
                    for ex,ey in [(3,3),(-3,3),(3,-3),(-3,-3)]:
                        nx, ny = x+ex, y+ey
                        if in_bounds(nx, ny):
                            zones.add((nx,ny))
                else:
                    zones = set(moore_square(x,y,1))
                    if not coat_on:
                        for ex,ey in [(2,2),(-2,2),(2,-2),(-2,-2)]:
                            nx, ny = x+ex, y+ey
                            if in_bounds(nx, ny):
                                zones.add((nx,ny))
                zones.add((x,y))
                hazards.update(zones)
            elif t == WATCH:
                # Base: Moore r=2
                # With ring: Moore r=2 + ears at (±3,±3)
                zones = set(moore_square(x,y,2))
                if ring_on:
                    for ex,ey in [(3,3),(-3,3),(3,-3),(-3,-3)]:
                        nx, ny = x+ex, y+ey
                        if in_bounds(nx, ny):
                            zones.add((nx,ny))
                zones.add((x,y))
                hazards.update(zones)
    return hazards


def perception(grid:List[str], pos:Tuple[int,int], ring_on:bool, coat_on:bool, vision:int, g_found:bool) -> List[Tuple[int,int,str]]:
    x,y = pos
    hazards = compute_hazards(grid, ring_on, coat_on)
    results: Dict[Tuple[int,int], str] = {}
    # Scan Chebyshev r=vision, excluding self
    for dx in range(-vision, vision+1):
        for dy in range(-vision, vision+1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x+dx, y+dy
            if not in_bounds(nx, ny):
                continue
            t = grid[nx][ny]
            # Hide Gollum after he's been found (he "follows" the player)
            if g_found and t == GOLLUM:
                t = EMPTY
            # Hide Coat after it's been picked up
            if coat_on and t == COAT:
                t = EMPTY
            if t in (ORC, URUK, NAZGUL, WATCH, GOLLUM, COAT) or (t == MOUNT and g_found):
                results[(nx,ny)] = t
            else:
                if (nx,ny) in hazards:
                    # If an enemy is actually there, prefer enemy token (handled above)
                    results.setdefault((nx,ny), 'P')
    # Stable order: sort by x then y
    out = [ (cx,cy,results[(cx,cy)]) for cx,cy in sorted(results.keys()) ]
    return out


def parse_map(map_path: Path) -> Tuple[List[str], Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    lines = [line.rstrip('\n') for line in map_path.read_text(encoding='utf-8').splitlines() if line.strip()!='']
    if len(lines) != HEIGHT:
        raise ValueError(f"Map must have {HEIGHT} rows, got {len(lines)}")
    for row in lines:
        if len(row) != WIDTH:
            raise ValueError(f"Each map row must have {WIDTH} columns; got {len(row)} in '{row}'")
    g_pos = m_pos = c_pos = None
    for i,row in enumerate(lines):
        for j,ch in enumerate(row):
            if ch == GOLLUM:
                g_pos = (i,j)
            elif ch == MOUNT:
                m_pos = (i,j)
            elif ch == COAT:
                c_pos = (i,j)
    if g_pos is None:
        raise ValueError("Map must contain Gollum 'G'")
    if m_pos is None:
        raise ValueError("Map must contain Mount Doom 'M'")
    return lines, g_pos, m_pos, c_pos


class Interactor:
    def __init__(self, agent_cmd: List[str], vision:int, grid:List[str], g_pos:Tuple[int,int], m_pos:Tuple[int,int], answers_dir: Path, run_id: str):
        self.agent_cmd = agent_cmd
        self.vision = vision
        self.grid = grid
        self.g_pos = g_pos
        self.m_pos = m_pos
        self.answers_dir = answers_dir
        self.ring_on = False
        self.coat_on = False
        self.pos = (0,0)
        self.g_found = False
        # Deterministic file names per map/vision (overwrite between runs)
        self.session_log = answers_dir / f"{run_id}_session.log"
        self.report_path = answers_dir / f"{run_id}_report.txt"
        self.ok = False
        self.final_e: str | None = None
        self.fail_reason = None
        self.steps = 0
        self.proc = None
        # Ensure session log is rewritten for each run (not appended across runs)
        try:
            with self.session_log.open('w', encoding='utf-8') as _clear:
                _clear.write('')
        except Exception:
            pass

    def _set_grid_cell(self, x:int, y:int, ch:str):
        # Safely mutate an immutable string row
        row = list(self.grid[x])
        row[y] = ch
        self.grid[x] = ''.join(row)

    def log(self, direction:str, text:str):
        with self.session_log.open('a', encoding='utf-8') as f:
            f.write(f"[{direction}] {text}\n")

    def write_to_agent(self, s:str):
        self.log('INTERACTOR->AGENT', s.rstrip('\n'))
        try:
            # If the agent already exited, don't attempt to write
            if self.proc.poll() is not None:
                # Try to see if it exited after printing a final e <int>
                tail = self.proc.stdout.read()
                if tail:
                    self.log('AGENT->INTERACTOR', tail.rstrip('\n'))
                    last = tail.strip().splitlines()[-1].strip()
                    if last.startswith('e '):
                        self.ok = True
                        self.final_e = last
                        return
                self.fail_reason = 'Program exited before interactor response'
                raise RuntimeError(self.fail_reason)

            self.proc.stdin.write(s)
            self.proc.stdin.flush()
        except OSError as e:
            # Pipe broken or invalid after agent exit; try to read remaining output
            tail = ''
            try:
                tail = self.proc.stdout.read()
            except Exception:
                pass
            if tail:
                self.log('AGENT->INTERACTOR', tail.rstrip('\n'))
                last = tail.strip().splitlines()[-1].strip()
                if last.startswith('e '):
                    self.ok = True
                    self.final_e = last
                    return
            self.fail_reason = f'Write failed: {e!s}'
            raise RuntimeError(self.fail_reason)

    def read_from_agent(self) -> str:
        # Read lines from agent. Lines starting with 'debug:' are logged but ignored as commands.
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return ''
            if line.startswith('debug:'):
                # Log debug lines to the session for visibility, but don't treat them as commands
                self.log('AGENT->INTERACTOR DEBUG', line.rstrip('\n'))
                continue
            self.log('AGENT->INTERACTOR', line.rstrip('\n'))
            return line

    def start_agent(self):
        self.proc = subprocess.Popen(
            self.agent_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def initial_handshake(self):
        # Print vision, Gollum coords, then initial perception
        self.write_to_agent(f"{self.vision}\n")
        gx, gy = self.g_pos
        self.write_to_agent(f"{gx} {gy}\n")
        per = perception(self.grid, self.pos, self.ring_on, self.coat_on, self.vision, self.g_found)
        self.write_to_agent(f"{len(per)}\n")
        for x,y,tok in per:
            self.write_to_agent(f"{x} {y} {tok}\n")

    def handle_toggle(self, cmd:str) -> bool:
        # Validate state
        if cmd == 'r':
            if self.ring_on:
                self.fail_reason = 'Wrong answer: ring already on'
                return False
            self.ring_on = True
        elif cmd == 'rr':
            if not self.ring_on:
                self.fail_reason = 'Wrong answer: ring already off'
                return False
            self.ring_on = False
        else:
            self.fail_reason = 'Wrong answer: invalid toggle command'
            return False
        # Check if current cell becomes lethal
        hazards = compute_hazards(self.grid, self.ring_on, self.coat_on)
        if self.pos in hazards:
            self.fail_reason = 'Wrong answer: lethal after ring toggle'
            return False
        # Output perception
        per = perception(self.grid, self.pos, self.ring_on, self.coat_on, self.vision, self.g_found)
        self.write_to_agent(f"{len(per)}\n")
        for x,y,tok in per:
            self.write_to_agent(f"{x} {y} {tok}\n")
        return True

    def handle_move(self, args:List[str]) -> bool:
        if len(args) != 3:
            self.fail_reason = 'Wrong answer: malformed move command'
            return False
        try:
            _, sx, sy = args
            nx, ny = int(sx), int(sy)
        except Exception:
            self.fail_reason = 'Wrong answer: non-integer move coordinates'
            return False
        x,y = self.pos
        # Movement validation
        if not in_bounds(nx, ny):
            self.fail_reason = 'Wrong answer: move out of bounds'
            return False
        if (nx,ny) == (x,y):
            self.fail_reason = 'Wrong answer: zero-length move'
            return False
        if abs(nx - x) + abs(ny - y) != 1:
            self.fail_reason = 'Wrong answer: non-adjacent move'
            return False
        # Lethal check at destination
        hazards = compute_hazards(self.grid, self.ring_on, self.coat_on)
        if (nx,ny) in hazards:
            self.fail_reason = 'Wrong answer: stepped into lethal cell'
            return False
        # Commit move
        self.pos = (nx,ny)
        # Auto-equip coat if on C and remove it from the grid (picked up)
        if self.grid[nx][ny] == COAT:
            self.coat_on = True
            self._set_grid_cell(nx, ny, EMPTY)
        # Perception after move
        per = perception(self.grid, self.pos, self.ring_on, self.coat_on, self.vision, self.g_found)
        self.write_to_agent(f"{len(per)}\n")
        for px,py,tok in per:
            self.write_to_agent(f"{px} {py} {tok}\n")
        # Gollum reveal flow: only after perception, print one line with Mount Doom coords
        if self.grid[nx][ny] == GOLLUM and not self.g_found:
            self.g_found = True
            mx, my = self.m_pos
            self.write_to_agent(f"{mx} {my}\n")
            # Remove Gollum from the map; he now "follows" the player
            self._set_grid_cell(nx, ny, EMPTY)
        return True

    def run(self):
        self.start_agent()
        try:
            self.initial_handshake()
            while True:
                self.steps += 1
                line = self.read_from_agent()
                if not line:  # EOF
                    self.fail_reason = 'Program exited without sending a final e <int>'
                    break
                line = line.strip()
                if not line:
                    self.fail_reason = 'Wrong answer: empty command'
                    break
                # Accept e <int> as valid completion
                if line.startswith('e '):
                    self.ok = True
                    self.final_e = line
                    break
                # Parse commands
                try:
                    if line == 'r' or line == 'rr':
                        if not self.handle_toggle(line):
                            break
                        continue
                    parts = line.split()
                    if parts[0] == 'm':
                        if not self.handle_move(parts):
                            break
                        continue
                except RuntimeError:
                    break
                # Unknown command
                self.fail_reason = f"Wrong answer: unknown command '{line}'"
                break
        finally:
            # Terminate the agent first to avoid blocking on pipe reads
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.kill()
            except Exception:
                pass

            # Best-effort: collect any remaining stderr without blocking indefinitely
            stderr = ''
            try:
                if self.proc:
                    try:
                        # Small timeout to avoid hangs
                        _, stderr = self.proc.communicate(timeout=0.2)
                    except Exception:
                        stderr = ''
            except Exception:
                stderr = ''
            # Report
            with self.report_path.open('w', encoding='utf-8') as rpt:
                if self.ok:
                    final_line = self.final_e.strip() if self.final_e else 'e <int>'
                    rpt.write(f'Result: SUCCESS (accepted {final_line})\n')
                else:
                    rpt.write('Result: FAILURE\n')
                    rpt.write(f'Reason: {self.fail_reason or "(unknown)"}\n')
                rpt.write(f'Steps processed: {self.steps}\n')
                rpt.write(f'Agent position: {self.pos}\n')
                rpt.write(f'Ring on: {self.ring_on}\n')
                rpt.write(f'Coat on: {self.coat_on}\n')
                rpt.write(f'Gollum found: {self.g_found}\n')
                if stderr:
                    rpt.write('\nAgent stderr (if any):\n')
                    rpt.write(stderr)


def main():
    ap = argparse.ArgumentParser(description='Interactive tester for Ring Destroyer assignment')
    ap.add_argument('--map', dest='map_path', type=Path, default=Path('maps/sample_map.txt'), help='Path to 13x13 map file')
    ap.add_argument('--vision', type=int, choices=[1,2], default=1, help='Agent vision radius (1 or 2)')
    ap.add_argument('--agent-cmd', type=str, default='go run .', help='Command to run the agent (reads from stdin, writes to stdout)')
    ap.add_argument('--answers-dir', type=Path, default=Path('answers'), help='Directory to write logs and reports')
    args = ap.parse_args()

    grid, g_pos, m_pos, _ = parse_map(args.map_path)

    answers_dir = args.answers_dir
    answers_dir.mkdir(parents=True, exist_ok=True)

    # Split agent command respecting spaces and quotes
    import shlex
    agent_cmd = shlex.split(args.agent_cmd)

    run_id = f"{args.map_path.stem}_v{args.vision}"
    it = Interactor(agent_cmd, args.vision, grid, g_pos, m_pos, answers_dir, run_id)
    it.run()
    print(f"Report written to: {it.report_path}")
    print(f"Transcript written to: {it.session_log}")

if __name__ == '__main__':
    main()
