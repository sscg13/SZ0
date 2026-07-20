#!/usr/bin/env python3
"""Rescore a cutechess match log with pentanomial (paired-opening) statistics.

Cutechess reports trinomial error bars, which treat every game as independent.
With a paired-opening book (`-repeat -games 2`) consecutive games share an
opening with reversed colours, so the book's contribution cancels within each
pair. Scoring per pair removes that variance instead of paying for it. On the
unbalanced shatranj book this is worth 20-30% of the standard error.

Usage:
    python testing/pentanomial.py match.log [match2.log ...]
    python testing/pentanomial.py --engine SZ0-test match.log

Round robins are handled: games are grouped into head-to-head encounters by
engine pair and each encounter is reported separately. Note these are pairwise
results, NOT the joint round-robin fit that experiments.md usually quotes --
use ordo/bayeselo for that.

Pairing assumes games 2k-1 and 2k share an opening, which is what cutechess
emits under `-repeat`. Each pair is validated (same two engines, reversed
colours) and unpaired games are reported and excluded rather than silently
mis-scored.
"""

import math
import re
import sys

GAME = re.compile(
    r"Finished game (\d+) \((.+?) vs (.+?)\): (1-0|0-1|1/2-1/2)"
)


def parse(path):
    """Return [(game_number, white, black, result)] sorted by game number.

    Games are keyed by number because `-concurrency` lets them finish out of
    order.
    """
    games = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = GAME.search(line)
            if m:
                games[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
    return [(n,) + games[n] for n in sorted(games)]


def score_for(engine, white, black, result):
    """Score of `engine` in one game, or None if it did not play."""
    if engine == white:
        return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    if engine == black:
        return {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
    return None


def build_pairs(games):
    """Split games into validated opening pairs.

    Returns (pairs, unpaired) where each pair is (game_a, game_b) with the same
    two engines and reversed colours.
    """
    pairs, unpaired = [], []
    i = 0
    while i < len(games):
        if i + 1 >= len(games):
            unpaired.append(games[i])
            break
        a, b = games[i], games[i + 1]
        if {a[1], a[2]} == {b[1], b[2]} and a[1] == b[2] and a[2] == b[1]:
            pairs.append((a, b))
            i += 2
        else:
            unpaired.append(a)
            i += 1
    return pairs, unpaired


def elo(p):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / p - 1.0)


def stats(values):
    """Mean, and standard error of the mean, of a sample."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("inf")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def line(label, mu, se):
    lo, hi = mu - 1.959964 * se, mu + 1.959964 * se
    los = 0.5 * (1.0 + math.erf((mu - 0.5) / (se * math.sqrt(2.0))))
    print(f"  {label}  Elo {elo(mu):+6.1f} +/- {(elo(hi) - elo(lo)) / 2:4.1f}"
          f"   [{elo(lo):+6.1f}, {elo(hi):+6.1f}]   LOS {los * 100:6.2f}%")


def report_encounter(engine, opponent, pairs):
    """Report one head-to-head, from `engine`'s perspective."""
    per_game, per_pair = [], []
    for a, b in pairs:
        sa = score_for(engine, a[1], a[2], a[3])
        sb = score_for(engine, b[1], b[2], b[3])
        per_game += [sa, sb]
        per_pair.append((sa + sb) / 2.0)

    mu, se_tri = stats(per_game)
    _, se_pen = stats(per_pair)
    counts = [sum(1 for s in per_pair if abs(s - k) < 1e-9)
              for k in (0.0, 0.25, 0.5, 0.75, 1.0)]

    print(f"{engine} vs {opponent}: "
          f"{len(per_game)} games ({len(per_pair)} pairs), score {mu:.4f}")
    print(f"  pentanomial [LL, LD, DD, DW, WW] = {counts}")
    line("trinomial  ", mu, se_tri)
    line("pentanomial", mu, se_pen)
    print(f"  standard error reduced {100 * (1 - se_pen / se_tri):.1f}% "
          f"by pairing")
    print()


def report(path, engine=None):
    games = parse(path)
    print(f"=== {path} ===")
    if not games:
        print("  no 'Finished game' lines found\n")
        return

    pairs, unpaired = build_pairs(games)
    if unpaired:
        shown = ", ".join(str(g[0]) for g in unpaired[:8])
        print(f"  warning: {len(unpaired)} of {len(games)} games could not be "
              f"paired (game numbers {shown}"
              f"{', ...' if len(unpaired) > 8 else ''}) and were excluded.")
        if not pairs:
            print("  no valid pairs -- was this run without `-repeat`? "
                  "Only the trinomial bar is meaningful.\n")
            return
    print()

    # Group pairs into head-to-head encounters, preserving first-seen order.
    encounters = {}
    for pair in pairs:
        key = frozenset((pair[0][1], pair[0][2]))
        if key not in encounters:
            encounters[key] = (pair[0][1], pair[0][2], [])
        encounters[key][2].append(pair)

    for first, second, enc_pairs in encounters.values():
        if engine is None:
            a, b = first, second
        elif engine == first:
            a, b = first, second
        elif engine == second:
            a, b = second, first
        else:
            continue
        report_encounter(a, b, enc_pairs)

    if len(encounters) > 1:
        print("  note: these are pairwise head-to-head results, not a joint")
        print("  round-robin fit -- use ordo/bayeselo for that.\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    chosen = None
    if "--engine" in args:
        i = args.index("--engine")
        chosen = args[i + 1]
        del args[i:i + 2]
    if not args:
        raise SystemExit(__doc__)
    for path in args:
        report(path, chosen)
