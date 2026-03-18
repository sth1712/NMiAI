"""
pathfinding.py — A*-algoritme for navigasjon i butikkgrid

Koordinatsystem: (0,0) øverst til venstre, X øker til høyre, Y nedover.
"""

import heapq
from typing import Optional


def heuristikk(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan-avstand mellom to punkter."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def finn_sti(
    start: tuple[int, int],
    mål: tuple[int, int],
    hindringer: set[tuple[int, int]],
    bredde: int,
    høyde: int,
) -> list[tuple[int, int]]:
    """
    A*-algoritme. Returnerer liste med koordinater fra start til mål (ekskl. start).
    Returnerer tom liste hvis ingen sti finnes.
    """
    if start == mål:
        return []

    # (f_score, g_score, posisjon, sti_til_nå)
    åpen_kø: list[tuple[int, int, tuple[int, int], list]] = []
    heapq.heappush(åpen_kø, (heuristikk(start, mål), 0, start, []))

    besøkt: dict[tuple[int, int], int] = {}  # posisjon -> beste g_score

    naboer = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # opp, ned, venstre, høyre

    while åpen_kø:
        f, g, pos, sti = heapq.heappop(åpen_kø)

        if pos in besøkt and besøkt[pos] <= g:
            continue
        besøkt[pos] = g

        if pos == mål:
            return sti + [pos]

        x, y = pos
        for dx, dy in naboer:
            nx, ny = x + dx, y + dy
            nabo = (nx, ny)

            if not (0 <= nx < bredde and 0 <= ny < høyde):
                continue
            if nabo in hindringer:
                continue

            ny_g = g + 1
            if nabo in besøkt and besøkt[nabo] <= ny_g:
                continue

            ny_f = ny_g + heuristikk(nabo, mål)
            heapq.heappush(åpen_kø, (ny_f, ny_g, nabo, sti + [nabo]))

    return []  # Ingen sti funnet


def sti_til_aksjoner(
    sti: list[tuple[int, int]], nåværende: tuple[int, int]
) -> list[str]:
    """
    Konverter en liste koordinater til bot-aksjoner (move_up/down/left/right).
    """
    aksjoner = []
    forrige = nåværende
    for pos in sti:
        dx = pos[0] - forrige[0]
        dy = pos[1] - forrige[1]

        if dx == 1:
            aksjoner.append("move_right")
        elif dx == -1:
            aksjoner.append("move_left")
        elif dy == 1:
            aksjoner.append("move_down")
        elif dy == -1:
            aksjoner.append("move_up")

        forrige = pos
    return aksjoner


def nærmeste_mål(
    start: tuple[int, int],
    kandidater: list[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Finn kandidaten med lavest Manhattan-avstand til start."""
    if not kandidater:
        return None
    return min(kandidater, key=lambda k: heuristikk(start, k))
