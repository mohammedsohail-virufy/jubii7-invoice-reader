"""Brute-force solver for "Five friends, five emirates".

Tries every assignment of emirate, drink and job (5!^3 = 1.7M) with early pruning, and
keeps those that satisfy all 10 clues. Exactly one survivor proves the answer is unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

PEOPLE = ("Aisha", "Ben", "Chen", "Dev", "Elena")
EMIRATES = ("Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Fujairah")
DRINKS = ("karak", "coffee", "mint tea", "juice", "water")
JOBS = ("engineer", "chef", "pilot", "nurse", "designer")

Assignment = dict[str, str]


@dataclass(frozen=True)
class Person:
    name: str
    emirate: str
    drink: str
    job: str


def emirates_ok(e: Assignment) -> bool:
    return e["Aisha"] not in ("Abu Dhabi", "Dubai", "Sharjah") and e["Chen"] != "Sharjah"  # clues 1, 2


def drinks_ok(e: Assignment, d: Assignment) -> bool:
    if d["Elena"] != "mint tea" or d["Dev"] != "karak":  # clues 5 (drink), 6
        return False
    return all(
        (e[p] != "Dubai" or d[p] == "coffee") and (e[p] != "Sharjah" or d[p] == "juice")  # clues 3, 4
        for p in PEOPLE
    )


def jobs_ok(e: Assignment, d: Assignment, j: Assignment) -> bool:
    if j["Elena"] == "chef" or j["Ben"] != "nurse":  # clues 5 (job), 9
        return False
    return all(
        (j[p] != "engineer" or e[p] == "Abu Dhabi")  # clue 7
        and (j[p] != "pilot" or (e[p] == "Ajman" and d[p] == "mint tea"))  # clue 8
        and (j[p] != "designer" or d[p] == "water")  # clue 10
        for p in PEOPLE
    )


def solve() -> list[tuple[Person, ...]]:
    solutions = []
    for ep in permutations(EMIRATES):
        e = dict(zip(PEOPLE, ep, strict=True))
        if not emirates_ok(e):
            continue
        for dp in permutations(DRINKS):
            d = dict(zip(PEOPLE, dp, strict=True))
            if not drinks_ok(e, d):
                continue
            for jp in permutations(JOBS):
                j = dict(zip(PEOPLE, jp, strict=True))
                if jobs_ok(e, d, j):
                    solutions.append(tuple(Person(p, e[p], d[p], j[p]) for p in PEOPLE))
    return solutions
