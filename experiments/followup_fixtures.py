"""Invented graph repair histories and independently scored cache queries."""

import json
import random


KINDS = ("components", "bipartite", "distance")
PREFIX = """import sys
from collections import deque

def solve():
    values = list(map(int, sys.stdin.buffer.read().split()))
    n, m = values[:2]
    graph = [[] for _ in range(n)]
    for index in range(m):
        a, b = values[2 + 2 * index:4 + 2 * index]
        a -= 1
        b -= 1
        graph[a].append(b)
        graph[b].append(a)
"""
BODIES = {
    "components": """    visited = [False] * n
    answer = 0
    for root in range(n):
        if visited[root]:
            continue
        answer += 1
        visited[root] = True
        pending = [root]
        while pending:
            vertex = pending.pop()
            for neighbour in graph[vertex]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    pending.append(neighbour)
""",
    "bipartite": """    colour = [-1] * n
    answer = 1
    for root in range(n):
        if colour[root] != -1:
            continue
        colour[root] = 0
        pending = deque([root])
        while pending:
            vertex = pending.popleft()
            for neighbour in graph[vertex]:
                if colour[neighbour] == -1:
                    colour[neighbour] = colour[vertex] ^ 1
                    pending.append(neighbour)
                elif colour[neighbour] == colour[vertex]:
                    answer = 0
""",
    "distance": """    distance = [-1] * n
    distance[0] = 0
    pending = deque([0])
    while pending:
        vertex = pending.popleft()
        for neighbour in graph[vertex]:
            if distance[neighbour] == -1:
                distance[neighbour] = distance[vertex] + 1
                pending.append(neighbour)
    answer = distance[n - 1]
"""}


def program(kind, offset=0):
    return PREFIX+BODIES[kind]+f"    print(answer + ({offset}))\n\nsolve()\n"


def oracle(kind, n, edges):
    """Union-find, parity union-find, or Bellman-Ford; reference programs use BFS/DFS."""
    parent, parity = list(range(n)), [0]*n
    def find(x):
        colour = 0
        while parent[x] != x:
            colour ^= parity[x]
            x = parent[x]
        return x, colour
    if kind in ("components", "bipartite"):
        possible = 1
        for a, b in edges:
            ra, pa = find(a-1)
            rb, pb = find(b-1)
            if ra == rb:
                possible &= pa != pb
            else:
                parent[ra], parity[ra] = rb, pa ^ pb ^ 1
        return len({find(i)[0] for i in range(n)}) if kind == "components" else int(possible)
    if kind != "distance":
        raise ValueError(kind)
    distances = [n+1]*n
    distances[0] = 0
    for _ in range(n-1):
        changed = False
        for a, b in edges:
            a, b = a-1, b-1
            old = distances[a], distances[b]
            distances[a] = min(distances[a], old[1]+1)
            distances[b] = min(distances[b], old[0]+1)
            changed |= old != (distances[a], distances[b])
        if not changed:
            break
    return -1 if distances[-1] == n+1 else distances[-1]


def graph(n, edges):
    return f"{n} {len(edges)}\n"+"".join(f"{a} {b}\n" for a, b in edges)


def small_cases(kind):
    shapes = [(1, []), (4, []), (3, [(1, 2), (2, 3)]),
        (6, [(i, i+1) for i in range(1, 6)]), (5, [(1, 1), (2, 3)]),
        (3, [(1, 2), (2, 3), (3, 1)]), (6, [(1, 6), (1, 2), (3, 4)]),
        (6, [(1, 2), (1, 2), (2, 6)])]
    return [(graph(n, edges), oracle(kind, n, edges)) for n, edges in shapes]


def statement(kind):
    goal = {"components": "Print the number of connected components, including isolated vertices.",
        "bipartite": "Print 1 if the whole graph is bipartite, otherwise print 0.",
        "distance": "Print the shortest distance in edges from vertex 1 to vertex n, or -1 if unreachable."}[kind]
    return ("Read n m followed by m undirected edges. Vertices are numbered 1 through n. "
        "1 <= n <= 200000; 0 <= m <= 200000. Self-loops and repeated edges are allowed. "
        +goal+" Read all input from stdin, write one integer to stdout. Use iterative traversal. "
        "Example input:\n"+small_cases(kind)[2][0]+"Example output:\n"+str(small_cases(kind)[2][1])+"\n")


def history(kind, variant=0, capture_edges=240):
    """Four seeded faulty full programs with substantive, valid public input captures.

    These are fabricated histories, not claims about actual acquisition attempts.
    Offsets deliberately make each recorded failure independently checkable.
    """
    messages = [{"role": "system", "content": "Follow the requested output format. No tools are available."},
                {"role": "user", "content": statement(kind)+"Return a complete Python program. Code only."}]
    for attempt, offset in enumerate((-1, 1, 2, 3)):
        stdin, answer = large_case(kind, variant, attempt, capture_edges)
        messages += [{"role": "assistant", "content": program(kind, offset)},
            {"role": "user", "content": "The published visible case failed.\nstdin:\n"+stdin+
             f"Expected stdout: {answer}\nObserved stdout: {answer+offset}\nReturn the complete corrected Python program. Code only."}]
    return messages


def large_case(kind, variant, attempt, capture_edges=240):
    rng = random.Random(5000+100*variant+attempt)
    edges = [(rng.randint(1, 112), rng.randint(1, 112)) for _ in range(capture_edges)]
    return graph(144, edges), oracle(kind, 144, edges)


def verification_task(kind, variant):
    # Both context bins use this same suite, including every visible history case.
    cases = [large_case(kind, variant, i, edges) for edges in (32, 240) for i in range(4)]+small_cases(kind)
    rng = random.Random(88000+variant)
    for _ in range(4):
        n = rng.randint(1, 9)
        edges = [(rng.randint(1, n), rng.randint(1, n)) for _ in range(rng.randint(0, 16))]
        cases.append((graph(n, edges), oracle(kind, n, edges)))
    return {"cases": [dict(index=i, input=stdin, answer=str(answer), visible=i < 8)
                      for i, (stdin, answer) in enumerate(cases)]}


def long_requests(count, capture_edges=240):
    return [dict(id=i, kind=KINDS[i % 3], messages=history(KINDS[i % 3], i//3, capture_edges),
                 seed=720000+i, temperature=.2, top_p=.95, max_tokens=768)
            for i in range(count)]


def semantic_group(kind, long):
    messages = history(kind) if long else [
        {"role": "system", "content": "Follow the requested output format. No tools are available."},
        {"role": "user", "content": statement(kind)+"Return a complete Python program."}]
    messages = [*messages, {"role": "assistant", "content": program(kind)}]
    cases = small_cases(kind)
    count = 9 if long else 33
    return [dict(id=i, kind=kind, messages=[*messages, {"role": "user", "content":
        "The last reference program is correct; earlier drafts are superseded. "
        "Compute its output for this input. Return only a JSON object with exactly one integer field, answer.\n"
        +"stdin:\n"+cases[i % len(cases)][0]}], seed=730000+i, temperature=.2, top_p=.95,
        max_tokens=128, expected_answer=cases[i % len(cases)][1]) for i in range(count)]


def semantic_verdict(text, status, expected):
    if status != "stop":
        return "incomplete"
    def pairs(items):
        if len({k for k, _ in items}) != len(items):
            raise ValueError("Duplicate keys")
        return dict(items)
    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        return "format_failure"
    if (type(value) is not dict or set(value) != {"answer"}
            or type(value["answer"]) is not int):
        return "format_failure"
    return "passed" if value["answer"] == expected else "wrong_answer"


def common_prefix(requests):
    tokens = requests[0]["prompt_ids"]
    length = min(len(r["prompt_ids"]) for r in requests)-1
    for r in requests[1:]:
        length = next((i for i in range(length) if r["prompt_ids"][i] != tokens[i]), length)
    if length <= 0:
        raise ValueError("No common forward prefix")
    return length
