
# ============================================================
# EMERGENCY AMBULANCE ROUTE OPTIMIZATION
# PURE PYTHON IMPLEMENTATION
# ============================================================
# Algorithms:
# 1. Breadth First Search (BFS)
# 2. Depth First Search (DFS)
# 3. Depth Limited Search (DLS)
# 4. Iterative Deepening Search (IDS)
# 5. Bidirectional Search
# 6. Greedy Best-First Search
# 7. A* Search
# 8. Hill Climbing
# 9. Genetic Algorithm
#
# Dijkstra is used as a benchmark for minimum travel-time cost.

# ============================================================

from collections import deque
import heapq
import math
import random
import time


# ============================================================
# 1. EMERGENCY ROAD DATA
# ============================================================
# (Location 1, Location 2, distance in km, speed in km/h,
#  congestion factor)
#
# The dataset is synthetic and intended for academic use.
# ============================================================

ROAD_DATA = [
    ("Ambulance Station", "A", 2.0, 40, 1.10),
    ("Ambulance Station", "B", 2.5, 50, 1.00),
    ("Ambulance Station", "C", 3.0, 40, 1.40),

    ("A", "D", 2.0, 40, 1.00),
    ("A", "E", 2.8, 45, 1.30),

    ("B", "E", 1.8, 35, 1.60),
    ("B", "F", 2.7, 50, 1.00),

    ("C", "F", 2.0, 40, 1.10),
    ("C", "G", 3.2, 45, 1.00),

    ("D", "E", 1.5, 30, 1.20),
    ("D", "H", 2.5, 40, 1.00),

    ("E", "F", 1.6, 35, 1.50),
    ("E", "H", 2.0, 40, 1.00),
    ("E", "I", 2.8, 50, 1.20),

    ("F", "G", 1.5, 45, 1.30),
    ("F", "I", 2.2, 40, 1.00),

    ("G", "I", 1.7, 45, 1.00),
    ("G", "J", 2.5, 40, 1.20),

    ("H", "I", 1.6, 35, 1.10),
    ("H", "K", 2.8, 50, 1.00),

    ("I", "J", 1.5, 45, 1.00),
    ("I", "K", 2.0, 40, 1.30),

    ("J", "K", 1.4, 35, 1.20),
    ("J", "Emergency Location", 3.0, 50, 1.10),
    ("K", "Emergency Location", 2.2, 40, 1.00),
]


# ============================================================
# 2. BUILD GRAPH
# ============================================================

graph = {}

for u, v, distance, speed, congestion in ROAD_DATA:

    travel_time = (
        distance / speed
    ) * 60 * congestion

    info = {
        "distance": distance,
        "speed": speed,
        "congestion": congestion,
        "time": travel_time
    }

    if u not in graph:
        graph[u] = {}

    if v not in graph:
        graph[v] = {}

    graph[u][v] = info
    graph[v][u] = info


ALL_NODES = list(graph.keys())

DEFAULT_START = "Ambulance Station"
DEFAULT_GOAL = "Emergency Location"


# ============================================================
# 3. BASIC FUNCTIONS
# ============================================================

def edge_info(u, v):
    return graph[u][v]


def route_distance(path):
    if not path:
        return 0.0

    total = 0.0

    for i in range(len(path) - 1):
        total += edge_info(
            path[i],
            path[i + 1]
        )["distance"]

    return total


def route_cost(path):
    if not path:
        return 0.0

    total = 0.0

    for i in range(len(path) - 1):
        total += edge_info(
            path[i],
            path[i + 1]
        )["time"]

    return total


def is_valid_path(path, start, goal):

    if not path:
        return False

    if path[0] != start:
        return False

    if path[-1] != goal:
        return False

    if len(path) != len(set(path)):
        return False

    for i in range(len(path) - 1):

        if path[i + 1] not in graph[path[i]]:
            return False

    return True


# ============================================================
# 4. FREE-FLOW HEURISTIC
# ============================================================

def free_flow_time(u, v):

    info = edge_info(u, v)

    return (
        info["distance"]
        / info["speed"]
    ) * 60


def free_flow_heuristic_table(goal):

    distances = {}

    for node in ALL_NODES:
        distances[node] = float("inf")

    distances[goal] = 0.0

    priority_queue = [
        (0.0, goal)
    ]

    while priority_queue:

        current_cost, current = heapq.heappop(
            priority_queue
        )

        if current_cost > distances[current]:
            continue

        for neighbor in graph[current]:

            new_cost = (
                current_cost
                + free_flow_time(
                    current,
                    neighbor
                )
            )

            if new_cost < distances[neighbor]:

                distances[neighbor] = new_cost

                heapq.heappush(
                    priority_queue,
                    (
                        new_cost,
                        neighbor
                    )
                )

    return distances


# ============================================================
# 5. BFS
# ============================================================

def bfs(start, goal):

    queue = deque()

    queue.append(
        (start, [start])
    )

    visited = {start}

    expanded = []

    while queue:

        current, path = queue.popleft()

        expanded.append(current)

        if current == goal:
            return path, expanded

        for neighbor in graph[current]:

            if neighbor not in visited:

                visited.add(neighbor)

                queue.append(
                    (
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, expanded


# ============================================================
# 6. DFS
# ============================================================

def dfs(start, goal):

    stack = [
        (start, [start])
    ]

    visited = set()

    expanded = []

    while stack:

        current, path = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        expanded.append(current)

        if current == goal:
            return path, expanded

        neighbors = list(
            graph[current].keys()
        )

        for neighbor in reversed(neighbors):

            if neighbor not in visited:

                stack.append(
                    (
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, expanded


# ============================================================
# 7. DEPTH LIMITED SEARCH
# ============================================================

def dls(start, goal, limit):

    expanded = []

    def search(current, path, depth):

        expanded.append(current)

        if current == goal:
            return path

        if depth == limit:
            return None

        for neighbor in graph[current]:

            if neighbor not in path:

                result = search(
                    neighbor,
                    path + [neighbor],
                    depth + 1
                )

                if result is not None:
                    return result

        return None

    path = search(
        start,
        [start],
        0
    )

    return path, expanded


# ============================================================
# 8. ITERATIVE DEEPENING SEARCH
# ============================================================

def ids(start, goal, max_depth=20):

    all_expanded = []

    for depth in range(
        max_depth + 1
    ):

        path, expanded = dls(
            start,
            goal,
            depth
        )

        all_expanded.extend(
            expanded
        )

        if path is not None:
            return path, all_expanded

    return None, all_expanded


# ============================================================
# 9. BIDIRECTIONAL SEARCH
# ============================================================

def bidirectional_search(start, goal):

    if start == goal:
        return [start], [start]

    forward_queue = deque([start])
    backward_queue = deque([goal])

    forward_parent = {
        start: None
    }

    backward_parent = {
        goal: None
    }

    expanded = []

    meeting = None

    while (
        forward_queue
        and backward_queue
    ):

        current = forward_queue.popleft()

        expanded.append(current)

        for neighbor in graph[current]:

            if neighbor not in forward_parent:

                forward_parent[neighbor] = current

                forward_queue.append(
                    neighbor
                )

                if neighbor in backward_parent:

                    meeting = neighbor
                    break

        if meeting is not None:
            break

        current = backward_queue.popleft()

        expanded.append(current)

        for neighbor in graph[current]:

            if neighbor not in backward_parent:

                backward_parent[neighbor] = current

                backward_queue.append(
                    neighbor
                )

                if neighbor in forward_parent:

                    meeting = neighbor
                    break

        if meeting is not None:
            break

    if meeting is None:
        return None, expanded

    left = []

    current = meeting

    while current is not None:

        left.append(current)

        current = forward_parent[current]

    left.reverse()

    right = []

    current = backward_parent.get(
        meeting
    )

    while current is not None:

        right.append(current)

        current = backward_parent[current]

    return left + right, expanded


# ============================================================
# 10. GREEDY BEST-FIRST SEARCH
# ============================================================

def greedy_best_first(start, goal):

    heuristic_table = (
        free_flow_heuristic_table(goal)
    )

    counter = 0

    priority_queue = [
        (
            heuristic_table[start],
            counter,
            start,
            [start]
        )
    ]

    visited = set()

    expanded = []

    while priority_queue:

        _, _, current, path = heapq.heappop(
            priority_queue
        )

        if current in visited:
            continue

        visited.add(current)

        expanded.append(current)

        if current == goal:
            return path, expanded

        for neighbor in graph[current]:

            if neighbor not in visited:

                counter += 1

                heapq.heappush(
                    priority_queue,
                    (
                        heuristic_table[neighbor],
                        counter,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, expanded


# ============================================================
# 11. A* SEARCH
# ============================================================

def a_star(start, goal):

    heuristic_table = (
        free_flow_heuristic_table(goal)
    )

    counter = 0

    priority_queue = [
        (
            heuristic_table[start],
            0.0,
            counter,
            start,
            [start]
        )
    ]

    best_g = {
        start: 0.0
    }

    expanded = []

    while priority_queue:

        f, g, _, current, path = (
            heapq.heappop(priority_queue)
        )

        if g > best_g.get(
            current,
            float("inf")
        ):
            continue

        expanded.append(current)

        if current == goal:
            return path, expanded

        for neighbor, info in (
            graph[current].items()
        ):

            new_g = (
                g + info["time"]
            )

            if new_g < best_g.get(
                neighbor,
                float("inf")
            ):

                best_g[neighbor] = new_g

                counter += 1

                new_f = (
                    new_g
                    + heuristic_table[neighbor]
                )

                heapq.heappush(
                    priority_queue,
                    (
                        new_f,
                        new_g,
                        counter,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, expanded


# ============================================================
# 12. DIJKSTRA BENCHMARK
# ============================================================

def dijkstra(start, goal):

    counter = 0

    priority_queue = [
        (
            0.0,
            counter,
            start,
            [start]
        )
    ]

    best_cost = {
        start: 0.0
    }

    expanded = []

    while priority_queue:

        cost, _, current, path = (
            heapq.heappop(priority_queue)
        )

        if cost > best_cost.get(
            current,
            float("inf")
        ):
            continue

        expanded.append(current)

        if current == goal:
            return path, expanded

        for neighbor, info in (
            graph[current].items()
        ):

            new_cost = (
                cost + info["time"]
            )

            if new_cost < best_cost.get(
                neighbor,
                float("inf")
            ):

                best_cost[neighbor] = new_cost

                counter += 1

                heapq.heappush(
                    priority_queue,
                    (
                        new_cost,
                        counter,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, expanded


# ============================================================
# 13. HILL CLIMBING
# ============================================================

def hill_climbing(start, goal):

    heuristic_table = (
        free_flow_heuristic_table(goal)
    )

    current = start

    path = [start]

    expanded = []

    history = [path.copy()]

    while current != goal:

        expanded.append(current)

        candidates = []

        for neighbor in graph[current]:

            if neighbor not in path:

                candidates.append(
                    neighbor
                )

        if not candidates:
            break

        candidates.sort(
            key=lambda node: (
                heuristic_table[node],
                graph[current][node]["time"]
            )
        )

        best_neighbor = candidates[0]

        if (
            heuristic_table[best_neighbor]
            >= heuristic_table[current]
        ):
            break

        current = best_neighbor

        path.append(current)

        history.append(
            path.copy()
        )

    if current == goal:

        expanded.append(current)

        return path, expanded, history

    return None, expanded, history


# ============================================================
# 14. GENETIC ALGORITHM
# ============================================================

def random_valid_route(
    start,
    goal,
    rng
):

    heuristic_table = (
        free_flow_heuristic_table(goal)
    )

    for _ in range(100):

        current = start

        path = [start]

        while current != goal:

            candidates = []

            for neighbor in graph[current]:

                if neighbor not in path:

                    candidates.append(
                        neighbor
                    )

            if not candidates:
                break

            weights = []

            for node in candidates:

                h = heuristic_table[node]

                weights.append(
                    1.0 / (h + 0.1)
                )

            current = rng.choices(
                candidates,
                weights=weights,
                k=1
            )[0]

            path.append(current)

            if len(path) > len(ALL_NODES):
                break

        if (
            current == goal
            and is_valid_path(
                path,
                start,
                goal
            )
        ):
            return path

    return None


def route_fitness(route):

    cost = route_cost(route)

    return 1.0 / (
        1.0 + cost
    )


def crossover(
    parent1,
    parent2,
    start,
    goal,
    rng
):

    common_nodes = []

    for node in parent1[1:-1]:

        if node in parent2[1:-1]:

            common_nodes.append(node)

    if not common_nodes:
        return parent1.copy()

    meeting = rng.choice(
        common_nodes
    )

    i = parent1.index(meeting)

    j = parent2.index(meeting)

    child = (
        parent1[:i]
        + parent2[j:]
    )

    if is_valid_path(
        child,
        start,
        goal
    ):
        return child

    return parent2.copy()


def mutate_route(
    route,
    start,
    goal,
    rng,
    mutation_rate
):

    if rng.random() >= mutation_rate:
        return route.copy()

    new_route = random_valid_route(
        start,
        goal,
        rng
    )

    if new_route is None:
        return route.copy()

    return new_route


def genetic_algorithm(
    start,
    goal,
    population_size=30,
    generations=60,
    mutation_rate=0.20,
    seed=42
):

    rng = random.Random(seed)

    population = []

    attempts = 0

    max_attempts = (
        population_size * 100
    )

    while (
        len(population)
        < population_size
        and attempts < max_attempts
    ):

        attempts += 1

        route = random_valid_route(
            start,
            goal,
            rng
        )

        if route is not None:
            population.append(route)

    if not population:
        return None, [], 0

    best_route = min(
        population,
        key=route_cost
    ).copy()

    progress = []

    evaluations = 0

    for generation in range(
        generations
    ):

        current_best = min(
            population,
            key=route_cost
        )

        evaluations += len(
            population
        )

        if (
            route_cost(current_best)
            < route_cost(best_route)
        ):

            best_route = (
                current_best.copy()
            )

        progress.append(
            (
                generation + 1,
                route_cost(best_route)
            )
        )

        ranked = sorted(
            population,
            key=route_fitness,
            reverse=True
        )

        elite_count = max(
            2,
            population_size // 3
        )

        parents = ranked[
            :elite_count
        ]

        new_population = [
            best_route.copy()
        ]

        while len(new_population) < (
            population_size
        ):

            parent1, parent2 = rng.sample(
                parents,
                2
            )

            child = crossover(
                parent1,
                parent2,
                start,
                goal,
                rng
            )

            child = mutate_route(
                child,
                start,
                goal,
                rng,
                mutation_rate
            )

            if is_valid_path(
                child,
                start,
                goal
            ):

                new_population.append(
                    child
                )

        population = new_population

    evaluations += len(
        population
    )

    return (
        best_route,
        progress,
        evaluations
    )


# ============================================================
# 15. RUN ONE ALGORITHM
# ============================================================

def run_algorithm(
    name,
    start,
    goal,
    dls_limit=8
):

    begin = time.perf_counter()

    history = None

    if name == "BFS":

        path, expanded = bfs(
            start,
            goal
        )

    elif name == "DFS":

        path, expanded = dfs(
            start,
            goal
        )

    elif name == "DLS":

        path, expanded = dls(
            start,
            goal,
            dls_limit
        )

    elif name == "IDS":

        path, expanded = ids(
            start,
            goal,
            max_depth=len(ALL_NODES)
        )

    elif name == "Bidirectional Search":

        path, expanded = (
            bidirectional_search(
                start,
                goal
            )
        )

    elif name == "Greedy Best-First":

        path, expanded = (
            greedy_best_first(
                start,
                goal
            )
        )

    elif name == "A*":

        path, expanded = a_star(
            start,
            goal
        )

    elif name == "Hill Climbing":

        path, expanded, history = (
            hill_climbing(
                start,
                goal
            )
        )

    elif name == "Genetic Algorithm":

        path, progress, evaluations = (
            genetic_algorithm(
                start,
                goal
            )
        )

        expanded = [
            "Fitness evaluation "
            + str(i + 1)
            for i in range(evaluations)
        ]

        history = progress

    else:

        raise ValueError(
            "Unknown algorithm: "
            + str(name)
        )

    runtime_ms = (
        time.perf_counter()
        - begin
    ) * 1000

    valid = is_valid_path(
        path,
        start,
        goal
    )

    if valid:

        distance = route_distance(path)

        cost = route_cost(path)

    else:

        distance = float("inf")

        cost = float("inf")

    return {
        "algorithm": name,
        "path": path,
        "expanded": expanded,
        "expanded_count": len(
            set(expanded)
        ),
        "distance": distance,
        "cost": cost,
        "runtime_ms": runtime_ms,
        "valid": valid,
        "history": history,
        "gap": None
    }


# ============================================================
# 16. ALL ALGORITHMS
# ============================================================

ALGORITHM_NAMES = [
    "BFS",
    "DFS",
    "DLS",
    "IDS",
    "Bidirectional Search",
    "Greedy Best-First",
    "A*",
    "Hill Climbing",
    "Genetic Algorithm"
]


# ============================================================
# 17. COMPARISON
# ============================================================

def compare_all(
    start,
    goal
):

    benchmark_path, benchmark_expanded = (
        dijkstra(
            start,
            goal
        )
    )

    benchmark_cost = route_cost(
        benchmark_path
    )

    results = []

    for name in ALGORITHM_NAMES:

        result = run_algorithm(
            name,
            start,
            goal
        )

        if result["valid"]:

            result["gap"] = (
                (
                    result["cost"]
                    - benchmark_cost
                )
                / benchmark_cost
            ) * 100

        results.append(result)

    return (
        benchmark_path,
        benchmark_expanded,
        benchmark_cost,
        results
    )


# ============================================================
# 18. PRINT SINGLE RESULT
# ============================================================

def print_result(
    result,
    benchmark_cost
):

    print()
    print("=" * 70)
    print(
        result["algorithm"]
    )
    print("=" * 70)

    if not result["valid"]:

        print("Route: NOT FOUND")
        print(
            "Nodes Expanded:",
            result["expanded_count"]
        )
        print(
            "Execution Time:",
            round(
                result["runtime_ms"],
                4
            ),
            "ms"
        )

        return

    print(
        "Route:",
        " -> ".join(
            result["path"]
        )
    )

    print(
        "Travel Time:",
        round(
            result["cost"],
            2
        ),
        "minutes"
    )

    print(
        "Distance:",
        round(
            result["distance"],
            2
        ),
        "km"
    )

    print(
        "Nodes Expanded:",
        result["expanded_count"]
    )

    print(
        "Execution Time:",
        round(
            result["runtime_ms"],
            4
        ),
        "ms"
    )

    print(
        "Optimality Gap:",
        round(
            result["gap"],
            2
        ),
        "%"
    )

    print(
        "Expanded Order:",
        " -> ".join(
            result["expanded"]
        )
    )


# ============================================================
# 19. PRINT COMPARISON TABLE
# ============================================================

def print_comparison_table(
    results,
    benchmark_cost
):

    print()
    print("=" * 105)
    print("ALGORITHM COMPARISON")
    print("=" * 105)

    header = (
        f"{'Algorithm':<24}"
        f"{'Valid':<8}"
        f"{'Time(min)':<13}"
        f"{'Distance(km)':<15}"
        f"{'Nodes':<10}"
        f"{'Runtime(ms)':<14}"
        f"{'Gap(%)':<10}"
    )

    print(header)
    print("-" * 105)

    for result in results:

        if result["valid"]:

            time_value = (
                f"{result['cost']:.2f}"
            )

            distance_value = (
                f"{result['distance']:.2f}"
            )

            gap_value = (
                f"{result['gap']:.2f}"
            )

        else:

            time_value = "N/A"
            distance_value = "N/A"
            gap_value = "N/A"

        print(
            f"{result['algorithm']:<24}"
            f"{'Yes' if result['valid'] else 'No':<8}"
            f"{time_value:<13}"
            f"{distance_value:<15}"
            f"{result['expanded_count']:<10}"
            f"{result['runtime_ms']:<14.4f}"
            f"{gap_value:<10}"
        )

    print("-" * 105)

    print(
        "Dijkstra Benchmark:",
        round(
            benchmark_cost,
            2
        ),
        "minutes"
    )


# ============================================================
# 20. DISPLAY ROAD DATA
# ============================================================

def display_road_data():

    print()
    print("=" * 90)
    print("EMERGENCY ROAD DATA")
    print("=" * 90)

    for (
        u,
        v,
        distance,
        speed,
        congestion
    ) in ROAD_DATA:

        travel_time = (
            distance
            / speed
        ) * 60 * congestion

        print(
            f"{u:<20} <-> "
            f"{v:<20} | "
            f"Distance: {distance:>4.1f} km | "
            f"Speed: {speed:>3} km/h | "
            f"Congestion: {congestion:.2f} | "
            f"Time: {travel_time:>5.2f} min"
        )


# ============================================================
# 21. TRAFFIC SUMMARY
# ============================================================

def traffic_summary():

    congestion_values = [
        c
        for _, _, _, _, c
        in ROAD_DATA
    ]

    speeds = [
        speed
        for _, _, _, speed, _
        in ROAD_DATA
    ]

    print()
    print("=" * 60)
    print("EMERGENCY AMBULANCE ROAD NETWORK")
    print("=" * 60)

    print(
        "Locations:",
        len(ALL_NODES)
    )

    print(
        "Road Links:",
        len(ROAD_DATA)
    )

    print(
        "Average Congestion:",
        round(
            sum(congestion_values)
            / len(congestion_values),
            2
        )
    )

    print(
        "Highest Congestion:",
        round(
            max(congestion_values),
            2
        )
    )

    print(
        "Maximum Speed:",
        max(speeds),
        "km/h"
    )


# ============================================================
# 22. SHOW ROUTE
# ============================================================

def show_route(
    path,
    title="Route"
):

    print()
    print(title)
    print("-" * len(title))

    if path is None:

        print(
            "No valid route found."
        )

        return

    print(
        " -> ".join(path)
    )

    print(
        "Distance:",
        round(
            route_distance(path),
            2
        ),
        "km"
    )

    print(
        "Travel Time:",
        round(
            route_cost(path),
            2
        ),
        "minutes"
    )


# ============================================================
# 23. RUN EXPERIMENT
# ============================================================

def run_experiment(
    start=DEFAULT_START,
    goal=DEFAULT_GOAL
):

    print()
    print("=" * 80)
    print("EMERGENCY AMBULANCE ROUTE OPTIMIZATION")
    print("=" * 80)

    print(
        "Ambulance Source:",
        start
    )

    print(
        "Emergency Location:",
        goal
    )

    benchmark_path, benchmark_expanded, benchmark_cost, results = (
        compare_all(
            start,
            goal
        )
    )

    print()
    print("DIJKSTRA BENCHMARK")

    print(
        " -> ".join(
            benchmark_path
        )
    )

    print(
        "Travel Time:",
        round(
            benchmark_cost,
            2
        ),
        "minutes"
    )

    print_comparison_table(
        results,
        benchmark_cost
    )

    return (
        benchmark_path,
        results
    )


# ============================================================
# 24. CONSOLE MENU
# ============================================================

def menu():

    start = DEFAULT_START
    goal = DEFAULT_GOAL

    while True:

        print()
        print("=" * 65)
        print("EMERGENCY AMBULANCE ROUTE OPTIMIZATION")
        print("=" * 65)

        print(
            "Source:",
            start
        )

        print(
            "Destination:",
            goal
        )

        print()
        print("1. BFS")
        print("2. DFS")
        print("3. Depth Limited Search")
        print("4. Iterative Deepening Search")
        print("5. Bidirectional Search")
        print("6. Greedy Best-First Search")
        print("7. A* Search")
        print("8. Hill Climbing")
        print("9. Genetic Algorithm")
        print("10. Compare All Algorithms")
        print("11. Dijkstra Benchmark")
        print("12. Display Road Data")
        print("13. Traffic Summary")
        print("14. Change Source/Destination")
        print("15. Exit")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":

            result = run_algorithm(
                "BFS",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "2":

            result = run_algorithm(
                "DFS",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "3":

            try:
                limit = int(
                    input(
                        "Enter depth limit: "
                    )
                )
            except ValueError:
                print(
                    "Invalid depth."
                )
                continue

            result = run_algorithm(
                "DLS",
                start,
                goal,
                limit
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "4":

            result = run_algorithm(
                "IDS",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "5":

            result = run_algorithm(
                "Bidirectional Search",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "6":

            result = run_algorithm(
                "Greedy Best-First",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "7":

            result = run_algorithm(
                "A*",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

        elif choice == "8":

            result = run_algorithm(
                "Hill Climbing",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

            if result["history"]:

                print(
                    "\nHill Climbing Progress:"
                )

                for step, path in enumerate(
                    result["history"],
                    start=1
                ):

                    print(
                        step,
                        ":",
                        " -> ".join(path)
                    )

        elif choice == "9":

            result = run_algorithm(
                "Genetic Algorithm",
                start,
                goal
            )

            benchmark_path, _ = dijkstra(
                start,
                goal
            )

            print_result(
                result,
                route_cost(
                    benchmark_path
                )
            )

            if result["history"]:

                print(
                    "\nGenetic Algorithm Progress:"
                )

                for generation, cost in (
                    result["history"]
                ):

                    print(
                        "Generation",
                        generation,
                        "| Best Cost:",
                        round(cost, 2),
                        "minutes"
                    )

        elif choice == "10":

            run_experiment(
                start,
                goal
            )

        elif choice == "11":

            path, expanded = dijkstra(
                start,
                goal
            )

            print()
            print(
                "DIJKSTRA BENCHMARK"
            )

            show_route(
                path,
                "Minimum Travel-Time Route"
            )

            print(
                "Nodes Expanded:",
                len(set(expanded))
            )

        elif choice == "12":

            display_road_data()

        elif choice == "13":

            traffic_summary()

        elif choice == "14":

            print()
            print("Available Locations:")

            for index, node in enumerate(
                ALL_NODES,
                start=1
            ):

                print(
                    index,
                    ".",
                    node
                )

            try:

                source_number = int(
                    input(
                        "Select ambulance source: "
                    )
                )

                goal_number = int(
                    input(
                        "Select emergency location: "
                    )
                )

                if not (
                    1 <= source_number <= len(ALL_NODES)
                    and
                    1 <= goal_number <= len(ALL_NODES)
                ):

                    print(
                        "Invalid location."
                    )

                    continue

                start = ALL_NODES[
                    source_number - 1
                ]

                goal = ALL_NODES[
                    goal_number - 1
                ]

                if start == goal:

                    print(
                        "Source and destination "
                        "must be different."
                    )

                    start = DEFAULT_START
                    goal = DEFAULT_GOAL

            except ValueError:

                print(
                    "Please enter valid numbers."
                )

        elif choice == "15":

            print(
                "\nEmergency Ambulance "
                "Route Optimization closed."
            )

            break

        else:

            print(
                "Invalid choice. "
                "Please select 1-15."
            )


# ============================================================
# 25. PROGRAM START
# ============================================================

if __name__ == "__main__":

    menu()
