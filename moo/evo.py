import random
from collections import namedtuple

import numpy as np

coords = np.linspace(0, 1, 50)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
GRID_HYPERVOLUME = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# NORM_RETURN = [-2.237, 118.93925]
# NORM_RISK = [0.00018, 2.03338]
# NORM_RETURN_FULL = [-92.88947, 118.93925]
# NORM_RISK_FULL = [0.00018, 3.16057]
NUM_ASSETS = 20


MarkowitzProblem = namedtuple(
    "MarkowitzProblem",
    [
        "MU",
        "SIGMA",
        "NORM_RETURN_MIN",
        "NORM_RETURN_MAX",
        "NORM_RISK_MIN",
        "NORM_RISK_MAX",
    ],
)


def random_portfolio():
    points = np.random.rand(19)
    points = np.concatenate(([0.0], points, [1.0]))
    points.sort()

    weights = np.diff(points).tolist()
    return weights


def portfolio_return(weights, problem: MarkowitzProblem):
    ret = sum(w * m for w, m in zip(weights, problem.MU))
    ret = (ret - problem.NORM_RETURN_MIN) / (
        problem.NORM_RETURN_MAX - problem.NORM_RETURN_MIN
    )
    return ret


def portfolio_variance(weights, problem: MarkowitzProblem):
    var = 0.0
    for i in range(NUM_ASSETS):
        for j in range(NUM_ASSETS):
            var += weights[i] * weights[j] * problem.SIGMA[i][j]
    var = (var - problem.NORM_RISK_MIN) / (
        problem.NORM_RISK_MAX - problem.NORM_RISK_MIN
    )
    return var


def evaluate2(weights, problem: MarkowitzProblem):
    var = portfolio_variance(weights, problem)
    ret = portfolio_return(weights, problem)
    return (var, 1 - ret)


def count_non_zero(weights, eps=0.01):
    """Count how many weights are non-trivially > 0.
    We use a small epsilon to allow for floating-point noise"""
    ret = sum(1 for w in weights if w > eps)
    ret = (ret - 1) / (20 - 1)
    return ret


def evaluate3(weights, problem: MarkowitzProblem):
    var = portfolio_variance(weights, problem)
    ret = portfolio_return(weights, problem)
    non_zero = count_non_zero(weights)
    return (var, 1 - ret, 1 - non_zero)


def dominates(obj1, obj2):
    """
    Check if obj1 = (o1_1, o1_2, ...) dominates obj2 = (o2_1, o2_2, ...).
    We say obj1 dominates obj2 if:
      - obj1 is <= obj2 in all objectives (since we consider all minimization)
      - obj1 is <  obj2 in at least one objective
    """
    for o1, o2 in zip(obj1, obj2):
        if o1 > o2:
            return False

    for o1, o2 in zip(obj1, obj2):
        if o1 < o2:
            return True

    return False


def hypervolume(front):
    ratio = 0
    for point in GRID_HYPERVOLUME:
        for sol in front:
            if dominates(sol[1], point):
                ratio += 1
                break
    return ratio / len(GRID_HYPERVOLUME)


def fast_non_dominated_sort(pop):
    """
    Perform the fast non-dominated sort on the population.
    pop is a list of tuples: (weights, (obj1, obj2), ...)
    Return a list of fronts, where each front is a list of indices of pop.
    """
    size = len(pop)
    S = [[] for _ in range(size)]  # S[i] will hold the solutions that i dominates
    n = [0] * size  # n[i] will hold the number of solutions that dominate i
    rank = [0] * size

    for p in range(size):
        p_objs = pop[p][1]
        for q in range(size):
            if p == q:
                continue
            q_objs = pop[q][1]
            if dominates(p_objs, q_objs):
                S[p].append(q)
            elif dominates(q_objs, p_objs):
                n[p] += 1

    # Identify the first front
    fronts = [[]]
    for i in range(size):
        if n[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    # Build subsequent fronts
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            # for each q dominated by p
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance_assignment(pop, front):
    """
    Assign crowding distance for each solution in a given front.
    pop is a list of (weights, objective_tuple)
    front is a list of indices into pop.
    We'll return a dictionary: index -> crowding distance.
    """
    distance = {i: 0.0 for i in front}
    if not front:
        return distance

    num_objectives = len(pop[front[0]][1])

    for m in range(num_objectives):
        front_sorted = sorted(front, key=lambda i: pop[i][1][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")

        f_min = pop[front_sorted[0]][1][m]
        f_max = pop[front_sorted[-1]][1][m]
        if f_max == f_min:
            continue

        for idx in range(1, len(front_sorted) - 1):
            prev_i = front_sorted[idx - 1]
            next_i = front_sorted[idx + 1]
            dist = (pop[next_i][1][m] - pop[prev_i][1][m]) / (f_max - f_min)
            distance[front_sorted[idx]] += dist

    return distance


def localsearch(W, problem: MarkowitzProblem, evaluate, max_iters=1000):
    w_score = evaluate(W, problem)
    indices1 = np.array(range(NUM_ASSETS))
    indices2 = np.array(range(NUM_ASSETS))

    for iteration in range(max_iters):
        improved = False

        np.random.shuffle(indices1)
        for i in indices1:
            np.random.shuffle(indices2)
            for j in indices2:
                if i == j:
                    continue

                neighbor = W.copy()
                neighbor[j] = neighbor[j] + neighbor[i]
                neighbor[i] = 0
                n_score = evaluate(neighbor, problem)
                assert_valid(neighbor)

                if dominates(n_score, w_score):
                    W = neighbor
                    w_score = n_score
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break
    assert_valid(W)
    return W


def make_new_population(pop, pop_size):
    """
    Given the combined population (parents + offspring),
    perform the NSGA-II selection of the next generation.
    pop is a list of (weights, objective_tuple).
    We return a new list of pop_size individuals.
    """
    fronts = fast_non_dominated_sort(pop)

    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            distances = crowding_distance_assignment(pop, front)
            sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
            needed = pop_size - len(new_pop)
            new_pop.extend(sorted_front[:needed])
            break

    return [pop[i] for i in new_pop]


def tournament_selection(pop, k=2):
    """
    Perform a tournament selection of size k and return the winner.
    We pick k individuals randomly from pop.
    The winner is the one with:
      1) Lower rank, or
      2) Higher crowding distance if the same rank
    But to do that, we need the rank and crowding info, so let's do:
      - We first do a full non-dominated sort and store rank.
      - Then we do a crowding distance assignment for each front.
    """
    fronts = fast_non_dominated_sort(pop)
    rank_of = {}
    for i, front in enumerate(fronts):
        for idx in front:
            rank_of[idx] = i

    distance_of = {}
    for front in fronts:
        dist = crowding_distance_assignment(pop, front)
        distance_of.update(dist)

    contenders = random.sample(range(len(pop)), k)
    best = contenders[0]
    for c in contenders[1:]:
        if rank_of[c] < rank_of[best]:
            best = c
        elif rank_of[c] == rank_of[best]:
            if distance_of[c] > distance_of[best]:
                best = c
    return pop[best]


def norm(weights):
    s = sum(weights)
    return [w / s for w in weights]


def invalid_crossover(w1, w2, crossover_rate=0.9):
    """
    Single-point crossover for simplicity.
    With 'crossover_rate' probability, do crossover,
    otherwise return copies of the original.
    """
    if random.random() < crossover_rate:
        point = random.randint(1, NUM_ASSETS - 1)
        child1 = w1[:point] + w2[point:]
        child2 = w2[:point] + w1[point:]
        return norm(child1), norm(child2)
    else:
        return norm(w1[:]), norm(w2[:])


def convex_combination_crossover(w1, w2):
    """
    Crossover using convex combination:
    w1, w2 are two parents
    with probability crossover_rate, return a random convex combination
    of w1 and w2.
    """
    alpha = np.random.uniform(0, 1)
    return alpha * np.array(w1) + (1 - alpha) * np.array(w2), (1 - alpha) * np.array(
        w1
    ) + alpha * np.array(w2)


def invalid_mutate(weights, mutation_rate=0.1):
    """
    Mutate each weight with probability = mutation_rate,
    then re-normalize so sum of weights = 1.
    """
    for i in range(NUM_ASSETS):
        if random.random() < mutation_rate:
            weights[i] *= random.uniform(0.3, 2.5)
            if weights[i] < 0:
                weights[i] = 0
    return norm(weights)


def fix(weights):
    """Fix numerical errors in weights."""
    weights = [max(0, w) for w in weights]
    s = sum(weights)
    if s > 1:
        weights = [w / s for w in weights]
    elif s < 1:
        weights = [w + (1 - s) / len(weights) for w in weights]
    return weights


def creep_mutation_simplex(w, max_step=0.1):
    """
    Creep mutation on a simplex:
    - choose indices i, j
    - shift a small amount of mass eps from w[i] to w[j]
    - ensure w[i] stays nonnegative
    """
    n = len(w)
    i = random.randrange(n)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)

    # How much can we safely steal from w[i]?
    # We choose eps up to min(max_step, w[i]) so we don't go negative.
    eps = random.uniform(0, min(max_step, w[i]))

    w[i] -= eps
    w[j] += eps

    return w


def assert_valid(weights):
    """
    Check if the weights are valid (non-negative and sum to 1).
    """
    assert len(weights) == NUM_ASSETS, f"Invalid length: {len(weights)} != {NUM_ASSETS}"
    assert all(w >= 0 for w in weights) and all(w <= 1 for w in weights), (
        f"Invalid weights: {weights}"
    )
    assert np.isclose(np.sum(weights), 1), f"Invalid sum: {sum(weights)} != 1"


def nsga2_markowitz(
    problem: MarkowitzProblem,
    pop_size: int,
    num_generations: int,
    evaluate=evaluate2,
    mutate=creep_mutation_simplex,
    crossover=convex_combination_crossover,
):
    hypervolume_list = []

    population = [random_portfolio() for _ in range(pop_size)]
    population.extend(np.identity(NUM_ASSETS).tolist())  # type: ignore
    pop_evaluated = [(ind, evaluate(ind, problem)) for ind in population]
    for gen in range(num_generations):
        hypervolume_list.append(hypervolume(pop_evaluated))
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop_evaluated)
            p2 = tournament_selection(pop_evaluated)

            c1_weights, c2_weights = crossover(p1[0], p2[0])

            c1_weights = mutate(c1_weights)
            c2_weights = mutate(c2_weights)

            c1_weights = localsearch(c1_weights, problem, evaluate)
            c2_weights = localsearch(c2_weights, problem, evaluate)

            c1_weights = fix(c1_weights)
            c2_weights = fix(c2_weights)

            assert_valid(c1_weights)
            assert_valid(c2_weights)

            c1_fit = evaluate(c1_weights, problem)
            c2_fit = evaluate(c2_weights, problem)

            offspring.append((c1_weights, c1_fit))
            offspring.append((c2_weights, c2_fit))

        combined = pop_evaluated + offspring
        pop_evaluated = make_new_population(combined, pop_size)

    return pop_evaluated, hypervolume_list
