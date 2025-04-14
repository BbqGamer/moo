import pathlib
import pickle
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from pygmo import hypervolume
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression

coords = np.linspace(0, 1, 50)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

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
    return np.array([var, 1 - ret])


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
    return np.array([var, 1 - ret, 1 - non_zero])


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

                if dominates(n_score, w_score):
                    W = neighbor
                    w_score = n_score
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break
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
    return alpha * np.array(w1) + (1 - alpha) * np.array(w2)


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
    initial_population: list,
    pop_size: int,
    num_generations: int,
    evaluate=evaluate2,
    mutate=creep_mutation_simplex,
    crossover=convex_combination_crossover,
):
    hypervolume_list = []

    pop_evaluated = [(ind, evaluate(ind, problem)) for ind in initial_population]
    best_hv = -float("inf")

    for gen in range(num_generations):
        hv = hypervolume([np.array(p[1]) for p in pop_evaluated])
        computed = hv.compute([1] * len(pop_evaluated[0][1]))
        if computed > best_hv:
            best_hv = computed
            best_pop = pop_evaluated[:]

        hypervolume_list.append(computed)
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop_evaluated)
            p2 = tournament_selection(pop_evaluated)

            c1_weights = crossover(p1[0], p2[0])
            c2_weights = crossover(p2[0], p1[0])

            c1_weights = mutate(c1_weights)
            c2_weights = mutate(c2_weights)

            c1_fit = evaluate(c1_weights, problem)
            c2_fit = evaluate(c2_weights, problem)

            offspring.append((c1_weights, c1_fit))
            offspring.append((c2_weights, c2_fit))

        combined = pop_evaluated + offspring
        pop_evaluated = make_new_population(combined, pop_size)

    return best_pop, hypervolume_list


def sbx_crossover_simplex(parent1, parent2, eta=15):
    """
    Perform SBX crossover on two parent vectors lying on the unit simplex.

    Parameters:
    - parent1: np.ndarray, first parent vector (elements in [0, 1], sum to 1).
    - parent2: np.ndarray, second parent vector (elements in [0, 1], sum to 1).
    - eta: float, distribution index for crossover.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Two offspring vectors on the unit simplex.
    """
    parent1 = np.asarray(parent1, dtype=np.float64)
    parent2 = np.asarray(parent2, dtype=np.float64)

    if parent1.shape != parent2.shape:
        raise ValueError("Parent vectors must have the same shape.")

    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)

    for i in range(parent1.shape[0]):
        x1 = parent1[i]
        x2 = parent2[i]

        if np.random.rand() <= 0.5:
            if abs(x1 - x2) > 1e-14:
                x_min = min(x1, x2)
                x_max = max(x1, x2)

                rand = np.random.rand()
                beta = 1.0 + (2.0 * (x_min) / (x_max - x_min))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                c1 = np.clip(c1, 0.0, 1.0)
                c2 = np.clip(c2, 0.0, 1.0)

                child1[i] = c1
                child2[i] = c2
            else:
                child1[i] = x1
                child2[i] = x2
        else:
            child1[i] = x1
            child2[i] = x2

    # Normalize to ensure the sum is 1
    child1_sum = child1.sum()
    child2_sum = child2.sum()

    if child1_sum > 0:
        child1 /= child1_sum
    else:
        # If sum is zero (due to numerical issues), assign uniform distribution
        child1 = np.full_like(child1, 1.0 / child1.size)

    if child2_sum > 0:
        child2 /= child2_sum
    else:
        child2 = np.full_like(child2, 1.0 / child2.size)

    return child1


def blx_alpha_crossover_simplex(parent1, parent2, alpha=0.5):
    """
    Perform BLX-α crossover on two parent vectors constrained to the unit simplex.

    Parameters:
    - parent1: np.ndarray, first parent vector (elements in [0, 1], sum to 1).
    - parent2: np.ndarray, second parent vector (elements in [0, 1], sum to 1).
    - alpha: float, the α parameter controlling the exploration range.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Two offspring vectors on the unit simplex.
    """
    parent1 = np.asarray(parent1, dtype=np.float64)
    parent2 = np.asarray(parent2, dtype=np.float64)

    if parent1.shape != parent2.shape:
        raise ValueError("Parent vectors must have the same shape.")

    if not np.allclose(parent1.sum(), 1.0) or not np.allclose(parent2.sum(), 1.0):
        raise ValueError("Parent vectors must sum to 1.")

    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    d = c_max - c_min

    lower_bound = np.maximum(c_min - alpha * d, 0.0)
    upper_bound = np.minimum(c_max + alpha * d, 1.0)

    # Generate offspring within the extended intervals
    child1 = np.random.uniform(lower_bound, upper_bound)
    child2 = np.random.uniform(lower_bound, upper_bound)

    # Normalize offspring to ensure they lie on the unit simplex
    child1_sum = child1.sum()
    child2_sum = child2.sum()

    if child1_sum > 0:
        child1 /= child1_sum
    else:
        # Assign uniform distribution if sum is zero
        child1 = np.full_like(child1, 1.0 / child1.size)

    if child2_sum > 0:
        child2 /= child2_sum
    else:
        child2 = np.full_like(child2, 1.0 / child2.size)

    return child1


def check_feasibility(child):
    ERROR_TOLERANCE = 1e-6
    if not np.isclose(np.sum(child), np.ones(1), rtol=ERROR_TOLERANCE):
        print("SUM ERR")
        print(np.sum(np.array(child)))
        print(child)
    assert np.isclose(np.sum(child), np.ones(1))
    if np.max(child) > 1:
        if (np.max(child) - 1) <= ERROR_TOLERANCE:
            child = np.clip(child, np.min(child), 1)
        else:
            print("MAX ERR")
            print(np.max(child))
            print(child)

    assert np.max(child) <= 1
    if np.min(child) < 0:
        if -1 * np.min(child) <= ERROR_TOLERANCE:
            child = np.clip(child, 0, np.max(child))
        else:
            print("MIN ERR")
            print(np.min(child))
            print(child)
    assert np.min(child) >= 0


def mutate_linprog(weights, mutation_rate=0.3):
    """
    Mutate each weight with probability = mutation_rate,
    then re-normalize so sum of weights = 1.
    """
    if random.random() > mutation_rate:
        return weights
    p1 = np.array(weights)
    p2 = np.array(random_portfolio())
    line_dir = p1 - p2

    # Step 3: Define the objective direction
    # We want to move as far as possible along the same direction
    v_dir = line_dir / np.linalg.norm(line_dir)

    # Step 4: Equality constraint: sum(x(t)) = 1
    A_eq = np.array([[np.sum(line_dir)]])
    b_eq = np.array([0])  # Ensures movement stays in the simplex

    # Step 5: Inequality constraints: 0 <= x_i(t) <= 1
    A_ub = []
    b_ub = []

    for i in range(20):
        # Upper bound: x_i(t) = p1[i] + t * line_dir[i] <= 1
        A_ub.append([line_dir[i]])
        b_ub.append(1 - p1[i])
        # Lower bound: x_i(t) = p1[i] + t * line_dir[i] >= 0
        # Equivalent to: -x_i(t) <= -0 => -p1[i] - t * line_dir[i] <= 0
        A_ub.append([-line_dir[i]])
        b_ub.append(p1[i])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Step 6: Objective function: maximize t * (v_dir @ line_dir)
    # linprog minimizes, so we negate it
    c = -(v_dir @ line_dir)

    # Step 7: Solve the linear program
    res = linprog(c=[c], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")

    # Step 8: Compute the final point if successful
    if res.success:
        t_opt = res.x[0]
        p3 = p1 + t_opt * line_dir
        check_feasibility(p3)
        return p3.tolist()
    return weights


def basic_random_pop(pop_size):
    return [random_portfolio() for _ in range(pop_size)]


def basic_identity_pop(pop_size):
    pop: list = np.identity(NUM_ASSETS).tolist()  # type: ignore
    while len(pop) < pop_size:
        pop.append(random_portfolio())
    return pop


def random_pairs_pop(pop_size):
    pairs = [(i, j) for i in range(NUM_ASSETS) for j in range(i + 1, NUM_ASSETS)]
    arr = np.zeros((len(pairs), NUM_ASSETS))
    for idx, (i, j) in enumerate(pairs):
        alpha = np.random.uniform(0.1, 0.9)
        arr[idx, i] = alpha
        arr[idx, j] = 1 - alpha
    np.random.shuffle(arr)
    return arr[:pop_size].tolist()


def read_asset_data(file_path):
    """Function to read asset data from a text file"""
    with open(file_path, "r") as file:
        lines = file.readlines()
        asset_name = lines[0].strip()
        num_points = int(lines[1].strip())
        data = [float(line.split()[1]) for line in lines[2 : num_points + 2]]
    return asset_name, data


NUM_GENERATIONS = 100

if __name__ == "__main__":
    asset_data = {}
    for file in sorted(pathlib.Path("data").glob("*Part1.txt")):
        asset_name, data = read_asset_data(file)
        asset_data[asset_name] = data
    asset_data = pd.DataFrame(asset_data)
    assert len(asset_data.columns) == NUM_ASSETS

    predicted_returns = np.zeros(NUM_ASSETS)
    for i, asset in enumerate(asset_data.columns):
        data = asset_data[asset].values
        model = LinearRegression()
        model.fit(np.arange(0, 101).reshape(-1, 1), data)
        # stock_mean, stock_std = np.mean(data), np.std(data)
        predictions = model.predict(np.arange(0, 201).reshape(-1, 1))
        cur = data[100]
        pred = predictions[200]
        ret = (pred - cur) / cur * 100
        predicted_returns[i] = ret

    NORM_RETURN_FULL = [-95, 120]
    NORM_RISK_FULL = [0.00015, 3.5057]

    problem = MarkowitzProblem(
        predicted_returns, asset_data.cov().values, *NORM_RETURN_FULL, *NORM_RISK_FULL
    )

    results = []
    for init_pop_func in [random_pairs_pop, basic_random_pop, basic_identity_pop]:
        for crossover_func in [
            sbx_crossover_simplex,
            blx_alpha_crossover_simplex,
            convex_combination_crossover,
        ]:
            for mutation_func in [mutate_linprog, creep_mutation_simplex]:
                for popsize in [20, 40, 60]:
                    intial_population = init_pop_func(popsize)
                    final_pop, hypervolumes = nsga2_markowitz(
                        problem,
                        intial_population,
                        popsize,
                        10,
                        evaluate=evaluate3,
                        mutate=mutation_func,
                        crossover=crossover_func,
                    )
                    print(
                        f"Finished {init_pop_func.__name__}, {crossover_func.__name__}, {mutation_func.__name__}, popsize={popsize} with hypervolume {hypervolumes[-1]}"
                    )
                    print("=========================================")
                    results.append(
                        (
                            init_pop_func.__name__,
                            crossover_func.__name__,
                            mutation_func.__name__,
                            popsize,
                            hypervolumes,
                            final_pop,
                        )
                    )

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
