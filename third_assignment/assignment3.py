import random
import math
import itertools
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

cities = [
    (4, 3),
    (9, 4),
    (1, 2),
    (5, 9),
    (6, 7),
    (3, 8)
]

POP_SIZE = 120
GENERATIONS = 400
ELITE_SIZE = 2
MUTATION_RATE = 0.15
TOURNAMENT_K = 5

def tour_distance(tour, cities=cities):
    dist = 0.0
    for i in range(len(tour)):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % len(tour)]]
        dist += math.hypot(a[0] - b[0], a[1] - b[1])
    return dist


def brute_force_optimal(cities):
    N = len(cities)
    best_perm = None
    best_dist = float('inf')
    for perm in itertools.permutations(range(N)):
        d = tour_distance(list(perm), cities)
        if d < best_dist:
            best_dist = d
            best_perm = perm
    return list(best_perm), best_dist


def init_population(pop_size, N):
    base = list(range(N))
    pop = []
    for _ in range(pop_size):
        p = base[:]
        random.shuffle(p)
        pop.append(p)
    return pop


def tournament_selection(pop, k=TOURNAMENT_K):
    selected = random.sample(pop, k)
    selected.sort(key=lambda t: tour_distance(t))
    return selected[0][:]


def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b + 1] = parent1[a:b + 1]

    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child


def swap_mutation(tour, mut_rate=MUTATION_RATE):
    tour = tour[:]
    for i in range(len(tour)):
        if random.random() < mut_rate:
            j = random.randrange(len(tour))
            tour[i], tour[j] = tour[j], tour[i]
    return tour


def evolve(pop, elite_size=ELITE_SIZE, mut_rate=MUTATION_RATE):
    newpop = []
    pop_sorted = sorted(pop, key=lambda t: tour_distance(t))
    # Elitizm
    newpop.extend(pop_sorted[:elite_size])
    # geri kalanlar crossover + mutasyon
    while len(newpop) < len(pop):
        p1 = tournament_selection(pop)
        p2 = tournament_selection(pop)
        child = ordered_crossover(p1, p2)
        child = swap_mutation(child, mut_rate)
        newpop.append(child)
    return newpop

def run_tsp_ga(cities, pop_size=POP_SIZE, generations=GENERATIONS,
               elite_size=ELITE_SIZE, mut_rate=MUTATION_RATE):
    N = len(cities)

    brute_tour, brute_dist = brute_force_optimal(cities)
    print(f"Brute-force optimal distance: {brute_dist:.4f}")
    print(f"Brute-force optimal tour: {brute_tour}")


    population = init_population(pop_size, N)
    best_history = []
    for gen in range(generations):
        population = evolve(population, elite_size, mut_rate)
        best = min(population, key=lambda t: tour_distance(t))
        best_history.append(tour_distance(best))
        if gen % 50 == 0 or gen == generations - 1:
            print(f"Gen {gen+1:3d}: Best distance = {tour_distance(best):.4f}")

    ga_best = min(population, key=lambda t: tour_distance(t))
    ga_best_dist = tour_distance(ga_best)
    print(f"\nGA best distance: {ga_best_dist:.4f}")
    print(f"GA best tour: {ga_best}")


    plt.figure(figsize=(8, 4))
    plt.plot(best_history)
    plt.title("GA Convergence (best distance per generation)")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tsp_ga_convergence.png')
    print("Convergence grafiği -> tsp_ga_convergence.png")

    def plot_tour(tour, title, subplot_index=None):
        xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
        ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
        plt.plot(xs, ys, marker='o')
        for i, (x, y) in enumerate(cities):
            plt.text(x + 0.08, y + 0.08, str(i))
        plt.title(title)
        plt.axis('equal')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_tour(ga_best, f"GA best (dist={ga_best_dist:.3f})")
    plt.subplot(1, 2, 2)
    plot_tour(brute_tour, f"Brute-force optimal (dist={brute_dist:.3f})")
    plt.tight_layout()
    plt.savefig('tsp_ga_tours.png')
    print("Tur görselleştirmesi -> tsp_ga_tours.png")

    return {
        'brute_tour': brute_tour,
        'brute_dist': brute_dist,
        'ga_best': ga_best,
        'ga_best_dist': ga_best_dist,
        'history': best_history
    }

if __name__ == '__main__':
    results = run_tsp_ga(cities)

