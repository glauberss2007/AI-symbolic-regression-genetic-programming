import pandas as pd
import time
import gp
from random import sample, random, randint, choice, uniform
from copy import deepcopy
from datetime import datetime

now = datetime.now()
formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')

#   *PARAMETROS*
BASE_DE_TREINO = "./datasets/synth1/synth1-train.csv"
BASE_DE_TESTE = "./datasets/synth1/synth1-test.csv"
DATASET = "synth1"
TAM_POPULACAO = 50
N_GERACOES = 5
PROFUNDIDADE_MAX = 4
TX_CRUZAMENTO = 0.9
TX_MUTACAO = 0.05
TIPO_SELECAO = "roleta"
TAM_TORNEIO = 2
TX_ELITISMO = 1
ARQUIVO_DE_SAIDA = formatted_date+"_resultado_"+TIPO_SELECAO+"_"+"TAM_POPULACAO"
NUMERO_REPETICOES = 1


def random_pick(choices, probabilities):
    x = uniform(0, 1)
    cumulative_probability = 0.0

    for item, item_probability in zip(choices, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            choice = item
            break

    return choice


def nodes(prim_dict, variables):
    functions = []
    terminals = []

    for key in prim_dict:
        if prim_dict[key] == 0:
            terminals.append(key)
        else:
            functions.append(key)

    primitives = functions + terminals
    return {"primitivos": primitives, "funcoes": functions,
            "terminais": terminals, "variaveis": variables}


def initializes(popsize, op_set, s, max_depth):
    population = []

    x = int(popsize / (2 * max_depth))
    for depth in range(1, max_depth + 1):
        for i in range(1, x + 1):
            _full = gp.Tree(op_set, s, "full", depth)
            _grow = gp.Tree(op_set, s, "grow", depth)
            population.append(_full)
            population.append(_grow)

    y = len(population)
    if y < popsize:
        for i in range(1, popsize - y + 1):
            population.append(gp.Tree(op_set, s, choice(["full", "grow"]),
                                      randint(1, max_depth)))

    return population


def evolve(pop, train, cross_rate, mut_rate, sel_type, tourn_size, max_depth, generations,
           elitist, repetition):
    target_fitness = 0.0
    best_solution = None
    fitness_solution = None

    print("Treinando em "+DATASET+" de treino.")

    result = []

    best, best_score, worst_score, mean_score, repeated = gp.evaluation(pop, train)
    best_solution = best
    fitness_solution = best_score

    k = 0
    while (True):

        if k >= generations:
            break

        rep_rate = 1.0 - (cross_rate + mut_rate)
        operations = ["cross", "mut", "rep"]
        probabilities = [cross_rate, mut_rate, rep_rate]

        better_solutions = 0
        worse_solutions = 0

        next_gen = []
        for i in range(len(pop)):
            op = random_pick(operations, probabilities)
            if op == "cross":
                child, better = gp.subtree_crossover(pop,sel_type, tourn_size, train,
                                                     elitist)
            elif op == "mut":
                child = gp.subtree_mutation(pop,sel_type, tourn_size, train,
                                            max_depth, elitist)
            elif op == "rep":
                child = gp.reproduction(pop,sel_type, tourn_size, train)
            next_gen.append(child)

            if elitist == 1:
                if op == "cross" and better == 1:
                    better_solutions += 1
                elif op == "cross" and better == 0:
                    worse_solutions += 1

        pop = next_gen

        print('Geracao ', k)
        best, best_score, worst_score, mean_score, repeated = gp.evaluation(pop, train)

        if (best_score < fitness_solution):
            best_solution = best
            fitness_solution = best_score

        winner = deepcopy(best)
        winner = winner.build_program(winner.root)
        print("Melhor funcao:", winner)
        print("Melhor fitness:", best_score)
        print("Pior fitness:", worst_score)
        print("Fitness medio:", mean_score)
        print("Qtd. individuos semelhantes:", repeated)

        if elitist == 1:
            print("Qtd. filhos melhores:", better_solutions)
            print("Qtd. filhos piores:", worse_solutions)

        result.append((repetition, k + 1, best_score, worst_score, mean_score,
                       repeated, better_solutions, worse_solutions, winner))

        print("************************************")

        pop = next_gen
        k += 1

    return best_solution, result


def read_data(filename):
    data = []
    file = open(filename, "r")
    for line in file:
        line_string = line.rstrip('\n')
        line_list = line_string.split(',')
        for i in range(len(line_list)):
            line_list[i] = float(line_list[i])
        line_tuple = tuple(line_list)
        data.append(line_tuple)

    file.close()

    return data


def main():
    train = read_data(BASE_DE_TREINO)
    test = read_data(BASE_DE_TESTE)

    op_set = {"+": 1, "-": 1, "*": 1, "/": 1, "rand": 0}

    if DATASET == "synth1" or DATASET == "synth2":
        v = ['x', 'y']
    else:
        v = ['x', 'y', 'z', 'a', 'b', 'c', 'p', 'q']

    for item in v:
        op_set[item] = 0

    s = nodes(op_set, v)

    train_results = []
    test_results = []
    for i in range(1, NUMERO_REPETICOES + 1):
        print("Execucao - %d" % i)
        pop = initializes(TAM_POPULACAO, op_set, s, PROFUNDIDADE_MAX)
        best_solution, result = evolve(pop, train, TX_CRUZAMENTO, TX_MUTACAO,TIPO_SELECAO,
                                       TAM_TORNEIO, PROFUNDIDADE_MAX, N_GERACOES, TX_ELITISMO,
                                       i)
        train_results = train_results + result
        winner = deepcopy(best_solution)
        winner = winner.build_program(winner.root)
        fitness = gp.fitness(best_solution, test)
        print("Teste em "+DATASET+" de teste!")
        print("Funcao:", winner)
        print("Fitness:", fitness)
        test_results.append((i, fitness, winner))
        print()

    col_names = ("experimento", "geracao", "melhor_resultado", "pior_resultado",
                 "media", "ind_repetido", "melhor_ind", "pior_ind", "melhor_funcao")
    frame = pd.DataFrame.from_records(train_results, columns=col_names)
    frame.to_csv(ARQUIVO_DE_SAIDA + "_treino.csv", index=False, sep=',', encoding='utf-8')

    col_names = ("experimento", "fitness", "melhor_funcao")
    frame = pd.DataFrame.from_records(test_results, columns=col_names)
    frame.to_csv(ARQUIVO_DE_SAIDA + "_teste.csv", index=False, sep=',', encoding='utf-8')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s minutos ---" % ((time.time() - start_time) / 60))
