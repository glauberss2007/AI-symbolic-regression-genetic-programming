from random import sample,randint, choice
from math import sqrt
from copy import deepcopy
import numpy as np
import random

class Node(object):

    def __init__(self, value):
        if value == "rand":
            self.value = str(choice([0.1, 0.2, 0.5, 1, 2, 3, 4, 5]))
        else:
            self.value = value

        self.left = None
        self.right = None
        self.parent = None


class Tree(object):

    def __init__(self, primitives, set_dict, method, depth):
        self.root = None
        self.primitives = primitives
        self.set_dict = set_dict
        self.depth = depth
        self.size = 2 ** (self.depth + 1) - 1
        self.last_level = 2 ** self.depth - 1
        if method == 'full':
            self.root = self._full(self.root, self.size, self.last_level, 0)
        elif method == 'grow':
            self.root = self._grow(self.root, self.size, self.last_level, 0)


    def _full(self, node, s, m, n, parent = None):

        if(m == 0):
            node = Node(choice(self.set_dict["terminais"]))
            node.parent = parent
        elif(n < m):
            node = Node(choice(self.set_dict["funcoes"]))
            node.parent = parent
            node.left = self._full(node.left, s, m, 2*n + 1, node)
            node.right = self._full(node.right, s, m, 2*n + 2, node)
        elif(n < s):
            node = Node(choice(self.set_dict["terminais"]))
            node.parent = parent

        return node


    def _grow(self, node, s, m, n, parent = None):

        if n == 0:
            if self.depth >= 1:
                prim = choice(self.set_dict["primitivos"])
                node = Node(prim)
                node.parent = parent
                node.left = self._grow(node.left, s, m, 2*n + 1, node)
                node.right = self._grow(node.right, s, m, 2*n + 2, node)
            elif self.depth == 0:
                prim = choice(self.set_dict["terminais"])
                node = Node(prim)
        elif (n < m):
            if parent.value not in self.set_dict["funcoes"]:
                node = None
            else:
                prim = choice(self.set_dict["primitivos"])
                node = Node(prim)
                node.parent = parent
                node.left = self._grow(node.left, s, m, 2*n + 1, node)
                node.right = self._grow(node.right, s, m, 2*n + 2, node)
        elif (n < s):
            if parent.value not in self.set_dict["funcoes"]:
                node = None
            else:
                node = Node(choice(self.set_dict["terminais"]))
                node.parent = parent

        return node


    def build_program(self, node):
        eq = ""
        if node != None:
            eq = node.value
            left = self.build_program(node.left)
            right = self.build_program(node.right)
            eq = "(" + left + eq + right + ")"

        return eq


    def nodes(self, node, i, n):

        if i > n:
            return []
        elif node == None:
            return []

        return [node] + self.nodes(node.left, 2*i + 1, n) + self.nodes(node.right, 2*i + 2, n)


    def random_node(self, n = 2):

        all_nodes = self.nodes(self.root, 0, 2*n + 2)
        i = randint(0, len(all_nodes) - 1)

        return all_nodes[i]

def subtree_crossover(population, sel_type,eps, k, data, elitist):
    if (sel_type == "torneio"):
        first, first_score = tournament_selection(population, k, data)
        second, second_score = tournament_selection(population, k, data)
    elif (sel_type == "roleta"):
        first, first_score = roulette_selection(population, data)
        second, second_score = roulette_selection(population, data)
    elif (sel_type == "lexicase"):
        # Cria uma lista de funções de caso
        first, first_score = epsilon_lexicase_selection(population,k, data, eps)
        second, second_score = epsilon_lexicase_selection(population,k, data, eps)

    first_parent = deepcopy(first)
    second_parent = deepcopy(second)

    cross_pt1 = first_parent.random_node()
    cross_pt2 = second_parent.random_node()

    new_individual = _crossover(first_parent, cross_pt1, cross_pt2)
    new_score = fitness(new_individual, data)
    mean_score = (first_score + second_score)/2.0

    x = 0
    if new_score < (first_score + second_score)/2.0:
        x = 1

    if (elitist == 0) or (new_score < first_score and new_score < second_score):
        return new_individual, x
    elif first_score < second_score:
        return first, x
    else:
        return second, x


def subtree_mutation(population,sel_type,eps, k, data, max_depth, elitist):
    if (sel_type == "torneio"):
        individual, score = tournament_selection(population, k, data)
    elif (sel_type == "roleta"):
        individual, score = roulette_selection(population, data)
    elif (sel_type == "lexicase"):
        individual, score = epsilon_lexicase_selection(population,k, data, eps)

    p = individual.primitives
    s = individual.set_dict

    init_options = ['full', 'grow']
    first_parent = deepcopy(individual)
    second_parent = Tree(p, s, choice(init_options), randint(1, max_depth))
    new_individual = _crossover(first_parent, first_parent.random_node(), second_parent.root)
    new_score = fitness(new_individual, data)

    if (elitist == 0) or new_score < score:
        return new_individual
    else:
        return individual

def reproduction(population,sel_type,eps, k, data):
    num_cases = int(len(data) * 0.5)
    cases = random.sample(data, num_cases)

    if (sel_type == "torneio"):
        individual, score = tournament_selection(population, k, data)
    elif (sel_type == "roleta"):
        individual, score = roulette_selection(population, data)
    elif (sel_type == "lexicase"):
        individual, score = epsilon_lexicase_selection(population,k,data,eps)

    return deepcopy(individual)

def fitness(tree, dataset):

    prog = tree.build_program(tree.root)
    variables = tree.set_dict["variaveis"]
    m = len(variables)
    yi = []
    y_eval = []

    for item in dataset:
        for i in range(m):
            vars()[variables[i]] = item[i]
        try:
            result = eval(prog)
        except:
            result = 0.0
        yi.append(item[-1])
        y_eval.append(result)

    yi = np.array(yi)
    y_eval = np.array(y_eval)

    #rmse = 0.0
    #rmse = sqrt((np.sum(yi - y_eval) ** 2) / yi.size)

    nrmse = 0.0
    nrmse = sqrt(np.sum((yi - y_eval) ** 2))
    nrmse = nrmse/sqrt(np.sum((yi - np.mean(yi)) ** 2))

    # Compute the size penalty to avoid bloat
    #penalty = 0.1
    #size_penalty = penalty * tree.size()

    # Add the size penalty to the fitness
    #nrmse = nrmse + size_penalty

    return nrmse

def tournament_selection(population, n, data):

    pop_sample = sample(population, n)
    best = None
    best_score = None
    for item in pop_sample:
        score = fitness(item, data)
        if (best_score == None) or (score < best_score):
            best = item
            best_score = score

    if best == None:
        return tournament_selection(population, n, data)

    return best, best_score

def roulette_selection(population, data):
    weights = [fitness(individual, data) for individual in population]
    max_weight = max(weights)
    scaled_weights = [max_weight - weight + 1e-6 for weight in weights]
    total_weight = sum(scaled_weights)
    probabilities = [weight / total_weight for weight in scaled_weights]
    index = np.random.choice(len(population), p=probabilities)
    if population[index] == None:
        return roulette_selection(population, data)
    return population[index], weights[index]

def epsilon_lexicase_selection(population, n, data, epsilon):
    # Sorteia um conjunto de casos de teste
    test_cases = sample(population, n)
    # Ordena os casos de teste lexicograficamente
    #test_cases.sort()
    # Inicializa a lista de indivíduos selecionados
    selected = []
    # Para cada indivíduo na população
    for individual in population:
        # Inicializa a flag de falha
        failed = False
        # Para cada caso de teste em ordem lexicográfica
        for case in test_cases:
            # Avalia o indivíduo para o caso de teste
            score = fitness(case, data)
            # Se o erro do indivíduo para o caso de teste for maior que epsilon
            if score > epsilon:
                # Marca o indivíduo como falho e para a avaliação
                failed = True
                break
        # Se o indivíduo passou em todos os casos de teste
        if not failed:
            # Adiciona o indivíduo à lista de selecionados
            selected.append(individual)
    # Se nenhum indivíduo passou em todos os casos de teste
    if not selected:
        # Seleciona um indivíduo aleatório da população
        return epsilon_lexicase_selection(population, n, data, epsilon)
    # Seleciona o melhor indivíduo dentre os selecionados
    best = None
    best_score = None
    for item in selected:
        score = fitness(item, data)
        if (best_score == None) or (score < best_score):
            best = item
            best_score = score

    if (best == None) or (best_score == None):
        return epsilon_lexicase_selection(population, n, data, epsilon)

    return best, best_score


def _crossover(tree1, cross_pt1, cross_pt2):

    parent = cross_pt1.parent
    aux = cross_pt2
    if parent == None:
        tree1.root = aux
        aux.parent = None
    elif parent.left == cross_pt1:
        parent.left = aux
        aux.parent = parent
    elif parent.right == cross_pt1:
        parent.right = aux
        aux.parent = parent

    return tree1


def evaluation(population, data):

    best = None
    best_score = None
    scores = []
    individuals = {}

    for individual in population:
        score = fitness(individual, data)
        scores.append(score)
        if score in individuals:
            individuals[score] += 1
        else:
            individuals[score] = 1
        if (best_score == None) or (score < best_score):
            best = individual
            best_score = score

    worst_score = np.max(scores)
    mean_score = np.mean(scores)

    repeated = 0
    for score in scores:
        if individuals[score] != 1:
            repeated += 1

    return best, best_score, worst_score, mean_score, repeated