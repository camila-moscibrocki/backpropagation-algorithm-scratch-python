from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


# carrega o arquivo CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        return dataset

# Converte a string 'column' em float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Converte a string 'column' em um integrador
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Encontra os valores maximos e minimos de cada coluna
def dataset_minmax(datase):
    minmax = list()
    stats =[[min(column), max(column)] for column in zip(*dataset)]
    return stats

# reestrutura as colunas do conjunto de dados em uma escala de 0 - 1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0])/ (minmax[i][1] - minmax[i][0])

# divide o conjunto de dados em subconjuntos k
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# calcula a porcentagem de acerto
def acuracy_metric(actual, predicted):
    correct = 0
    for i in  range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
