from random import seed
from random import random
from math import exp


# Inicialização da rede

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Criação da primeira rede
# Camada oculta tem um neurônio com 2 pesos de entrada mais o viés
# seed(1)
# network = initialize_network(2, 1, 2)
# for layer in network:
# print(layer) - output oculta


# Propagação futura - Ativação do Neurônio weight é um peso da rede, input é uma entrada, i é o índice de um peso ou
# uma entrada e bias é um peso especial que não tem entrada para multiplicar com
# activation = sum(weight_i * input_i) + bias


# Criação da função activate() - A função assume que o viés é o último peso na lista de pesos

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transferência do neuronio
# output = 1 / (1 + e ^ (-activation))


# Transferência ativação do neurônio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Propagação de retorno - implementa a propagação direta de uma linha de dados do conjunto de dados com a rede
# neural
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Define a rede em linha com um neurônio oculto que espera 2 valores de entrada e uma camada de saída com dois neurônios
# Calcula a ativação do neurônio para uma entrada - teste da propagação de retorno
# network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#           [{'weights': [0.2550690257394217, 0.49543508709194095]},
#            {'weights': [0.4494910647887381, 0.651592972722763]}]]
# row = [1, 0, None]
# output = forward_propagate(network, row)
# print(output)


# Erro de retropropagação
# Calcula a derivada de uma saída de neurônio
def transfer_derivative(output):
    return output * (1.0 - output)


# para o cálculo de erro temos: error = (expected - output) * transfer_derivative(output) onde "expected" é o valor
# da saída esperada para o neuronio, "output" é o valor atribuido ao neuronio e "transfer_derivative" calcula a
# inclinação do valor de saída do neurônio

# Erro de retropropagação e armazenamento em neuronios
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
#    print(layer)

# Os pesos da rede são atualizados da seguinte forma - weight = weight + learning_rate * error * input Onde: Onde
# weight é um dado peso, learning_rate é um parâmetro a ser especificado, error é o erro calculado pelo procedimento
# de retropropagação para o neurônio e entrada é o valor de entrada que causou o erro

# Atualizando os pesos da rede de acordo com o erro
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']


# Treino da RNA - A rede é atualizada usando a descida de gradiente estocástico
# Treino efetuado a partir de um número fixo de epochs

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Teste - Treino do algoritimo de retropropagação
seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
    print(layer)
