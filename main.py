import time
import pandas as pd
import os
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Como sera reaproveitado, deixei como global a base de testes
x_testes, y_testes = load_svmlight_file("dados/test.txt")
dados_testes = [x_testes.toarray(), y_testes]


# carrega todos os 20 mil registros de treinamento e
def carrega_conjunto_treinamento():
    x_data, y_data = load_svmlight_file("dados/train.txt")
    return [x_data.toarray(), y_data]


# pega o tamanho desejado da base de treinamento
def get_conjunto_treinamento(tudo, tamanho):
    return [tudo[0][:tamanho], tudo[1][:tamanho]]

# generaliza a execução do treino e teste e retorna vetor de resultado
def treinar_e_testar(classificador_nome, classificador, dados_treinamento, tamanho):
    # Treinamento
    amostras = get_conjunto_treinamento(dados_treinamento, tamanho)
    inicio_treinamento = time.time()
    classificador.fit(amostras[0], amostras[1])
    tempo_treinamento = time.time() - inicio_treinamento
    # Teste
    inicio_testes = time.time()
    predict = classificador.predict(dados_testes[0])
    tempo_testes = time.time() - inicio_testes
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    f1score = f1_score(dados_testes[1], predict, labels=labels, average='weighted')
    acuracia = classificador.score(dados_testes[0], dados_testes[1])
    matriz_de_confusao = confusion_matrix(dados_testes[1], predict)
    filename_matriz = classificador_nome + '_' + str(tamanho) + '.csv'
    pd.DataFrame(matriz_de_confusao).to_csv('out/'+filename_matriz, decimal=',', index=False, sep=';', float_format='%.4f')
    return [tempo_testes, tempo_treinamento, f1score, acuracia, matriz_de_confusao]


def knn(dados_treinamento, tamanho):
    print(f'kNN com {tamanho} amostras...')
    classificador = KNeighborsClassifier()
    resultado = treinar_e_testar('knn', classificador, dados_treinamento, tamanho)
    return ['knn', tamanho] + resultado[:-1]  # exclui, por hora, matriz de confusao


def naive_bayes(dados_treinamento, tamanho):
    print(f'Naive Bayes com {tamanho} amostras...')
    classificador = GaussianNB()
    resultado = treinar_e_testar('naive_bayes', classificador, dados_treinamento, tamanho)
    return ['naive_bayes', tamanho] + resultado[:-1]  # exclui, por hora, matriz de confusao


def lda(dados_treinamento, tamanho):
    print(f'LDA com {tamanho} amostras...')
    classificador = LinearDiscriminantAnalysis()
    resultado = treinar_e_testar('lda', classificador, dados_treinamento, tamanho)
    return ['lda', tamanho] + resultado[:-1]  # exclui, por hora, matriz de confusao


def logistic_regression(dados_treinamento, tamanho):
    print(f'Regressao Logistica com {tamanho} amostras...')
    classificador = LogisticRegression(max_iter=200)
    resultado = treinar_e_testar('regressao_logistica', classificador, dados_treinamento, tamanho)
    return ['regressao_logistica', tamanho] + resultado[:-1]  # exclui, por hora, matriz de confusao


def perceptron(dados_treinamento, tamanho):
    print(f'Perceptron com {tamanho} amostras...')
    classificador = Perceptron()
    resultado = treinar_e_testar('perceptron', classificador, dados_treinamento, tamanho)
    return ['perceptron', tamanho] + resultado[:-1]  # exclui, por hora, matriz de confusao


# apaga os arquivos da pasta out
def limpa_diretorio():
    for file in os.scandir('out'):
        if file.name.endswith(".csv"):
            os.unlink(file.path)


# roda os testes
def executa_testes():
    limpa_diretorio()
    resultados = []
    dados_treinamento = carrega_conjunto_treinamento()
    lista_classificadores = ['naive_bayes', 'lda', 'logistic_regression', 'perceptron', 'knn']
    for classificador in lista_classificadores:
        for qtde_amostras in range(1000, 20001, 1000):
            if classificador == 'knn':
                resultados.append(knn(dados_treinamento, qtde_amostras))
            elif classificador == 'naive_bayes':
                resultados.append(naive_bayes(dados_treinamento, qtde_amostras))
            elif classificador == 'lda':
                resultados.append(lda(dados_treinamento, qtde_amostras))
            elif classificador == 'logistic_regression':
                resultados.append(logistic_regression(dados_treinamento, qtde_amostras))
            elif classificador == 'perceptron':
                resultados.append(perceptron(dados_treinamento, qtde_amostras))
            else:
                print(f'Classificador {classificador} não implementado ainda!')

    header_csv = ['Classificador', 'Amostras', 'Tempo de Treinamento', 'Tempo de Testes', 'Acuracia', 'F1Score']
    pd.DataFrame(resultados).to_csv('out/resultados.csv', header=header_csv, decimal=',', index=False, sep=';', float_format='%.4f')


if __name__ == "__main__":
    executa_testes()
