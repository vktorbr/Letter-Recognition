#importando bibliotecas necessarias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#carregamento do dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'

#Definindo nome das colunas do dataset
names = ['class', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar',
         'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
dataset = pandas.read_csv(url, names = names)

print('Shape dos dados:')
print(dataset.shape)

print('Visualizando o conjunto inicial dos dados:')
print(dataset.head(20))

print('Conhecendo os dados estatisticos dos dados carregados:')
print(dataset.describe())

print('Conhecendo a distribuicao dos dados por classes:')
print(dataset.groupby('class').size())

print('Criando graficos de caixa da distribuicao das classes')
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
#plt.savefig('grafico de caixa.png')
plt.show()

print('Criando histogramas dos dados por classes')
dataset.hist()
#plt.savefig('histograma.png')
plt.show()

print('Criando graficos de dispersao dos dados')
axs = scatter_matrix(dataset)
#plt.yticks(rotation=0)
#plt.xticks(rotation=90)
n = len(dataset.columns) - 1
for x in range(n):
    for y in range(n):
        ax = axs[x, y]
        ax.xaxis.label.set_rotation(0)
        ax.yaxis.label.set_rotation(0)
#plt.savefig('grafico de dispersao.png')
#plt.show()