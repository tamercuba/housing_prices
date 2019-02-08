import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score

houses_data = pd.read_csv('train.csv', index_col = 'Id')
teste1= pd.read_csv('test.csv')
teste = pd.read_csv('test.csv', index_col = 'Id')
classe = houses_data.SalePrice

houses_features = ['LotArea','TotRmsAbvGrd','1stFlrSF',]
teste = teste[houses_features]
previsores = houses_data[houses_features]
#revisores_1 = houses_data[houses_features]

model = DecisionTreeRegressor(random_state=0)
model.fit(previsores, classe)
previsoes = model.predict(teste)
resultado = pd.DataFrame(previsoes, index = teste1['Id'])
resultado.to_csv('result.csv', sep=',')
