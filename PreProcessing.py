import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder
import seaborn as sns

%matplotlib inline

# Leitura do Dataset
path = kagglehub.dataset_download("kingabzpro/gambling-behavior-bustabit")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(csv_file_path)

# Limpeza de Dados
imputer = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value= -1)
imputer = imputer.fit(df[['CashedOut', 'Bonus', 'Profit']])
df[['CashedOut', 'Bonus', 'Profit']] = imputer.transform(df[['CashedOut', 'Bonus', 'Profit']])

# Conversão de Tipos de Dados
df['Username'] = df['Username'].astype(str)

# Remoção de Duplicatas
df.duplicated()
df.drop_duplicates()

# Normalização e Padronização de Dados
scaler = MinMaxScaler()
df[['BetScaledColumn']] = scaler.fit_transform(df[['Bet']])
scaler = StandardScaler()
df[['BettandardizedColumn']] = scaler.fit_transform(df[['Bet']])

# Encoding
df['Condition'] = df['CashedOut'].apply(lambda x: 'Won' if x > 0 else 'Lost')
encoder = OneHotEncoder(sparse_output=False)
encoded_data = pd.DataFrame(encoder.fit_transform(df[['Condition']]),
                            columns=encoder.get_feature_names_out(['Condition']))
df = pd.concat([df, encoded_data], axis=1)

# Detecção e Tratamento de Outliers
Q1 = df['Bet'].quantile(0.25)
Q3 = df['Bet'].quantile(0.75)
IQR = Q3 - Q1
df['outliers'] = (df['Bet'] < (Q1 - 1.5 * IQR)) | (df['Bet'] > (Q3 + 1.5 * IQR))
idx = df[df['outliers'] == True].index
df.drop(idx, inplace=True)

#Fazendo uma tabela filtrada com os ganhadores e perdedores
df_winners = df[df['Condition_Won'] == 1]
df_losers = df[df['Condition_Lost'] == 1]

sns.histplot(data=df_winners, x='CashedOut')
plt.title('Quanto os ganhadores sacaram')
plt.xlabel('Saque')
plt.ylabel('Usuarios')
plt.grid(True)
plt.show()

sns.histplot(data=df_losers, x='Bet')
plt.title('Quanto os usuários perderam')
plt.xlabel('Aposta')
plt.ylabel('Usuarios')
plt.grid(True)
plt.show()

print(f'\nUsando df.hist: \n{df.hist()}')

# Log Transform
df_winners['CashedOut_Log'] = np.log(df_winners['CashedOut'])
df_losers['Bet_Log'] = np.log(df_losers['Bet'])

sns.histplot(data=df_winners, x='CashedOut_Log')
plt.title('Quanto os ganhadores sacaram após transformação logarítmica')
plt.xlabel('Saque')
plt.ylabel('Usuarios')
plt.grid(True)
plt.show()

sns.histplot(data=df_losers, x='Bet_Log')
plt.title('Quanto os usuários perderam após transformação logarítmica')
plt.xlabel('Aposta')
plt.ylabel('Usuarios')
plt.grid(True)
plt.show()
