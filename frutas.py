# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%
from sklearn import tree
arvores = tree.DecisionTreeClassifier(random_state=42)

# %%
y = df['Fruta']
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]

# %%
arvores.fit(x, y)


# %%

arvores.predict([[0,0,0,0]])


# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

tree.plot_tree(arvores, feature_names=caracteristicas, class_names=arvores.classes_, filled=True)
