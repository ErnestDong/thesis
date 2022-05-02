import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
x = [i / 10 ** 3 for i in range(2000)]
y_assets = [i if i < 0.8 else 0.8 for i in x]
y_bond = [i if i < 0.6 else 0.6 for i in x]
y_equity = [y_assets[i] - y_bond[i] for i in range(2000)]
res = pd.DataFrame({"Asset": y_assets, "Bond": y_bond, "Equity": y_equity}, index=x)
res.plot()
filename = "plts/payoffs.png"
plt.savefig(filename)
