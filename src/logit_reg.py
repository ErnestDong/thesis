#%%
import configparser

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    plot_confusion_matrix,
    plot_roc_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sqlalchemy import create_engine
import seaborn as sns

rc = {"font.sans-serif": "SimHei", "axes.unicode_minus": False}
sns.set(style="white", font="SimHei", context="paper", rc=rc)

#%%
conf = configparser.ConfigParser()
conf.read("config.ini", encoding="utf-8")
engine = create_engine("mysql+pymysql://root:Lyj822919@localhost/thesis")
res_sql = "test"
res = pd.read_sql(res_sql, engine, index_col="enterprise")
Y = res["default"]
X = res.drop(columns=["default"])
X = X.copy()
X = sm.add_constant(X)
model = sm.Logit(Y, X.astype(float))
result = model.fit()
print(result.summary().as_latex())
# pred = result.predict(X)
# fig, ax = plt.subplots()
# result = pred.sort_values().reset_index(drop=True)
# plt.plot(result)
# ax.scatter([(6412 - 203)], [0.114852], marker="x")
# plt.xticks = []
# plt.show()

# %%
y_pred = pred.apply(lambda x: 1 if x > 0.5 else 0)
print("Accuracy: {:.2f}".format(accuracy_score(Y, y_pred)))
print("Error rate: {:.2f}".format(1 - accuracy_score(Y, y_pred)))
print("Precision: {:.2f}".format(precision_score(Y, y_pred)))
print("Recall: {:.2f}".format(recall_score(Y, y_pred)))
print("f1_score: {:.2f}".format(f1_score(Y, y_pred)))

# %%
logreg = LogisticRegression()
logreg.fit(X, Y)
# fig = plot_confusion_matrix(logreg, X, y_pred)
logit_roc_auc = roc_auc_score(Y, logreg.predict(X))
fpr, tpr, thresholds = roc_curve(Y, logreg.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic ROC")
plt.legend(loc="lower right")

plt.show()
plt.savefig("logreg/logreg_roc.png")
# %%
