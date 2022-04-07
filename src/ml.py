#%%
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    plot_roc_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
sns.set(style="white", context="paper")
#%%

engine = create_engine("mysql+pymysql://root:Lyj822919@localhost:3306/thesis")
res = pd.read_sql("logit_model", engine, index_col="index")
dt = tree.DecisionTreeClassifier(
    criterion="entropy",
    random_state=30,
    max_depth=4,
    # min_samples_leaf=10,
    min_samples_split=10,
)
rf = RandomForestClassifier()
lr = LogisticRegression()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    res.drop(columns=["default"]), res["default"], test_size=0.3
)
# Xtrain, Ytrain = res.drop(columns=["default"]), res["default"]
dt = dt.fit(Xtrain, Ytrain)
rf = rf.fit(Xtrain, Ytrain)
lr = lr.fit(Xtrain, Ytrain)
# score = clf.score(Xtest, Ytest)

dot_data = tree.export_graphviz(
    dt,
    feature_names=res.drop(columns=["default"]).columns,
    class_names=["no default", "default"],
    filled=True,
    rounded=True,
)

graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render("ml/decision_tree", view=False)
#%%
for model in [lr, dt, rf]:
    predictions = model.predict(Xtest)
    cm = confusion_matrix(Ytest, predictions, labels=dt.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
    name = str(model.__class__.__name__) + "\t"
    print(name + "Accuracy: {:.2f}".format(accuracy_score(Ytest, predictions)))
    print(name + "Error rate: {:.2f}".format(1 - accuracy_score(Ytest, predictions)))
    print(name + "Precision: {:.2f}".format(precision_score(Ytest, predictions)))
    print(name + "Recall: {:.2f}".format(recall_score(Ytest, predictions)))
    print(name + "f1_score: {:.2f}".format(f1_score(Ytest, predictions)))
    disp.plot()
    plt.savefig(f"ml/{str(model.__class__.__name__)}.png")

#%%
disp = plot_roc_curve(lr, Xtest, Ytest)
plot_roc_curve(dt, Xtest, Ytest, ax=disp.ax_)
plot_roc_curve(rf, Xtest, Ytest, ax=disp.ax_)
plt.savefig("ml/roc.png")
# %%
