import pandas as pd
"""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt"""
from DecisionTreeClassifier import DecisionTreeClassifier
from tree_visualizer import visualize_tree

train_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/train_data.csv", header=0)
test_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/test_data.csv", header=0)

"""model = DecisionTreeClassifier(max_depth=3)
model.fit(train_data[["X"]], train_data["y"])

test_data["Предсказанный_Зачет"] = model.predict(test_data[["X"]])"""

"""print(classification_report(test_data["y"], test_data["Предсказанный_Зачет"]))
tree.plot_tree(model, feature_names=["X"], filled=True)
plt.show()"""

model = DecisionTreeClassifier(max_depth=5, metric="entropy")
model.fit(train_data["X"].tolist(), train_data["y"].tolist())

model.print_tree(model.root, 0)
visualize_tree(model)
