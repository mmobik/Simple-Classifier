import pandas as pd
"""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt"""
from DecisionTreeClassifier import DecisionTreeClassifier
from tree_visualizer import visualize_tree
import time
import tracemalloc

train_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/train_data.csv", header=0)
test_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/test_data.csv", header=0)

"""model = DecisionTreeClassifier(max_depth=3)
model.fit(train_data[["X"]], train_data["y"])

test_data["Предсказанный_Зачет"] = model.predict(test_data[["X"]])"""

"""print(classification_report(test_data["y"], test_data["Предсказанный_Зачет"]))
tree.plot_tree(model, feature_names=["X"], filled=True)
plt.show()"""

start = tracemalloc.start()
start_time = time.time()
model = DecisionTreeClassifier(max_depth=6, metric="gini")
model.fit(train_data["X"].tolist(), train_data["y"].tolist())
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()

print(f"Текущая память: {current / 1024:.2f} КБ")
print(f"Пиковая память: {peak / 1024:.2f} КБ")
print(f"Время обучения: {end - start_time:.4f} секунд")

real_data = test_data["y"].tolist()
x = test_data["X"].tolist()
y = test_data["y"].tolist()

print(f"Точность: {model.accuracy(x, y)}")

visualize_tree(model)
