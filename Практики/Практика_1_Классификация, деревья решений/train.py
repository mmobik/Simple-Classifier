import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt

train_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/train_data.csv")
test_data = pd.read_csv("Практики/Практика_1_Классификация, деревья решений/data/test_data.csv")

model = DecisionTreeClassifier(max_depth=3)
model.fit(train_data[["Лекции", "Семинары"]], train_data["Зачет"])
test_data["Предсказанный_Зачет"] = model.predict(test_data[["Лекции", "Семинары"]])

print(classification_report(test_data["Зачет"], test_data["Предсказанный_Зачет"]))
tree.plot_tree(model, feature_names=["Лекции", "Семинары"], filled=True)
plt.show()