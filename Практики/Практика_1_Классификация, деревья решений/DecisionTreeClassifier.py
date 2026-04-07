import math

class Node:
    """Базовый класс представляет узел дерева"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Parameters:
        feature_index: Индекс признака, используемого для разделения
        thereshold: Пороговое значение для разделения
        left: Ссылка на левый узел
        right: Ссылка на правый узел
        value: Значнеие узла
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    """Классификатор решений по классам"""
    def __init__(self, max_depth=None, min_samples_split=2, metric="gini"):
        """
        Parametrs:
        max_depth: Максимальная глубина дерева
        min_samples_split: Минимальное разбиение
        min_impurity_decrease
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.depth = 0
        self.metric = metric

    def _build_tree(self, X, y, depth=0):
        """Рекурсивно строит дерево"""

        if self._should_stop(y, depth):
            return Node(value=self._get_most_common_label(y))
        
        best_gini, best_threshold, best_groups = self._find_best_split(X, y)
        if best_gini == float("inf"):
            return Node(value=self._get_most_common_label(y))
        
        left_X = []
        right_X = []
        left_y , right_y = best_groups

        for feature, label in zip(X, y):
            if feature <= best_threshold:
                left_X.append(feature)
            else:
                right_X.append(feature)
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        

        return Node(
            feature_index=0,
            threshold = best_threshold,
            left=left_child,
            right = right_child)

    def _find_best_split(self, X, y):
        """Находит лучшее разделение для текущего узла"""
        features = set(X)
        groups = {}
        best_gini = float("inf")
        best_groups = []
        best_threshold = 0
        # Наилучшее разделение достигается при наименьшей G(I)
        for threshold in features:
            group_1 = []
            group_2 = []
            for feature, result in zip(X, y):
                if feature <= threshold:
                    group_1.append(result)
                else:
                    group_2.append(result)
            groups[threshold] = [group_1, group_2]
        
        for key, value in groups.items():
            if not value[0] or not value[1]:
                continue
            split_impurity = self._calculate_split_impurity(value[0], value[1])
            if split_impurity < best_gini:
                best_gini = split_impurity
                best_groups = groups[key]
                best_threshold = key
            else:
                continue

        return [best_gini, best_threshold, best_groups]
            
    def _should_stop(self, y, depth):
        """Проверяет условия остановки"""
        if len(set(y)) == 1:
            return True
        
        if self.max_depth is not None and depth > self.max_depth:
            return True
        
        if len(y) < self.min_samples_split:
            return True
        
        return False
    
    def _calculate_gini(self, labels):
        """Вычисляет коэффициент Джини"""
        m = len(labels)
        p = sum(x == 0 for x in labels) / m
        q = sum(x == 1 for x in labels) / m
        return 1 - p ** 2 - q ** 2
    
    def _calculate_split_impurity(self, left_y, right_y):
        """Вычисляет взвешенную нечистоту разделения"""
        m1 = len(left_y)
        m2 = len(right_y)
        m = m1 + m2
        if self.metric == "gini":
            return m1 / m * self._calculate_gini(left_y) + m2 / m * self._calculate_gini(right_y)
        if self.metric == "entropy":
            return m1 / m * self._calculate_entropy(left_y) + m2 / m * self._calculate_entropy(right_y)
        else:
            raise ValueError("Неизвестный критерий для разбиения")
    
    def _calculate_entropy(self, labels):
        if not labels:
            return 0
        elif len(set(labels)) == 1:
            return 0
        m = len(labels)
        p = sum(x== 0 for x in labels) / m
        q = sum(x== 1 for x in labels) / m
        return -(p * math.log(p, 2) + q * math.log(q, 2))
    
    def _get_most_common_label(self, y):
        """Возвращает наиболее частый класс"""
        if not y:
            return None

        count_0 = sum(1 for label in y if label == 0)
        count_1 = len(y) - count_0

        if count_0 > count_1:
            return 0
        elif count_1 > count_0:
            return 1
        # При одинаковой длине приоритет отдаем единице
        else:
            return 1

    def get_depth(self):
        """Возвращает глубину дерева"""
        return self._calculate_depth(self.root)
    
    def _calculate_depth(self, node):
        """Рекурсивно вычисляет глубину"""
        if node is None:
            return 0
        if node.value is not None:
            return 0
        
        return 1 + max(self._calculate_depth(node.left), self._calculate_depth(node.right))

    def fit(self, X, y):
        """Обучает дерево на данных"""
        self.root = self._build_tree(X, y, depth=0)
         
    def _predict_single(self, x, node):
        """Предсказывает классы для новых данных"""
        if node.value is not None:
            return node.value
        if x <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, x):
        if isinstance(x, int):
            return self._predict_single(x, node=self.root)
        return [self._predict_single(el, self.root) for el in x]

    def print_tree(self, node, level, side="root"):
        if node is None:
            return

        indent = "  " * level
        if node.value is not None:
            print(f"{indent}{side}: Лист -> класс {node.value}")
        else:
            print(f"{indent}{side}: Узел -> признак {node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, level + 1, "left")
            self.print_tree(node.right, level + 1, "right")
    
    def accuracy(self, X: list[int], targets: list[int]):
        total_predicted = 0
        predicted = self.predict(X)
        for predict, target in zip(predicted, targets):
            if predict == target:
                total_predicted += 1
        return total_predicted / len(X)
    