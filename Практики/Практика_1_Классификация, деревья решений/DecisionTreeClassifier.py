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
        pairs = sorted(zip(X, y))
        n = len(pairs)
        
        # Начальные счетчики (все справа)
        right_0 = sum(1 for _, label in pairs if label == 0)
        right_1 = n - right_0
        left_0 = 0
        left_1 = 0
        
        best_gini = float('inf')
        best_threshold = None
        best_left_y = []
        best_right_y = []
        
        i = 0
        while i < n - 1:
            current_x = pairs[i][0]
            
            # Перекидываем все объекты с одинаковым X
            while i < n and pairs[i][0] == current_x:
                _, label = pairs[i]
                if label == 0:
                    left_0 += 1
                    right_0 -= 1
                else:
                    left_1 += 1
                    right_1 -= 1
                i += 1
            
            if i >= n or left_0 + left_1 == 0 or right_0 + right_1 == 0:
                continue
            
            next_x = pairs[i][0]
            threshold = (current_x + next_x) / 2
            
            # Считаем gini через счетчики без создания списков
            gini = self._calculate_split_impurity_from_counts(
                left_0, left_1, right_0, right_1
            )
            
            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold
                # Запоминаем группы только для лучшего порога (один раз)
                best_left_y = [label for x_val, label in pairs if x_val <= threshold]
                best_right_y = [label for x_val, label in pairs if x_val > threshold]
        
        if best_threshold is None:
            return [float('inf'), None, None]
        
        return [best_gini, best_threshold, [best_left_y, best_right_y]]

    def _calculate_split_impurity_from_counts(self, left_0, left_1, right_0, right_1):
        """Вычисляет взвешенную нечистоту разделения по счетчикам"""
        n_left = left_0 + left_1
        n_right = right_0 + right_1
        n_total = n_left + n_right
        
        if n_left == 0 or n_right == 0:
            return float('inf')
        
        if self.metric == "gini":
            gini_left = 1 - (left_0/n_left)**2 - (left_1/n_left)**2
            gini_right = 1 - (right_0/n_right)**2 - (right_1/n_right)**2
            return (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
        
        elif self.metric == "entropy":
            ent_left = 0
            ent_right = 0
            if left_0 > 0:
                ent_left -= (left_0/n_left) * math.log(left_0/n_left, 2)
            if left_1 > 0:
                ent_left -= (left_1/n_left) * math.log(left_1/n_left, 2)
            if right_0 > 0:
                ent_right -= (right_0/n_right) * math.log(right_0/n_right, 2)
            if right_1 > 0:
                ent_right -= (right_1/n_right) * math.log(right_1/n_right, 2)
            return (n_left/n_total) * ent_left + (n_right/n_total) * ent_right
        
        else:
            raise ValueError("Неизвестный критерий для разбиения")

            
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
    