names = [2, 5, 7, 2, 10, 2, 3, 4]
ages = [1, 1, 1, 0, 1, 0, 0, 0]

def calculate_gini(labels):
    """Вычисляет коэффициент Джини"""
    m = len(labels)
    p = sum(x == 0 for x in labels) / m
    q = sum(x == 1 for x in labels) / m
    return 1 - p ** 2 - q ** 2

def calculate_split_impurity(left_y, right_y):
    """Вычисляет взвешенную нечистоту разделения"""
    m1 = len(left_y)
    m2 = len(right_y)
    m = m1 + m2
    return m1 / m * calculate_gini(left_y) + m2 / m * calculate_gini(right_y)

def _find_best_split(X, y):
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
        split_impurity = calculate_split_impurity(value[0], value[1])
        if split_impurity < best_gini:
            best_gini = split_impurity
            best_groups = groups[key]
            best_threshold = key
        else:
            continue

    return [best_gini, best_threshold, best_groups]

 = _find_best_split(names, ages)
print(left_x, left_y)