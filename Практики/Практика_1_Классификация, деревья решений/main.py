def get_gini_impurity(group: list[int]) -> float:
    m = len(group)
    p = sum(x == 0 for x in group) / m
    q = sum(x == 1 for x in group) / m
    return 1 - p ** 2 - q ** 2


def get_impurity(group1: list[int], group2: list[int]) -> float:
    m1 = len(group1)
    m2 = len(group2)
    m = m1 + m2
    return m1 / m * get_gini_impurity(group1) + m2 / m * get_gini_impurity(group2)


"""def should_stop(node):
    if get_gini_impurity(node) == 0:
        return True
    
    if len(node) < min_samples:
        return True
    
    if depth >= max_depth:
        return True
    
    if improvement < min_improvement:
        return True
    
    return False
"""


def model(feature: list[int], results: list[int]) -> float:
    best_impurity = float('inf')
    best_threshold = None
    for threshold in set(feature):
        group1 = [y for x, y in zip (feature, results) if x <= threshold]
        group2 = [y for x, y in zip(feature, results) if x > threshold]

        if not group1 or not group2:
            continue

        impurity = get_impurity(group1, group2)

        if impurity < best_impurity:
            best_impurity = impurity
            best_threshold = threshold
    
    print(f"Лучший порог: {best_threshold}, нечистота: {best_impurity:.3f}")
    print(group1, group2)
    return best_impurity


lectures = [8, 6, 8, 8, 8, 6]
results  = [0, 0, 1, 1, 1, 1]

model(lectures, results)

