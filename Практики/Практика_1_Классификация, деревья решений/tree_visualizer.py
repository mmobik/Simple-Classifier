import matplotlib.pyplot as plt
from DecisionTreeClassifier import DecisionTreeClassifier


def _assign_indices_to_leaves(root, current_index):
    if root.left is None and root.right is None:
        root.leaf_index = current_index
        current_index += 1
        return current_index
    
    current_index = _assign_indices_to_leaves(root.left, current_index)
    current_index = _assign_indices_to_leaves(root.right, current_index)

    return current_index

def _assign_positions(node, depth=0):
    node.depth = depth
    if node.left is None and node.right is None:
        node.position = node.leaf_index
        print(node.position)
        return node.leaf_index
    
    min_pos = _assign_positions(node.left, depth + 1)
    max_pos = _assign_positions(node.right, depth + 1)
    mid = round((min_pos + max_pos) / 2)
    print(mid)
    node.position = mid
    return node.position

def draw(node, is_right_child=False):
    if node.value is not None:
        text = f"class: {node.value}"
    else:
        text = f"X <= {node.threshold}"

    # Если это правый лист, добавляем смещение
    x = node.position * 100
    if is_right_child and node.left is None and node.right is None:
        x += 100
    
    y = node.depth * 100
    rectangle = plt.Rectangle((x, y), 50, 50, edgecolor="blue", facecolor="lightblue", fill=True)
    plt.gca().add_patch(rectangle)
    
    if node.left is not None:
        plt.arrow(x, y + 50, -50, 50, fc='black', ec='black')
    if node.right is not None:
        plt.arrow(x + 50, y + 50, 50, 50, fc='black', ec='black')

    plt.text(x + 25, y + 25, text, ha='center', va='center')

    if node.left:
        draw(node.left, is_right_child=False)
    if node.right:
        draw(node.right, is_right_child=True)
    
    

def visualize_tree(tree):
    """Функция для визаулизации решающего дерева"""
    root = tree.root
    leaves_count = _assign_indices_to_leaves(root, 0)
    _assign_positions(root)

    plt.figure(figsize=(8, 8))

    # Т.к Листья на позициях: 0, 100, 200, 300. Максимальная позиция = 300, а count=4, эмприрически подобрано
    x = (leaves_count + 1) * 100
    y = (tree.get_depth() + 1) * 100
    plt.xlim(-50, x + 50)
    plt.ylim(-50, y)
    draw(root)
    plt.show()
