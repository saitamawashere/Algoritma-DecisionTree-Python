from collections import Counter
import math

def get_class_labels(data):
    return [row[-1] for row in data]

def get_entropy(data):
    class_labels = get_class_labels(data)
    class_counts = Counter(class_labels)
    total_instances = len(class_labels)
    entropy = 0
    for count in class_counts.values():
        probability = count / total_instances
        entropy -= probability * log_base_2(probability)
    return entropy

def log_base_2(x):
    return math.log2(x) if x > 0 else 0

def split_data(data, attribute_index, value):
    return [row for row in data if row[attribute_index] == value]

def get_best_attribute(data, attributes):
    entropies = []
    for attribute_index in range(len(attributes)):
        attribute_values = set(row[attribute_index] for row in data)
        attribute_entropy = 0
        for value in attribute_values:
            subset = split_data(data, attribute_index, value)
            subset_entropy = get_entropy(subset)
            subset_probability = len(subset) / len(data)
            attribute_entropy += subset_probability * subset_entropy
        entropies.append(attribute_entropy)
    return attributes[entropies.index(min(entropies))]

def build_decision_tree(data, attributes):
    class_labels = get_class_labels(data)
    if len(set(class_labels)) == 1:
        return class_labels[0]
    if len(attributes) == 0:
        return Counter(class_labels).most_common(1)[0][0]
    best_attribute = get_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    attribute_index = attributes.index(best_attribute)
    attribute_values = set(row[attribute_index] for row in data)
    for value in attribute_values:
        subset = split_data(data, attribute_index, value)
        remaining_attributes = attributes[:attribute_index] + attributes[attribute_index+1:]
        tree[best_attribute][value] = build_decision_tree(subset, remaining_attributes)
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = instance[attributes.index(attribute)]
    if value not in tree[attribute]:
        return "Unknown"
        #return (play for play in get_class_labels(data) if play == 'No').__next__() or 'Yes'
        #jika menggunakan kode diatas, maka akan menghasilkan output 'Yes' jika tidak ada 'No' pada data, akan tetapi kita menggunakan unknown karna hasil data tidak ada pada dataset
    subtree = tree[attribute][value]
    return predict(subtree, instance)

# Main program
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No']
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

decision_tree = build_decision_tree(data, attributes)
print("Decision Tree:")
print(decision_tree)

# Predict new instances
new_instances = [
    ['Rainy', 'Cool', 'High', 'Strong', 'No'],
    ['Sunny', 'Hot', 'Normal', 'Weak', 'Yes'],
]

for instance in new_instances:
    prediction = predict(decision_tree, instance)
    print(f"Prediction for {instance}: {prediction}")