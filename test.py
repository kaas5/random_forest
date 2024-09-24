import csv
import numpy as np

def class_to_token(str):
    #print(str)
    if str == "Iris-setosa": return 0
    if str == "Iris-virginica": return 1
    if str == "Iris-versicolor": return 2
    return None

def read_dataset():
    res = []
    with open('IRIS.xls', newline='') as csvfile:
        first = True
        for row in csv.reader(csvfile, delimiter=','):
            if first: 
                first = False
            else:
                row[4] = class_to_token(row[4])
                row = [float(r) for r in row]
                res.append(row)
    return np.array(res)[:,0:4], np.array(res)[:,4]

data, labels = read_dataset()

def gini_factor(labels):
    # hoe lager hoe beter
    total = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / total

    # https://victorzhou.com/blog/gini-impurity/
    # ^ deze dan? 
    gini = np.sum(probabilities - np.power(probabilities, 2))
    return gini

def select_random_subset(data, labels, n):
    indices = np.random.choice(data.shape[0], size=n, replace=False)
    return data[indices], labels[indices]

def split_data(data, labels, attribute, split_point):
    data_compare = data[:,attribute]
    labels_left = labels[data_compare >= split_point]
    labels_right = labels[data_compare < split_point]
    data_left = data[data_compare >= split_point,:]
    data_right = data[data_compare < split_point,:]
    #print(data_left, labels_left, data_right, labels_right)
    return data_left, labels_left, data_right, labels_right

def attribute_loop(data, labels, attribute):
    data = data[:,attribute]
    gini_values = []
    split_values = []

    for split_point in data:
        subset_left = labels[data >= split_point]
        subset_right = labels[data < split_point]

        gini_left = gini_factor(subset_left)
        gini_right = gini_factor(subset_right)

        gini = (subset_left.shape[0] / data.shape[0]) * gini_left + (subset_right.shape[0] / data.shape[0]) * gini_right

        gini_values.append(gini)
        split_values.append(split_point)
        #print(gini, split_point)
        
    best_gini_value_index = np.argmin(np.array(gini_values))
    best_split_value = split_values[best_gini_value_index]
    best_gini_value = gini_values[best_gini_value_index]

    gini_node = gini_factor(labels)
    gini_gain = gini_node - best_gini_value

    return gini_gain, best_split_value 

def select_best_attribute(sub_data, sub_labels):
    best_gain = 0.0
    best_attr = None
    best_split = None

    for i in range(0, data.shape[1]):
        gini_gain, best_split_value = attribute_loop(sub_data, sub_labels, i)
        #print(gini_gain, best_split_value)

        if gini_gain > best_gain:
            best_gain = gini_gain
            best_attr = i
            best_split = best_split_value

    print('best gain', best_gain)
    return best_attr, best_split

class Tree:

    def __init__(self):
        # structure of a node in the tree (== key-value in the dictionary)
        # attribute, split, node_id_l, node_id_r
        self.tree_dict = {}
    
    def set_attribute_split(self, data, labels, node_id):
        if data.shape[0] != 0:
            attribute, split = select_best_attribute(data, labels)
            if attribute is None:
                # TODO pak de label van deze leaf op een andere manier?
                classes, counts = np.unique(labels, return_counts=True)
                
                self.tree_dict[node_id] = classes[np.argmax(counts)]
                #self.tree_dict[node_id] = labels[0]
                return

            dl, ll, dr, lr = split_data(data, labels, attribute, split)
            
            node_id_l = self.find_child_id(node_id)
            node_id_r = self.find_child_id(node_id)
            self.tree_dict[node_id] = attribute, split, node_id_l, node_id_r

            self.set_attribute_split(dl, ll, node_id_l)
            self.set_attribute_split(dr, lr, node_id_r)

    def start_tree(self, data, labels):
        parent_id = 0
        self.set_attribute_split(data, labels, parent_id)

    def start_traverse_tree(self, datapoint):
        node_id = 0
        while type(self.tree_dict[node_id]) is not np.float64:
            attribute, split, node_id_l, node_id_r = self.tree_dict[node_id]
            if datapoint[attribute] >= split:
                node_id = node_id_l
            else:
                node_id = node_id_r
        return self.tree_dict[node_id]

    def find_child_id(self, parent_id):
        while parent_id in self.tree_dict.keys():
            parent_id += 1
        self.tree_dict[parent_id] = None
        return parent_id
    
    def print_all_nodes(self):
        for id in self.tree_dict.keys():
            print(id, self.tree_dict[id])

sub_data, sub_labels = select_random_subset(data, labels, 50)

#print(attribute_loop(sub_data, sub_labels, 0))
#print(select_best_attribute(sub_data, sub_labels))

tree = Tree()
tree.start_tree(sub_data,sub_labels)

print(tree.start_traverse_tree(sub_data[0]))
print(sub_labels[0])

tree.print_all_nodes()