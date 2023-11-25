import numpy as np #(működik a Moodle-ben is)
import csv

#adattag = feature
#címke, döntési érték = label

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    entropy = 0
    # összes elem
    total = n_cat1 + n_cat2
    # arányok
    if total > 0:
        r_cat1 = n_cat1/total
        r_cat2 = n_cat2/total
    else:
        # ha 0 elemű a csoport végtelennel térünk vissza
        return float('inf')

    # entrópia kiszámítása a két csoportra

    if r_cat1 > 0:
        sub_entropy1 = r_cat1 * np.log2(r_cat1)
        entropy -=sub_entropy1

    if r_cat2 > 0:
        sub_entropy2 = r_cat2 * np.log2(r_cat2)
        entropy -=sub_entropy2


    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    # inicializálás
    best_separation_feature, best_separation_value, best_entropy = 0, 0, float('inf')

    # végig iterálunk az összes adattagon
    for feature_index in range(features.shape[1]):
        # ennek az adattagnak az összes értéke rendezve
        unique = np.unique(features[:, feature_index])

        # az összes rekordon végig iterálunk
        for value in unique:

            # az adattagokat két csoportra osztjuk a jelenlegi érték alapján numpy csoportművelettel
            # az egyenlő értékek a kisebb csoportba kerülnek a feladatkiírás alapján
            group1 = labels[features[:, feature_index] <= value]
            group2 = labels[features[:, feature_index] > value]

            # kiszámoljuk mindkét csoport entrópiáját
            n_labels = len(labels)
            n_g1 = len(group1)
            n_g2 = len(group2)

            entropy = ((n_g1/n_labels) * get_entropy(np.sum(group1==0), np.sum(group1==1)) +
                       (n_g2/n_labels) * get_entropy(np.sum(group2==0), np.sum(group2==1)))

            # ha jobb mint az eddigi, akkor frissítjük a legjobb entrópiát, szeparáció értékét és adattagját
            if entropy < best_entropy:
                best_entropy = entropy
                best_separation_feature = feature_index
                best_separation_value = value

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################


# csv fájl beolvasása, majd visszatérés egy numpy tömbbel
def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data).astype(float)
    return data


# csv fájlba írás, a numpy float-ok int-re konvertálása, majd annak a kiíratása
def save_csv(path, data):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        for element in data:
            writer.writerow([element.astype(int)])


# fa tanítására szolgáló rekurzív függvény
def train_tree(data):

    # az adattagok és a döntések szétválasztása
    features = data[:, :-1]
    labels = data[:, -1]

    # a legjobb adattag és a hozzá tartozó legjobb érték megkeresése a szétválasztáshoz
    best_separation_feature, best_separation_value = get_best_separation(features, labels)

    # az adatok egy kisebb és egy nagyobb csoportra bontása a megtalált értékek alapján
    group1 = data[data[:, best_separation_feature] <= best_separation_value]
    group2 = data[data[:, best_separation_feature] > best_separation_value]

    # dictionary, amely egy csomópont a fában, ez eleinte a root node,
    # majd később a csoportok helyén további node-ok jelennek meg
    node = {"feature": best_separation_feature, "value": best_separation_value, "groups": [group1, group2]}

    # végig iterálunk a két csoporton
    for i, group in enumerate(node["groups"]):
        #ha az adott csoportban csak egy fajta döntési érték található, tehát az 0 és 1-esek számának entrópiája 0
        #akkor a csoportot az adott csomópontban a döntés értékével helyettesítjük
        if get_entropy(np.sum(group[:, -1] == 0), np.sum(group[:, -1] == 1)) == 0:
            node["groups"][i] = group[0, -1]
            # amennyiben a csoport entrópiája nem 0, azon a csomóponton tovább folytatjuk a tanítást
            # tehát rekurzívan tovább hívjuk rá a train_tree függvényt
        else:
            node["groups"][i] = train_tree(group)

    return node


# a döntési fa rekurzív kiíratása
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('{}[Feature{} <= {}]'.format(depth*' ', node['feature'], node['value']))
        print_tree(node['groups'][0], depth+1)
        print_tree(node['groups'][1], depth+1)
    else:
        print('{}[{}]'.format(depth*' ', node))


# a döntési fa kiértékelése
def evaluate_tree(tree, data_test):
    #predikciók listája
    evaluations = []
    for data in data_test:
        evaluations.append(evaluate_node(tree, data))
    return evaluations


# a csomópont rekurzív kiértékelése
def evaluate_node(node, data ):
    # ha a csomópont nem dictionary, tehát levél(az értéke egy szám), visszatérünk vele
    if not isinstance(node, dict):
        return node
    # ha az adott csomóponthoz tartozó adattaghoz tartozó értéknél a kapott adat értéke kisebb vagy egyenlő,
    # akkor az első csoporton hívjuk tovább a függvényt, ellenkező esetben a másodikon.
    if data[node["feature"]] <= node["value"]:
        return evaluate_node(node["groups"][0], data)
    else:
        return evaluate_node(node["groups"][1], data)


def main():
    # a fa tanítása:
    data_train = read_csv('train.csv')
    tree = train_tree(data_train)
    #print_tree(tree) #valami komolyabbra számítottam megmondom őszintén...

    # a fa kiértékelése:
    data_test = read_csv('test.csv')
    data_results = evaluate_tree(tree, data_test)
    save_csv('results.csv', data_results)

    return 0


if __name__ == "__main__":
    main()
