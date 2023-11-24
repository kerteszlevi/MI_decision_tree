import numpy as np #(működik a Moodle-ben is)


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    # összes elem
    total = n_cat1 + n_cat2
    # arányok
    r_cat1 = n_cat1/total
    r_cat2 = n_cat2/total

    # entrópia kiszámítása a két csoportra
    entropy = r_cat1 * np.log2(r_cat1) + r_cat2 * np.log2(r_cat2)
    entropy = -entropy
    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    # inicializálás
    best_separation_feature, best_separation_value, best_entropy = 0, 0, 0

    # végig iterálunk az összes adattagon
    for feature_index in range(features.shape[1]):
        # ennek az adattagnak az összes értéke rendezve
        unique = np.unique(features[:, feature_index])

        # az összes rekordon végig iterálunk
        for value in unique:

            # a címkéket két csoportra osztjuk a jelenlegi érték alapján numpy csoportművelettel
            group1 = labels[features[:, feature_index] <= value]
            group2 = labels[features[:, feature_index] > value]

            # kiszámoljuk mindkét csoport entrópiáját
            n_labels = len(labels)
            n_g1 = len(group1)
            n_g2 = len(group2)

            entropy = ((n_g1/n_labels) * get_entropy(np.sum(group1==0), np.sum(group1==1)) +
                       (n_g2/n_labels) * get_entropy(np.sum(group2==0), np.sum(group2==1)))

            # ha jobb mint az eddigi, akkor frissítjük a legjobbakat
            if entropy < best_entropy:
                best_entropy = entropy
                best_separation_feature = feature_index
                best_separation_value = value

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    return 0

if __name__ == "__main__":
    main()
