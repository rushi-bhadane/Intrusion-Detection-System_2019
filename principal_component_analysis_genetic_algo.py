import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deap import creator, base, tools, algorithms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys

'''def train(data):
    data.dropna(inplace=True)
    print('dfi ', data.shape)
    y = data[' Label']
    print('y ', y.shape)
    
    X = data.drop(columns=[' Label'])
    print('X ', X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(acc)'''

def avg(l):
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    
    if(individual.count(0) != len(individual)):
        
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]


        
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)   
        X_subset = np.nan_to_num(X_subset)
        y = np.nan_to_num(y)
        
        clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)


        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    
    return hof


def bestIndividual(hof, X, y):
    
    maxAccurcy = 0.0
    for individual in hof:
        for i in range(0, len(individual.fitness.values)):
            if(individual.fitness.values[i] > maxAccurcy):
                maxAccurcy = individual.fitness.values[i]
                _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


def getArguments():

    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen


if __name__ == '__main__':
    
    df = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv',header = 0)
      # Separating out the features
    X = df.iloc[:, 0:-1]
    #x = x.drop(df.columns[[0,1,3]], axis=1)
    y = df.iloc[:,-1]
    no_of_columns = len(X.columns)
    
    '''from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)'''
    
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=78)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    # print pca components
    
    dataframePath = pd.concat([principalDf, y], axis = 1)
    n_pop = 2
    n_gen = 2
    dfi = dataframePath.copy()
    
    X = dfi.iloc[:, 0:-1]
    #x = x.drop(df.columns[[0,1,3]], axis=1)
    y = dfi.iloc[:,-1]
    cols = X.columns
        
    scaled_features = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, index=X.index, columns=X.columns)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    



    
    individual = [1 for i in range(no_of_columns)]
    accuracy_all_features = str(getFitness(individual, X_train, y_train))
    print("Accuracy with all features: \t" + accuracy_all_features + "\n")

    
    hof = geneticAlgorithm(X_train, y_train, n_pop, n_gen)
    for individual in hof:
        print(individual.fitness.values)


    accuracy, individual, header = bestIndividual(hof, X_train, y)
    getFitness(individual, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')

    


    dfi = np.nan_to_num(dfi)
    X = dfi[header]

    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    
    classify = classification_report(y_test, y_pred)
    
    FP = cm.sum(axis = 0) - np.diag(cm)
    FN = cm.sum(axis = 1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print(TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print(PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate 
    FDR = FP/(TP+FP)
    
    F1 = 2 * (TPR * PPV) / (TPR + PPV)
    print(F1)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")    