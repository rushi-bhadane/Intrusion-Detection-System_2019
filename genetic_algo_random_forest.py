import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deap import creator, base, tools, algorithms
import sys


def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """
    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]


        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # apply classification algorithm
        clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
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

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
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
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen


if __name__ == '__main__':
    # get dataframe path, population number and generation number from command-line argument
    df = pd.read_csv('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',header = 0)
      # Separating out the features
    X = df.iloc[:, 0:-1]
    #x = x.drop(df.columns[[0,1,3]], axis=1)
    # Separating out the target
    y = df.iloc[:,-1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    
# Standardizing the features
    
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    
    
    '''from sklearn.decomposition import PCA
    pca = PCA(n_components=78)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    
    dataframePath = pd.concat([principalDf, y], axis = 1)'''
    n_pop = 2
    n_gen = 2
    # read dataframe from csv
    #df = dataframePath.copy()

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]

    individual = [1 for i in range(len(X.columns))]
    accuracy_all_features = str(getFitness(individual, X, y))
    print("Accuracy with all features: \t" + accuracy_all_features + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X, y, n_pop, n_gen)
    for individual in hof:
        print(individual.fitness.values)
    
# select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')

    # read dataframe from csv one more time
    #df = pd.read_csv('Friday.csv', header = 0, sep = ',')


    # with feature subset
    X = df[header]

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
    print("recall", TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("prec", PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    F1 = 2 * (TPR * PPV) / (TPR + PPV)
    print("f1", F1)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")