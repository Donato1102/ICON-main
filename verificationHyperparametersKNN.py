import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def iperparametri_knn(data):
    """
    Esegue una ricerca a griglia per trovare i migliori iperparametri per un KNeighborsClassifier.
    
    Parametri:
    data (DataFrame): Il dataset contenente le feature e le etichette. Deve contenere una colonna 'label' per le etichette
                      e una colonna 'filename' che verrà rimossa.
    
    Ritorna:
    KNeighborsClassifier: Il modello addestrato con i migliori iperparametri trovati.
    """
    # Rimozione delle colonne non necessarie
    datas = data.drop(columns=['filename'])
    
    # Separazione delle feature e delle etichette
    X = np.array(datas.drop(columns=['label']))
    y = np.array(datas['label'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Definizione della griglia di iperparametri
    parameters = {
        'n_neighbors': [1, 4, 20],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [5, 10, 20, 30]
    }

    # Creazione del modello KNeighborsClassifier
    model = KNeighborsClassifier()
    
    # Configurazione di GridSearchCV
    grid_search = GridSearchCV(model, param_grid=parameters, cv=3, n_jobs=4, verbose=2)
    
    # Esecuzione della ricerca a griglia
    grid_search.fit(X_train, Y_train)
    
    # Stampa dei migliori parametri trovati
    print("Best parameters:", grid_search.best_params_)
    
    # Ritorna il modello con i migliori iperparametri
    return grid_search.best_estimator_
