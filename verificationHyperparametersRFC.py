import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def iperparametri_rfc(data):
    """
    Esegue una ricerca a griglia per trovare i migliori iperparametri per un RandomForestClassifier.
    
    Parametri:
    data (DataFrame): Il dataset contenente le feature e le etichette. Deve contenere una colonna 'label' per le etichette
                      e una colonna 'filename' che verrà rimossa.
    
    Ritorna:
    RandomForestClassifier: Il modello addestrato con i migliori iperparametri trovati.
    """
    # Rimozione delle colonne non necessarie
    datas = data.drop(columns=['filename'])
    
    # Separazione delle feature e delle etichette
    X = np.array(datas.drop(columns=['label']))
    y = np.array(datas['label'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Definizione della griglia di iperparametri
    param_grid = {
        'bootstrap': [True],
        'max_depth': [8, 15, 25],
        'max_features': [1, 20, 28],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 4],
        'n_estimators': [100, 150, 200, 250]
    }
    
    # Creazione del modello RandomForestClassifier
    model = RandomForestClassifier()
    
    # Configurazione di GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=4, verbose=2)
    
    # Esecuzione della ricerca a griglia
    grid_search.fit(X_train, Y_train)
    
    # Stampa dei migliori parametri trovati
    print("Best parameters:", grid_search.best_params_)
    
    # Ritorna il modello con i migliori iperparametri
    return grid_search.best_estimator_


