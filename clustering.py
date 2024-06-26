from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def purity_score(y_true, y_pred):
    """
    Calcola il purity score dato le etichette vere e quelle predette.
    
    Parametri:
    y_true (array-like): Etichette vere.
    y_pred (array-like): Etichette predette.
    
    Ritorna:
    float: Il purity score calcolato.
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def perform_clustering(file_path):
    """
    Esegue il clustering su un dataset specificato e calcola il purity score.

    Parametri:
    file_path (str): Il percorso del file CSV contenente il dataset.

    Il dataset deve contenere una colonna 'label' per le etichette e una colonna 'filename' che verrà rimossa durante il preprocessing.
    """
    
    # Caricamento del dataset
    df = pd.read_csv(file_path)

    # Rimozione delle colonne non necessarie
    columns = df.columns.to_list()
    columns.remove('label')
    columns.remove('filename')

    # Separazione delle feature e delle etichette
    X = df[columns]
    Y = df['label']

    # Normalizzazione delle feature
    minmax = MinMaxScaler()
    scaledX = pd.DataFrame(minmax.fit_transform(X), columns=X.columns.to_list())

    # Determinazione del numero di cluster
    num_clusters = len(Y.unique())
    if num_clusters <= 0:
        raise ValueError("Il numero di cluster deve essere maggiore di 0")

    # Esecuzione del clustering
    kmeans = KMeans(n_clusters=num_clusters, max_iter=3000, verbose=True)
    kmeans.fit(scaledX)
    clusters = kmeans.labels_

    # Calcolo e stampa del purity score
    purity = purity_score(Y, clusters)
    print(f"Purity Score: {purity:.4f}")

# Esegui il clustering
perform_clustering('dataset/data.csv')
