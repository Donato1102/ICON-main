import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import seaborn as sns

def conf_mat(test, pred, class_names, title):
    """
    Plotta la matrice di confusione non normalizzata.
    
    Parametri:
    test (array-like): Etichette reali.
    pred (array-like): Etichette predette.
    class_names (list): Nomi delle classi.
    title (str): Titolo del grafico.
    """
    try:
        # Generazione della matrice di confusione
        mat = metrics.confusion_matrix(test, pred)
        
        # Configurazione del plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(mat,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot=True,
                    fmt='d')
        
        # Aggiunta di titolo e etichette
        plt.title(title)
        plt.ylabel('Etichetta Reale')
        plt.xlabel('Etichetta Predetta')
        
        # Mostra il grafico
        plt.show()
        
    except Exception as e:
        print(f"Errore durante la creazione della matrice di confusione: {e}")

def evaluate_model(test, pred):
    """
    Valuta le performance del modello e stampa metriche comuni.
    
    Parametri:
    test (array-like): Etichette reali.
    pred (array-like): Etichette predette.
    """
    try:
        accuracy = accuracy_score(test, pred)
        precision = precision_score(test, pred, average='macro')
        recall = recall_score(test, pred, average='macro')
        f1 = f1_score(test, pred, average='macro')
        
        print("Metriche di valutazione del modello:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n")
        
    except Exception as e:
        print(f"Errore durante la valutazione del modello: {e}")

