import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import metrics
import pandas as pan
from verificationHyperparametersKNN import iperparametri_knn
from verificationHyperparametersRFC import iperparametri_rfc
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB


def plot_learning_curves(model, X, y, model_name):
    """
    Mostra la curva di apprendimento per il modello specificato.

    Parametri:
    model: Modello da valutare.
    X (array): Dati di input.
    y (array): Etichette.
    model_name (str): Nome del modello.
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy')

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()



def get_data(directory):
    """
    Carica il dataset e separa le feature dalle etichette.

    Parametri:
    directory (str): Percorso del file CSV contenente il dataset.

    Ritorna:
    tuple: Tuple contenente i dati di training e di test, le etichette, le classi uniche e il dataset originale.
    """
    data = pan.read_csv(directory)
    datas = data.drop(columns=['filename'])
    X_data = np.array(datas.drop(columns=['label']))
    y_data = np.array(data['label'])
    n_class = datas.drop_duplicates(subset='label')['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, Y_train, Y_test, n_class, data,X_data,y_data

# Caricamento del dataset e selezione delle feature
directory = "dataset/data.csv"
X_train, X_test, Y_train, Y_test, n_class, data,X_data,y_data = get_data(directory)

# KNN
classif_KNN = iperparametri_knn(data)
print("KNN:")
pred = classif_KNN.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix KNN')
plot_learning_curves(classif_KNN, X_data, y_data, "KNN")


# Naive Bayes
print("Naive Bayes:")
classif_NB = GaussianNB()
classif_NB.fit(X_train, Y_train)
pred = classif_NB.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix NB')

# Random Forest Classifier
classif_RFC = iperparametri_rfc(data)
print("Random Forest Classifier:")
pred = classif_RFC.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix RFC')
plot_learning_curves(classif_RFC, X_data, y_data, "RFC")

# Importanza delle feature
calibers = classif_RFC.feature_importances_
std = np.std([tree.feature_importances_ for tree in classif_RFC.estimators_], axis=0)
indexes = np.argsort(calibers)[::-1]

# Stampa l'importanza delle feature
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indexes[f], calibers[indexes[f]]))

# Visualizzazione delle importanze delle feature
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), calibers[indexes], color="r", yerr=std[indexes], align="center")
plt.xticks(range(X_train.shape[1]), indexes)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#prendo le primee 5 feature per importanza
k=5
data=data.iloc[:,indexes[:k+1]]
print("Feature selezionate + label: ",data.columns)
X_train, X_test, Y_train, Y_test, n_class, data,X_data,y_data = get_data(directory)

#RIPETO ESPERIMENTI CON LE FEATURE SELEZIONATE PER VEDERE COME VARIA

# KNN
classif_KNN = iperparametri_knn(data)
print("KNN:")
pred = classif_KNN.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix KNN')
# Mostra le curve di apprendimento per i modelli KNN e RFC
plot_learning_curves(classif_KNN, X_data, y_data, "KNN")


# Naive Bayes
print("Naive Bayes:")
classif_NB = GaussianNB()
classif_NB.fit(X_train, Y_train)
pred = classif_NB.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix NB')

# Random Forest Classifier
classif_RFC = iperparametri_rfc(data)
print("Random Forest Classifier:")
pred = classif_RFC.predict(X_test)
metrics.evaluate_model(Y_test, pred)
metrics.conf_mat(Y_test, pred, n_class, title='Confusion Matrix RFC')
plot_learning_curves(classif_RFC, X_data, y_data, "RFC")
