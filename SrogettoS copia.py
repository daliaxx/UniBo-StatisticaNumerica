import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve


"Step 1: Scelta del dataset"

# Caricare il dataset
data = pd.read_csv('dataenergy.csv')
print(f"Dimensione del dataset iniziale: {data.shape[0]} righe")
# Identificare le colonne numeriche nel dataset
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()


"Step 2: Pre-Processing"

# Rimuovere le colonne inutili
data = data.drop(columns=["Longitude", "Latitude", "Land Area(Km2)", "gdp_per_capita", "gdp_growth",
                          "Density(P/Km2)","Financial flows to developing countries (US $)",
                          "Energy intensity level of primary energy (MJ/$2017 PPP GDP)", "Year",
                          "Primary energy consumption per capita (kWh/person)"])
print(data.info())
print(f"Dimensione del dataset iniziale: {data.shape[0]} righe")
# Controllare la presenza e rimuovere i valori NaN
print(data.isnull().sum())
data = data.dropna()

# Verificare se ci sono ancora NaN
print(data.isnull().sum())

# Descrivere il dataset per capire i range delle variabili numeriche
print(data.describe())
print(f"Dimensione del dataset iniziale: {data.shape[0]} righe")


"Step 3: Esplorazione dei Dati (EDA)"

# Variabili numeriche del dataset
numeric_cols = data.select_dtypes(include=[np.number]).columns
print(f"Variabili numeriche: {numeric_cols}")

# Creare un Boxplot per ogni variabile numerica
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Visualizzare la distribuzione delle variabili numeriche
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Creazione della matrice di correlazione
corr_matrix = data[numeric_cols].corr()
print("Matrice di correlazione di Pearson:\n", corr_matrix)

# Visualizzare la matrice di correlazione
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice di Correlazione')
plt.show()

# Analisi Univariata
data[numeric_cols].hist(bins=30, figsize=(20, 15), color='blue', edgecolor='black')
plt.suptitle('Istogrammi delle variabili numeriche')
plt.show()


# Analisi Bivariata 
# Calcolo della correlazione tra le due variabili
correlation = data["Low-carbon electricity (% electricity)"].corr(data["Renewables (% equivalent primary energy)"])
print(f"Correlazione più alta: {correlation}")
# Scatter plot per vedere la relazione tra "Low-carbon electricity (% electricity) e Renewables (% equivalent primary energy)"
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data["Renewables (% equivalent primary energy)"], y=data["Low-carbon electricity (% electricity)"])
plt.title('Scatter plot tra Low-carbon electricity (% electricity) e Renewables (% equivalent primary energy)')
plt.xlabel("Low-carbon electricity (% electricity)")
plt.ylabel("Renewables (% equivalent primary energy)")
plt.show()

# Analisi Multivariata
plt.figure(figsize=(10, 6))
sns.pairplot(data[['Electricity from fossil fuels (TWh)', 'Low-carbon electricity (% electricity)',
                   'Electricity from renewables (TWh)', 'Value_co2_emissions_kt_by_country']])
plt.suptitle('Pair Plot delle Variabili Numeriche')
plt.show()

'Step 4: Splitting'

# Definire le feature (X) e il target (y)
target = 'Renewable energy share in the total final energy consumption (%)'
X = data[numeric_cols].drop(columns=[target])
y = data[target]

# Verifica delle dimensioni del dataset prima dello splitting
print(f"Dimensione del dataset: {X.shape[0]} righe")

# Dividiamo il dataset in 70% training, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")


'Step 5: Regressione'

#  identifichiamo le coppie dalla matrice di correlazione
coppie_correlazione_uno = [('Low-carbon electricity (% electricity)', 'Renewables (% equivalent primary energy)')]  
coppie_correlazione_due = [('Access to clean fuels for cooking','Access to electricity (% of population)')] 

def esegui_regressione(data, var1, var2):
    # Creazione degli array per X e y
    X_reg = data[[var1]].values.reshape(-1, 1)
    y_reg = data[var2].values

    # Creazione e addestramento del modello di regressione lineare
    modello_regressione = LinearRegression()
    modello_regressione.fit(X_reg, y_reg)

    # Predizioni
    y_pred = modello_regressione.predict(X_reg)

    # Coefficienti della regressione e calcolo delle metriche
    coefficiente = modello_regressione.coef_[0]
    intercetta = modello_regressione.intercept_
    r2 = modello_regressione.score(X_reg, y_reg)
    mse = mean_squared_error(y_reg, y_pred)

    # Analisi dei residui
    residui = y_reg - y_pred
    (mu, sigma) = stats.norm.fit(residui)

    # Visualizzazione dei risultati
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reg, y_reg, color='green', label='Dati')
    plt.plot(X_reg, y_pred, color='purple', linewidth=2, label='Retta di regressione')
    plt.title(f'Regressione Lineare tra {var1} e {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residui, kde=True, stat="density", color='purple')
    plt.title(f'Analisi di normalità dei residui per {var1} vs {var2}')
    plt.xlabel('Residui')
    plt.ylabel('Densità')
    plt.show()

    print(f"Regressione tra {var1} e {var2}:")
    print(f"Coefficiente: {coefficiente}")
    print(f"Intercetta: {intercetta}")
    print(f"R^2: {r2}")
    print(f"MSE: {mse}")
    print("\n")

# Chiamata della funzione per la prima coppia
esegui_regressione(data, coppie_correlazione_uno[0][0], coppie_correlazione_uno[0][1])

# Chiamata della funzione per la seconda coppia
esegui_regressione(data, coppie_correlazione_due[0][0], coppie_correlazione_due[0][1])


'Step 6: Addestramento del Modello'

# Trasformazione del target in binario
data['target'] = (data['Renewable energy share in the total final energy consumption (%)'] > data['Renewable energy share in the total final energy consumption (%)'].mean()).astype(int)

# Utilizza i set di training, validation e test già definiti
X = data[['Low-carbon electricity (% electricity)', 'Renewables (% equivalent primary energy)']]
y = data['target']

# Suddividi i dati in training, validation e test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Addestramento del modello di Regressione Logistica sul training set
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predizioni sul validation set
y_val_pred_log = logistic_model.predict(X_val)

# Valutazione del modello di Regressione Logistica sul validation set
print()
print("Valutazione del modello di Regressione Logistica (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_log)}")
print(classification_report(y_val, y_val_pred_log, zero_division=0))

# Matrice di Confusione per Regressione Logistica sul validation set
conf_matrix_log_val = confusion_matrix(y_val, y_val_pred_log)
cm_display_log_val = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_log_val, display_labels=[False, True])
cm_display_log_val.plot()
plt.title('Matrice di Confusione - Regressione Logistica (Validation Set)')
plt.show()

# Addestramento del modello SVM sul training set
modello_svm = SVC(kernel='linear', C=1)
modello_svm.fit(X_train, y_train)

# Predizioni sul validation set
y_val_pred_svm = modello_svm.predict(X_val)

# Valutazione del modello SVM sul validation set
print()
print("Valutazione del modello SVM (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_svm)}")
print(classification_report(y_val, y_val_pred_svm))

# Matrice di Confusione per SVM sul validation set
conf_matrix_svm_val = confusion_matrix(y_val, y_val_pred_svm)
cm_display_svm_val = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_val, display_labels=[False, True])
cm_display_svm_val.plot()
plt.title('Matrice di Confusione - SVM (Validation Set)')
plt.show()

# Visualizzazione delle previsioni di classificazione per la Regressione Logistica
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_val.iloc[:, 0], y=X_val.iloc[:, 1], hue=y_val_pred_log, palette='bright', marker='o', alpha=0.7)
plt.title('Classificazione con Regressione Logistica')
plt.xlabel('Low-carbon electricity (% electricity)')
plt.ylabel('Renewables (% equivalent primary energy)')
plt.show()

# Visualizzazione delle previsioni di classificazione per SVM
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_val.iloc[:, 0], y=X_val.iloc[:, 1], hue=y_val_pred_svm, palette='bright', marker='s', alpha=0.7)
plt.title('Classificazione con SVM')
plt.xlabel('Low-carbon electricity (% electricity)')
plt.ylabel('Renewables (% equivalent primary energy)')
plt.show()


'Step 7: Hyperparameter Tuning'

# Definiamo la griglia di parametri per SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Configura la Grid Search con la validazione incrociata
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)

# Esegui la Grid Search sul training set
grid_search.fit(X_train, y_train)

# Mostra i migliori iperparametri trovati
print(f"I migliori iperparametri sono: {grid_search.best_params_}")

# Valuta il modello ottimale sul validation set
best_svm_model = grid_search.best_estimator_
y_val_pred_best_svm = best_svm_model.predict(X_val)

print()
print("Valutazione del modello ottimizzato SVM (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_best_svm)}")
print(classification_report(y_val, y_val_pred_best_svm))

# Matrice di Confusione per SVM ottimizzato sul validation set
conf_matrix_best_svm_val = metrics.confusion_matrix(y_val, y_val_pred_best_svm)
cm_display_best_svm_val = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best_svm_val, display_labels=[False, True])
cm_display_best_svm_val.plot()
plt.title('Matrice di Confusione - SVM Ottimizzato (Validation Set)')
plt.show()


'Step 8: Valutazione della Performance'

# Valutazione finale sul test set
y_test_pred_best_svm = best_svm_model.predict(X_test)
print()
print("Valutazione finale del modello ottimizzato SVM sul test set:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_best_svm)}")
print(classification_report(y_test, y_test_pred_best_svm))

# Matrice di Confusione finale per SVM ottimizzato sul test set
conf_matrix_test_best_svm = metrics.confusion_matrix(y_test, y_test_pred_best_svm)
cm_display_test_best_svm = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test_best_svm, display_labels=[False, True])
cm_display_test_best_svm.plot()
plt.title('Matrice di Confusione - SVM Ottimizzato (Test Set)')
plt.show()

'Step 9: Studio statistico sui risultati della valutazione'

# Ripetiamo l'addestramento e il testing k volte (con k ≥ 10) per valutare la robustezza del modello
k = 10
accuratezze = []

# Usare i migliori parametri trovati nel punto 7 per il modello
best_params_cv = grid_search.best_params_

for i in range(k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    modello_svm = SVC(**best_params_cv)
    modello_svm.fit(X_train, y_train)
    predizioni = modello_svm.predict(X_test)
    acc = accuracy_score(y_test, predizioni)
    accuratezze.append(acc)
    
# Analisi statistica descrittiva delle metriche di errore
accuratezze = np.array(accuratezze)
media_acc = np.mean(accuratezze)
std_acc = np.std(accuratezze)
intervallo_confidenza = stats.norm.interval(0.95, loc=media_acc, scale=std_acc/np.sqrt(k))

print()
print(f"Media Accuracy: {media_acc}")
print(f"Deviazione Standard: {std_acc}")
print(f"Intervallo di Confidenza al 95%: {intervallo_confidenza}")

# Istogramma e boxplot delle accuratezze
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(accuratezze, bins=10, edgecolor='black')
plt.title('Distribuzione delle Accuratezze')
plt.xlabel('Accuracy')
plt.ylabel('Frequenza')

plt.subplot(1, 2, 2)
plt.boxplot(accuratezze, vert=False)
plt.title('Boxplot delle Accuratezze')
plt.xlabel('Accuracy')
plt.xlim(0.5, 1.0) 

plt.tight_layout()
plt.show()


# Funzione per tracciare le learning curves
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="hotpink")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="hotpink",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

# Creare il modello di regressione lineare
model = LinearRegression()

# Tracciare le learning curves
plot_learning_curve(model, "Learning Curves (Linear Regression)", X_train, y_train, cv=5, n_jobs=-1)
plt.show()