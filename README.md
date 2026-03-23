# MLOps Project 1

Projekt wykonany w ramach zajęć z Narzędzi Uczenia Maszynowego / MLOps.

## Opis projektu

Celem projektu było zbudowanie kompletnego pipeline'u treningowego z użyciem:

- PyTorch
- PyTorch Lightning
- MLflow do monitorowania eksperymentów
- Optuna do optymalizacji hiperparametrów

Projekt rozwiązuje problem **klasyfikacji binarnej danych tabelarycznych** na zbiorze **Breast Cancer Wisconsin Dataset**.

## Dataset

W projekcie wykorzystano dataset `load_breast_cancer()` dostępny w bibliotece `scikit-learn`.

Charakterystyka danych:
- typ danych: tabelaryczne / numeryczne
- liczba cech: 30
- zadanie: klasyfikacja binarna

## Model

Zastosowany model to prosty **MLP (Multilayer Perceptron)** dla danych tabelarycznych.

Architektura modelu:
- warstwa wejściowa: 30 cech
- dwie warstwy ukryte
- aktywacja ReLU
- Dropout
- warstwa wyjściowa: 2 neurony

Funkcja straty:
- `CrossEntropyLoss`

Optymalizator:
- `Adam`

## Struktura projektu

```text
MLOps_Project1/
│
├── mlruns/
├── src/
│   ├── config.py
│   ├── data_module.py
│   ├── dataset.py
│   ├── lightning_module.py
│   ├── model.py
│   ├── train.py
│   ├── tune_optuna.py
│   └── utils.py
│
├── README.md
└── requirements.txt
````

## Najważniejsze pliki

* `dataset.py` – własna klasa `TabularDataset`
* `data_module.py` – `BreastCancerDataModule` w PyTorch Lightning
* `model.py` – architektura modelu MLP
* `lightning_module.py` – `BreastCancerLightningModule`
* `train.py` – trening modelu bazowego
* `tune_optuna.py` – strojenie hiperparametrów Optuną
* `config.py` – konfiguracja projektu

## Przygotowanie danych

W projekcie wykonano:

* podział danych na train / validation / test
* standaryzację cech przy użyciu `StandardScaler`
* dopasowanie scalera tylko na zbiorze treningowym
* transformację zbioru walidacyjnego i testowego tym samym scalerem

Dzięki temu uniknięto **data leakage**.

## Monitoring

Do monitorowania eksperymentów wykorzystano **MLflow** zintegrowany z PyTorch Lightning.

Logowane metryki:

* `train_loss`
* `val_loss`
* `train_acc`
* `val_acc`
* `test_loss`
* `test_acc`

Monitoring pozwolił sprawdzić, czy model rzeczywiście się uczy oraz porównać wyniki między uruchomieniami.

## Optymalizacja hiperparametrów

Do strojenia hiperparametrów wykorzystano **Optuna**.

Strojone hiperparametry:

* `learning_rate`
* `hidden_dim`
* `dropout`
* `batch_size`

Cel optymalizacji:

* maksymalizacja `val_acc`

Każdy trial Optuny był logowany jako osobny run w MLflow.

## Uruchamianie projektu

Trening modelu bazowego:

```bash
python src/train.py
```

Optymalizacja hiperparametrów:

```bash
python src/tune_optuna.py
```

Uruchomienie MLflow UI:

```bash
mlflow ui
```

Następnie należy wejść w przeglądarce na adres podany w terminalu, zwykle:

```text
http://127.0.0.1:5000
```

W MLflow należy wybrać:

* **Model training**
* eksperyment `mlops_project1_breast_cancer`

## Wyniki

Model bazowy osiągnął bardzo dobre wyniki:

* validation accuracy około `0.978`
* test accuracy około `0.956`

Optuna znalazła kilka konfiguracji o podobnie wysokiej jakości. Najlepszy trial osiągnął:

* `best val_acc = 0.9780`

W praktyce tuning nie poprawił znacząco wyniku względem baseline, ale pozwolił porównać różne konfiguracje i potwierdzić stabilność rozwiązania.

## Napotkane problemy

Podczas realizacji projektu pojawiły się m.in. następujące problemy:

* zrozumienie struktury wymaganej przez PyTorch Lightning
* poprawne rozdzielenie Dataset / DataModule / LightningModule
* poprawne skalowanie danych po podziale zbioru
* konfiguracja i obsługa MLflow
* integracja Optuny z Lightning i MLflow

## Technologie

* Python
* PyTorch
* PyTorch Lightning
* scikit-learn
* NumPy
* Pandas
* MLflow
* Optuna

## Autorzy

Paweł Kierkosz 155995, Bartłomiej Rudowicz 155993

## Cel projektu

Celem projektu nie było zbudowanie bardzo złożonego modelu, ale przygotowanie poprawnie zorganizowanego i kompletnego workflow treningowego w stylu MLOps z użyciem PyTorch Lightning, monitorowania eksperymentów oraz hyperparameter optimization.

```
```
