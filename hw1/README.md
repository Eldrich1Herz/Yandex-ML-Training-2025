* Attention: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/01_attention.ipynb)

* Classification: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/02_hw_fmnist_classification.ipynb)

# ML Course: Attention Mechanism & FashionMNIST Classification

Репозиторий содержит решения двух задач из курса машинного обучения Girafe AI (hw01_classification_and_attention).

## Структура проекта

```
├── ML_Attention_implementation.py    # Реализация механизмов внимания
└── ML_Separation_(FashionMNIST_classification).py  # Классификация FashionMNIST
```

---

## 1. Attention Implementation (`ML_Attention_implementation.py`)

Реализация двух типов механизмов внимания на NumPy:

### Функции

| Функция | Описание |
|---------|----------|
| `softmax(vector)` | Вычисление softmax для матрицы с численной стабилизацией |
| `multiplicative_attention(...)` | Multiplicative (Dot-Product) Attention |
| `additive_attention(...)` | Additive (Bahdanau) Attention |

### Multiplicative Attention

Формула:
```
e_i = s^T * W_mult * h_i
α_i = softmax(e_i)
context = Σ_i(α_i * h_i)
```

Где:
- `s` — состояние декодера
- `h_i` — состояния энкодера
- `W_mult` — матрица весов

### Additive Attention

Формула:
```
e_i = v^T * tanh(W_enc * h_i + W_dec * s)
α_i = softmax(e_i)
context = Σ_i(α_i * h_i)
```

Где:
- `W_enc`, `W_dec` — матрицы проекции
- `v` — вектор весов для получения скалярного значения

### Параметры функций

| Параметр | Размерность | Описание |
|----------|-------------|----------|
| `decoder_hidden_state` | `(n_features_dec, 1)` | Состояние декодера |
| `encoder_hidden_states` | `(n_features_enc, n_states)` | Состояния энкодера |
| `W_mult` | `(n_features_dec, n_features_enc)` | Матрица весов для multiplicative |
| `v_add` | `(n_features_int, 1)` | Вектор весов для additive |
| `W_add_enc` | `(n_features_int, n_features_enc)` | Матрица для энкодера |
| `W_add_dec` | `(n_features_int, n_features_dec)` | Матрица для декодера |

---

## 2. FashionMNIST Classification (`ML_Separation_(FashionMNIST_classification).py`)

Задача классификации изображений на датасете FashionMNIST.

### Архитектура модели

```
FashionMNISTModel (CNN):
├── conv1: Conv2d(1 → 32, kernel_size=3, padding=1)
├── max_pool2d: 2x2
├── conv2: Conv2d(32 → 64, kernel_size=3, padding=1)
├── max_pool2d: 2x2
├── fc1: Linear(64*7*7 → 128)
├── dropout: 0.25
└── fc2: Linear(128 → 10)
```

### Гиперпараметры

| Параметр | Значение |
|----------|----------|
| Оптимизатор | Adam |
| Learning rate | 0.001 |
| Функция потерь | CrossEntropyLoss |
| Размер батча | 32 |
| Эпохи | 10 |

### Требования к точности

| Датасет | Порог |
|---------|-------|
| Train Accuracy | ≥ 0.905 |
| Test Accuracy | ≥ 0.885 |

### Генерация submission

При наличии файла `hw_fmnist_data_dict.npy` скрипт генерирует `submission_dict_fmnist_task_1.json` с предсказаниями для контеста.

---

## Требования

```
numpy >= 1.20.0
torch >= 1.9.0
torchvision >= 0.10.0
matplotlib >= 3.4.0
```

## Запуск

```bash
# Attention implementation — импортируйте функции в ноутбук
# FashionMNIST classification
python "ML_Separation_(FashionMNIST_classification).py"
```

> **Примечание для Windows:** В скрипте учтены особенности многопроцессорности (`num_workers=0`) и добавлена проверка `if __name__ == '__main__'`.

## Ссылки

- [Attention ноутбук](https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/01_attention.ipynb)
- [FashionMNIST задача](https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/02_hw_fmnist_classification.ipynb)
- [Курс Girafe AI](https://github.com/girafe-ai/ml-course)

