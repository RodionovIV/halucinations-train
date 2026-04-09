# Hallucination Detection Experiments

Эксперименты по бинарной классификации галлюцинаций LLM на датасете `final_features.csv`.

Целевая метрика: **PR-AUC** (Average Precision).

## Методы

Скрипты реализуют два семейства подходов из литературы:

| Метод | Статья | Идея |
|---|---|---|
| **SEPs** | Kossen et al., arXiv 2406.15927 (Oxford OATML, 2024) | Линейные зонды на скрытых состояниях слоёв LLM предсказывают семантическую энтропию |
| **MHAD** | Zhang et al., IJCAI 2025 | Отбор нейронов через linear probe + MLP на скрытых состояниях первого и последнего токена |

## Структура датасета

Файл `final_features.csv` содержит три типа признаков:

- **Статистики неопределённости** — `avg_logprob`, `avg_entropy`, `perplexity`, `avg_margin` и др. (~23 признака)
- **Скрытые состояния** — агрегаты (norm, mean, std, maxabs) по слоям 22, 24, 25 и позициям (first, last, minlogprob, mean) (~48 признаков)
- **Межслоевые разности** — L2 и косинусное расстояние между слоями (~16 признаков)
- **Последовательные признаки** — per-token массивы в JSON (`token_logprobs_json` и др.)
- Колонка `label` (**нужно добавить**): 0 = корректный ответ, 1 = галлюцинация

## Установка

```bash
pip install -r requirements.txt
```

## Использование

Во всех скриптах обязателен флаг `--label_col` с именем целевой колонки.

### 1. Классические модели — `train_classical.py`

Логистическая регрессия, Random Forest, Gradient Boosting, XGBoost, LightGBM.
Запускается по всем именованным наборам признаков из `features.py`.

```bash
# Все наборы признаков, все модели
python train_classical.py --data final_features.csv --label_col label

# Только один набор признаков
python train_classical.py --data final_features.csv --label_col label \
    --feature_set mhad_full

# Доступные наборы признаков:
#   uncertainty, meta, uncertainty+meta,
#   mhad_hidden, mhad_full, sep_layers, sep_first+last, all_features
```

Результат: `results_classical.csv`

---

### 2. SEP-зонды по слоям — `train_sep_probes.py`

Логистические регрессионные зонды по каждому слою, позиции и их комбинациям.
Sweep по силе регуляризации C. Воспроизводит процедуру отбора слоёв из статьи.

```bash
python train_sep_probes.py --data final_features.csv --label_col label
```

Результат: `results_sep_probes.csv`

---

### 3. MHAD MLP — `train_mhad_mlp.py`

Отбор топ-k признаков через linear probe (MHAD Step-1), затем MLP классификаторы
разных глубин на наборах с hidden states первого и последнего токена.

```bash
python train_mhad_mlp.py --data final_features.csv --label_col label

# Параметры
python train_mhad_mlp.py --data final_features.csv --label_col label \
    --epochs 80 --top_k_neurons 64 --device cuda
```

Результат: `results_mhad_mlp.csv`

---

### 4. Табличные нейросети — `train_tabular_nn.py`

Архитектуры, специально разработанные для табличных данных:

| Модель | Описание |
|---|---|
| **TabMLP** | Глубокий Residual MLP с BatchNorm и skip-connections |
| **TabTransformer** | Multi-head self-attention над эмбеддингами признаков |
| **AutoInt** | Dilated self-attention для моделирования попарных взаимодействий |

```bash
python train_tabular_nn.py --data final_features.csv --label_col label \
    --feature_set all_features --epochs 60 --device auto
```

Результат: `results_tabular_nn.csv`

---

### 5. Sequence-модели — `train_sequence_models.py`

Обрабатывают per-token массивы (`token_logprobs_json`, `token_entropies_json`,
`token_top1_probs_json`, `token_margins_json`) как временные последовательности.

| Модель | Описание |
|---|---|
| **Conv1D** | 1-D CNN + глобальный усреднённый пулинг |
| **Bi-GRU** | Двунаправленный GRU с attention-weighted pooling |
| **TCN** | Temporal Convolutional Network с дилатированными свёртками |
| **TransformerSeq** | Transformer encoder с CLS-токеном |

```bash
python train_sequence_models.py --data final_features.csv --label_col label \
    --max_len 256 --epochs 40 --device auto
```

Результат: `results_sequence_models.csv`

---

### 6. Сводный отчёт — `summarise_results.py`

Читает все `results_*.csv`, выводит топ-20 конфигураций и сохраняет график.

```bash
python summarise_results.py
```

Результат: `summary_results.png`

---

## Быстрый старт (все эксперименты)

```bash
pip install -r requirements.txt

python train_classical.py       --data final_features.csv --label_col label
python train_sep_probes.py      --data final_features.csv --label_col label
python train_mhad_mlp.py        --data final_features.csv --label_col label --device auto
python train_tabular_nn.py      --data final_features.csv --label_col label --device auto
python train_sequence_models.py --data final_features.csv --label_col label --device auto

python summarise_results.py
```

## Флаги

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--data` | Путь к CSV | `final_features.csv` |
| `--label_col` | Имя целевой колонки | `label` |
| `--n_splits` | Количество фолдов CV | `5` |
| `--epochs` | Эпохи обучения (нейросети) | зависит от скрипта |
| `--batch` | Размер батча | `256` |
| `--lr` | Learning rate | `1e-3` |
| `--device` | `cpu` / `cuda` / `mps` / `auto` | `auto` |
| `--output` | Путь к файлу результатов | зависит от скрипта |
| `--seed` | Random seed | `42` |
