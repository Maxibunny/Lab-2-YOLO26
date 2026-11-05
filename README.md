# Лабораторная работа № 2 — YOLO26n: детекция мебели на Android

**Задача:** обучить свою модель YOLO26 на четырёх классах, экспортировать её в LiteRT, запустить в Android-приложении, измерить время инференса и обеспечить воспроизводимое обучение через Docker.

**Вариант 5.** `0 — chair (стул)`, `1 — table (стол)`, `2 — bed (кровать)`, `3 — couch (диван)`.

> **Статус:** обучение и экспорт модели подтверждены сохранёнными результатами Google Colab. Сборка и запуск APK на физическом Android-устройстве, успешный запуск Docker на стороннем ПК и качество на независимом тесте должны быть проверены отдельно. Валидационная метрика не заменяет тестовую.

## 1. Результат обучения

Обучение проводилось в Google Colab на NVIDIA Tesla T4. Источник чисел — сохранённый вывод `notebooks/Lab2.ipynb` и сводка `docs/colab_result_summary.json`.

| Показатель | Значение |
|---|---|
| Модель | YOLO26n (`yolo26n.pt`, fine-tuning) |
| Версия Ultralytics в Colab | `8.4.202` |
| Коммит исходного кода Ultralytics | `c4e0ba9310ca5c567adb2b02f351f4d721557c71` |
| Датасет | 2 285 train + 404 validation = 2 689 изображений |
| Параметры | 640 px, batch 8, seed 42, patience 25, максимум 100 эпох |
| Обучение | остановлено после 95 эпох; лучшая контрольная точка — 70-я эпоха |
| **mAP@0.5 (validation)** | **0,8027** |
| mAP@0.5:0.95 (validation) | 0,5883 |
| AP@0.5 по классам | chair 0,7239; table 0,8034; bed 0,8157; couch 0,8678 |
| LiteRT `best.tflite` | вход FP32 `[1,3,640,640]`, выход `[1,8,8400]` |

Метрика **mAP@0.5 ≥ 0,8 на независимой тестовой выборке преподавателя пока не подтверждена**.

## 2. Исходный датасет и правила разметки

Использован [HomeObjects-3K](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/) от Ultralytics (авторы Glenn Jocher и Muhammad Rizwan; лицензия AGPL-3.0): [прямая загрузка исходного ZIP](https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip).

Исходный датасет содержит 12 классов, но для этой работы оставлены 4. Номера классов были преобразованы из оригинальных `bed=0, sofa=1, chair=2, table=3` в **`chair=0, table=1, bed=2, couch=3`**. Разбиение исходного набора на train/validation сохранено. Подготовленный набор включает YOLO-разметку (`labels/train`, `labels/val`) и изображения (`images/train`, `images/val`).

> Для воспроизведения **именно этого** эксперимента используйте `prepared.zip`, а не необработанный оригинальный ZIP. Независимого test-раздела в HomeObjects-3K нет.

## 3. Где скачать веса и размеченный датасет

**[Папка с материалами в Google Drive](https://drive.google.com/drive/u/0/folders/1At6BhubWZABF40FQ_W-4dRtwUxH9f1NS)**. 

Скачайте два файла, созданных в Colab:

1. **`prepared.zip`** — подготовленные изображения и YOLO-разметка. В исходной структуре Drive: `Lab2_Furniture/prepared.zip`.
2. **`results_bundle.zip`** — `runs/best.pt`, логи/метрики обучения и `android/app/src/main/assets/best.tflite`. В исходной структуре Drive: `Lab2_Furniture/experiments/furniture_yolo26n_640_v2/results_bundle.zip`.

`best.pt` используется для Python и повторной оценки, а `best.tflite` — для Android. Эти архивы и тяжёлые бинарные файлы намеренно **не хранятся в GitHub**.

### Как восстановить проект на Windows

Клонируйте GitHub-репозиторий с submodule (подставьте URL своего репозитория):

```powershell
git clone --recurse-submodules https://github.com/USERNAME/Lab2_YOLO26.git
cd Lab2_YOLO26
```

Скопируйте скачанные `prepared.zip` и `results_bundle.zip` **в текущую папку** `Lab2_YOLO26`. Затем выполните:

```powershell
Expand-Archive -LiteralPath '.\prepared.zip' -DestinationPath '.\dataset' -Force
Expand-Archive -LiteralPath '.\results_bundle.zip' -DestinationPath '.' -Force
python .\scripts\fix_local_dataset_config.py
python .\scripts\verify_project.py
```

Ожидаемые пути:

```text
dataset/prepared/images/train/
dataset/prepared/images/val/
dataset/prepared/labels/train/
dataset/prepared/labels/val/
dataset/prepared/data.yaml
dataset/prepared/docker_data.yaml
runs/best.pt
runs/metrics.json
runs/ultralytics_commit.txt
android/app/src/main/assets/best.tflite
```

Если какого-либо файла нет, нельзя считать локальный проект полностью восстановленным. В исходном ZIP проекта обученных весов и датасета нет.

## 4. Docker: сборка и тест обучения на Windows

Требования: Git for Windows, Docker Desktop с Linux/WSL2 backend, запущенный Docker Engine, свободное место для образа и данных. Официальная инструкция: [Docker Desktop + WSL2](https://docs.docker.com/desktop/features/wsl/).

PowerShell — **из корневого каталога `Lab2_YOLO26_Furniture`**, где находится `Dockerfile`:

**1. Проверить Docker и датасет**

```powershell
docker version
Test-Path '.\dataset\prepared\images\train'
Test-Path '.\dataset\prepared\docker_data.yaml'
```

Две последние команды должны вернуть `True`. Если `docker_data.yaml` отсутствует, после распаковки датасета выполните `python .\scripts\fix_local_dataset_config.py`.

**2. Закрепить код Ultralytics на версии из Colab.** Если репозиторий клонирован без submodule или проект создан из ZIP, выполните ` .\setup_git.bat ` один раз. Затем:

```powershell
$sha = 'c4e0ba9310ca5c567adb2b02f351f4d721557c71'
git -C '.\external\ultralytics' fetch origin $sha
git -C '.\external\ultralytics' checkout --detach $sha
git submodule status
```

Точка SHA у `external/ultralytics` должна соответствовать `$sha`. Отключённая ветка (`detached HEAD`) для закреплённого submodule — нормальное состояние.

**3. Собрать Docker-образ:**

```powershell
docker build --progress=plain -t lab2-yolo26 .
```

Сборка может потребовать загрузки Python-пакетов, включая PyTorch. Датасет и веса в образ не копируются: они передаются через тома при запуске.

**4. Выполнить smoke-test: одну эпоху на CPU:**

```powershell
New-Item -ItemType Directory -Force '.\runs-docker' | Out-Null

docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\dataset\prepared,target=/workspace/dataset/prepared,readonly" `
  --mount "type=bind,source=$($PWD.Path)\runs-docker,target=/workspace/runs" `
  lab2-yolo26 `
  --data /workspace/dataset/prepared/docker_data.yaml `
  --runs /workspace/runs `
  --epochs 1 `
  --batch 2 `
  --device cpu `
  --name smoke_cpu
```

В PowerShell знак `` ` `` должен стоять **последним символом строки** (без пробелов после него). При необходимости первая загрузка `yolo26n.pt` требует интернета.

**5. Проверить результаты:**

```powershell
Test-Path '.\runs-docker\smoke_cpu\weights\best.pt'
Test-Path '.\runs-docker\best.pt'
Test-Path '.\runs-docker\metrics.json'
Get-Content '.\runs-docker\metrics.json'
```

При **успешном завершении контейнера** все три проверки файлов должны дать `True`. `runs-docker` не отправляется в GitHub. Если Docker вылетает по памяти, для smoke-test можно уменьшить `--batch 2` до `--batch 1` и `--workers 0`.

Этот запуск доказывает, что скрипт может **запускать обучение из Docker с внешними входными и выходными каталогами**, но **не повторяет 95-эпоховый эксперимент**. Для полноценного воспроизведения нужно обучить модель повторно с согласованными версиями зависимостей и гиперпараметрами. Этот Dockerfile не проверен на конкретном NVIDIA GPU окружении.

## 5. Android Studio: запуск на устройстве

Откройте в Android Studio **подкаталог `android/`**. В нём находится готовое Kotlin-приложение для обработки фотографий (не видеопотока камеры). Перед сборкой обязательно перенесите `best.tflite` из `results_bundle.zip`.

На Windows в папке `android` при отсутствии Gradle Wrapper выполните:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_gradle_windows.ps1
.\gradlew.bat :app:assembleDebug
```

Затем запустите приложение на Android-телефоне с USB-отладкой. Проверьте обнаружение мебели на фотографиях, выполните «Профилирование (12 запусков)» (2 прогрева, 10 измерений), сохраните CSV с временем инференса и скриншоты распознавания. Без APK, работающего на устройстве, эти пункты нельзя считать выполненными.

## 6. Оценка на независимом тесте

После получения тестовых фотографий с YOLO-разметкой создайте `dataset/instructor_test/images/test/`, `dataset/instructor_test/labels/test/` и `dataset/instructor_test/test.yaml`. В `test.yaml` должен быть верный `path` и тот же порядок классов `chair`, `table`, `bed`, `couch`.

```powershell
python .\scripts\evaluate_test.py --weights .\runs\best.pt --test-data .\dataset\instructor_test\test.yaml
```

Результат — тестовая метрика `map50_test` (требуется **не менее 0,8**). Пока тест не предоставлен, значение неизвестно.

## 7. Содержимое репозитория

| Путь | Содержимое |
|---|---|
| `notebooks/Lab2.ipynb` | Сохранённые ячейки и результаты обучения Colab |
| `docs/colab_result_summary.json` | Сводка фактически полученных результатов |
| `scripts/train.py` | Обучение и расчёт метрик |
| `scripts/fix_local_dataset_config.py` | Создание Windows- и Docker-путей в YAML |
| `scripts/verify_project.py` | Проверка наличия обязательных артефактов |
| `scripts/evaluate_test.py` | Тестирование модели на независимой выборке |
| `android/` | Kotlin-приложение LiteRT |
| `Dockerfile` | Образ обучения |
| `.gitmodules`, `external/` | Ссылки на исходники Ultralytics и Android-пример (`git submodule`) |
| `START_HERE_AFTER_COLAB.md` | Дополнительные инструкции |

Локальные большие каталоги `dataset/prepared/`, `runs/`, `runs-docker/` и файлы `*.pt`, `*.tflite`, `*.zip` исключены из Git. Для полноценного повторения эксперимента нужны **и код, и публично доступные зафиксированные данные/веса**. Docker сам не содержит подготовленный датасет.

## 8. Ссылки и лицензирование

- [Ultralytics / YOLO26 — исходный код](https://github.com/ultralytics/ultralytics)
- [HomeObjects-3K — описание, лицензия, классы](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/)
- [Git — submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Docker Desktop для Windows](https://docs.docker.com/desktop/features/wsl/)
- [Инструкция GitHub по загрузке локального проекта](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

Использованные исходники и датасет имеют собственные лицензии. Перед распространением производных материалов проверьте условия AGPL-3.0 и лицензии дополнительных компонентов.
