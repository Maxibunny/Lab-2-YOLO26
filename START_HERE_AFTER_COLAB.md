# Лабораторная №2 — что делать после Google Colab

**Вариант №5:** `0=chair`, `1=table`, `2=bed`, `3=couch`. Работа с тестовыми фотографиями; приложение не использует камеру в реальном времени.

## Что подтвердил сохранённый Lab2.ipynb

- Обучение `yolo26n.pt`, коммит Ultralytics `c4e0ba9310ca5c567adb2b02f351f4d721557c71`, версия 8.4.202; NVIDIA Tesla T4.
- Существуют сохранённые результаты в ноутбуке: 95 выполненных эпох из 100 (ранняя остановка patience=25, лучшая эпоха 70).
- 2285 изображений train, 404 validation; **mAP50 validation = 0.8027135954392056**; AP50 chair=0.72392674695, table=0.80338058888, bed=0.81573047849, couch=0.86781656744.
- Экспортированный `best.tflite` имеет **вход [1,3,640,640] FP32 (NCHW)** и выход `[1,8,8400] FP32` (сырые предсказания).
- Ноутбук пишет, что `results_bundle.zip` (38.78 MiB) и `prepared.zip` сохранены на **Google Drive**. Они не находились во вложенном исходном `lab2.zip`. Это нельзя заменить пустыми каталогами.
- **Точность 0.8 на независимом тесте не подтверждена.** Мобильная сборка и время на телефоне также пока не подтверждены.

## Шаг A. Сначала перенеси два архива из Drive

В Google Drive найди:

1. `MyDrive/Lab2_Furniture/prepared.zip` — размеченный датасет.
2. `MyDrive/Lab2_Furniture/experiments/furniture_yolo26n_640_v2/results_bundle.zip` — веса, tflite, логи и метрики.

На Windows открой **PowerShell** в папке, внутри которой лежит `Lab2_YOLO26_Furniture` (или замени пути на свои):

```powershell
New-Item -ItemType Directory -Force -Path '.\01_Results_Colab' | Out-Null
# Скопируй скачанные prepared.zip/results_bundle.zip в 01_Results_Colab через Проводник
Expand-Archive -LiteralPath '.\01_Results_Colab\prepared.zip' -DestinationPath '.\Lab2_YOLO26_Furniture\dataset' -Force
Expand-Archive -LiteralPath '.\01_Results_Colab\results_bundle.zip' -DestinationPath '.\Lab2_YOLO26_Furniture' -Force
cd '.\Lab2_YOLO26_Furniture'
# В prepared.zip может остаться абсолютный путь Colab; пересоздаём YAML без изменения картинок.
python scripts/fix_local_dataset_config.py
python scripts/verify_project.py
```

`fix_local_dataset_config.py` приводит `data.yaml` к абсолютному пути **этого Windows ПК**, а `docker_data.yaml` — к `/workspace/dataset/prepared`. Это необходимо, поскольку `prepared.zip` уже существовал в Drive, а рабочий YAML затем переписывался в Colab; сам старый ZIP мог не обновиться.

В выводе должно быть `OK` для `runs/best.pt`, `runs/metrics.json`, `android/app/src/main/assets/best.tflite`, `dataset/prepared/data.yaml`, `dataset/prepared/docker_data.yaml`. В ZIP их **нет** намеренно, поскольку они создаются обучением в Drive. При отсутствии файлов **не собирай приложение**.

Проверь, что на месте **именно** `android/app/src/main/assets/best.tflite` (не `best.tflite.tflite` и не другая папка).

## Шаг B. Собери и запусти Android

1. Установи **Android Studio**, SDK Platform 35 / Android SDK Build Tools, JDK 17 (обычно встроен в Studio). Android-телефон должен иметь Android 8.0+ (API 26+).
2. В Studio выбери **File → Open**, открой **`Lab2_YOLO26_Furniture/android`**. Это Gradle-проект; открывать нужно именно `android`.
3. Если Gradle Wrapper отсутствует, открой PowerShell в `.../android` и запусти: `powershell -ExecutionPolicy Bypass -File .\bootstrap_gradle_windows.ps1`. Скрипт скачивает **официальный** Gradle 8.9. Убедись, что появились `gradlew.bat` и `gradle/wrapper/gradle-wrapper.jar`.
4. В Studio выполни **Sync Project with Gradle Files**, затем **Build → Build Bundle(s) / APK(s) → Build APK(s)** (название зависит от версии Studio). Альтернатива в PowerShell из `android`: `.\gradlew.bat :app:assembleDebug`. Файл APK ожидается после успешной сборки по пути `android/app/build/outputs/apk/debug/app-debug.apk`.
5. На смартфоне включи **Для разработчиков → Отладка по USB**; подключи к ПК и разреши отладку. В Studio выбери устройство, запусти конфигурацию `app` кнопкой Run.
6. Передай на телефон собственные реальные фотографии стула, стола, кровати, дивана (можно несколько объектов на одном фото). Нажми **«Одно фото»** либо **«Выбрать папку»**. Успешная работа: видна фотография с рамками, русскими названиями классов и временем модели в миллисекундах.
7. Нажми **«Профилирование (12 запусков)»**: 2 разогрева + 10 измеренных вызовов `Interpreter.run()`. Затем **«Сохранить CSV с временем»**. В исправленной версии эти 10 измерений сохраняются как строки `mode=benchmark` с `run_index=1..10`. Сохрани CSV и скриншоты четырёх классов для защиты.
8. Если на устройстве ошибка, смотри **View → Tool Windows → Logcat**, фильтр `FurnitureYOLO`. В журнале напечатана форма входа (ожидается `[1, 3, 640, 640]`) и тип раскладки NCHW.

**Ключевое исправление:** исходный APK требовал `[1,640,640,3]`; в этом проекте `FurnitureDetector.kt` поддерживает NCHW `[1,3,H,W]`, который реально указан в выходе твоего Colab. Весы заново обучать или экспортировать для этого **не нужно**.

**Если нет модели в Android assets**, исходный архив из вопроса содержал только `ADD_MODEL_HERE.txt`: скачать **именно** `results_bundle.zip` с Drive и распаковать его.

## Шаг C. Docker (на Windows, не в Colab)

Установи Docker Desktop с backend WSL2 и Git for Windows. Убедись, что Docker Desktop **запущен**.

В PowerShell из **корня** `Lab2_YOLO26_Furniture`:

```powershell
.\setup_git.bat
# Фиксация того же исходного кода YOLO, что применялся в Google Colab:
$sha = (Get-Content '.\runs\ultralytics_commit.txt').Trim()
# Должно быть: c4e0ba9310ca5c567adb2b02f351f4d721557c71
# Если файла нет, но это именно подтверждённая версия из предоставленного блокнота:
# $sha = 'c4e0ba9310ca5c567adb2b02f351f4d721557c71'
git -C '.\external\ultralytics' fetch origin $sha
git -C '.\external\ultralytics' checkout --detach $sha
git add .gitmodules external/ultralytics external/mobile-yolo

docker info
docker build -t lab2-yolo26 .
New-Item -ItemType Directory -Force -Path '.\runs-docker' | Out-Null
# Smoke-test: запускаем ОДНУ эпоху на CPU. Это НЕ повтор полноценного обучения.
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\dataset\prepared,target=/workspace/dataset/prepared,readonly" `
  --mount "type=bind,source=$($PWD.Path)\runs-docker,target=/workspace/runs" `
  lab2-yolo26 `
  --data /workspace/dataset/prepared/docker_data.yaml `
  --runs /workspace/runs --epochs 1 --batch 2 --device cpu --name smoke_cpu
```

Проверь `runs-docker/smoke_cpu/`, `runs-docker/best.pt`, `runs-docker/metrics.json`. Если Docker выдаёт ошибку памяти — уменьшай `--batch` до 1 и выдели ему больше ОЗУ. CPU-обучение займёт время; этот smoke-test нужен **только для проверки воспроизводимого запуска**.

Для полноценного обучения на совместимом NVIDIA GPU с NVIDIA Container Toolkit вместо CPU можно использовать `--gpus all --device 0 --epochs 100 --batch 8 --patience 25 --seed 42 --name furniture_yolo26n_640_v2`, но этот Dockerfile не проверен на CUDA в данной среде. `--gpus all` само по себе не гарантирует совместимость драйвера, CUDA и образа.

В Python-скрипте `scripts/train.py` параметры `patience=25`, `workers=2`, `seed=42`, `batch=8` теперь соответствуют сохранённому запуску Colab. **Идентичная метрика при полном повторном обучении не гарантируется**, если версии PyTorch/CUDA отличаются.

## Шаг D. Проверка на независимом тестовом наборе

Когда преподаватель предоставит фотографии с YOLO-разметкой, положи их в `dataset/instructor_test/images/test/` и `dataset/instructor_test/labels/test/`. Для каждого JPG/PNG — соответствующий TXT с классом 0..3 и нормированными координатами `class cx cy w h`.

Создай `dataset/instructor_test/test.yaml` (подставь абсолютный путь папки в `path`):

```yaml
path: C:/YOUR_PATH/Lab2_YOLO26_Furniture/dataset/instructor_test
train: images/test
val: images/test
test: images/test
names:
  0: chair
  1: table
  2: bed
  3: couch
```

После установки Python + Ultralytics (или в Colab, где весы и пакет уже есть) выполни:

```powershell
python scripts/evaluate_test.py --weights runs/best.pt --test-data dataset/instructor_test/test.yaml
```

Итог `runs/test_metrics.json` содержит `map50_test` и `target_0_8_met`. **Для зачёта тест должен показать ≥ 0,8**. Значение validation = 0,8027 нельзя выдавать за тестовое.

## Шаг E. Публичный GitHub

После проверки Docker и Android:

```powershell
git status
git submodule status
git add .
git commit -m "Lab 2: YOLO26 furniture Android LiteRT and Docker"
# git remote add origin https://github.com/ВАШ_ЛОГИН/ВАШ_РЕПОЗИТОРИЙ.git
git branch -M main
git push -u origin main
```

Должны быть сохранены исходники Android, Dockerfile, скрипты, `.gitmodules`, SHA submodule, `README`, ссылка на размеченный датасет. Веса и ZIP-архивы исключены из Git (`.gitignore`); положи их в GitHub Release или облако с доступом и дай проверяемую ссылку. Отметь версию набора HomeObjects-3K; инструкция со ссылкой: https://docs.ultralytics.com/datasets/detect/homeobjects-3k/ .

## Что невозможно подтвердить по присланным файлам

- В исходном `lab2.zip` **нет** экспортированной модели, обученных весов, датасета и логов; они доступны только по путям Drive, указанным в ноутбуке. Я не могу проверить целостность этих бинарных файлов без `results_bundle.zip` и `prepared.zip`.
- APK нельзя считать проверенным, пока Android Studio на твоём ПК не соберёт проект, а устройство не выполнит инференс.
- Замеры времени на смартфоне и mAP на закрытом тесте ещё отсутствуют.
