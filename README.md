# Распознавание дорожных знаков на видеозаписях

CV проект в рамках курса **Глубокое обучение на практике**

Команда:

- Борисов Дмитрий Сергеевич (@dsborisov)
- Изюмова Анастасия Витальевна (@starminalush)
- Азаматов Ильдар Русланович (@eelduck)

## Содержание репозитория

- В папке [src/](/src/) содержится код запуска демонстрации проекта.
- Файлы модели содержатся на [диске](https://disk.yandex.ru/d/wrJI_jGpbC3yVQ)
- В папке models должны находиться актуальные веса лучшей модели
- В папке [notebooks/](/notebooks/) содержатся jupyter ноутбуки с обучением модели, инференсом, подготовкой датасета

## Начало работы

### Подготовка окружения

- Устанавливаем `python 3.10`
    - Windows

      Устанавливаем через [официальный установщик](https://www.python.org/downloads/)

    - Linux

        ```bash
        sudo apt install python3.10
        ```

- Устанавливаем [poetry](https://python-poetry.org/docs/#installation)
    - Windows

      Используйте [официальные инструкции](https://python-poetry.org/docs/#windows-powershell-install-instructions)
      или команду `powershell`

        ```powershell
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
        ```

    - Linux

        ```bash
        curl -sSL https://install.python-poetry.org | python3 -
        ```
- Устанавливаем требуемые пакеты с помощью команды
    ```bash
    poetry install
    ```

- Скачиваем веса модели [отсюда](https://disk.yandex.ru/d/wrJI_jGpbC3yVQ) и перемещаем их в папку [models](/models/)

## Запуск системы

- Для запуска системы введите команды
    ```bash
    cd src/
    python gradio_app.py
    ```
  После этого у вас запустится Gradio приложение по адресу http://localhost:7860/

## Демо

- Загрузите видеофайл с регистратора в левое поле
- Нажмите кнопку __Submit__
- Дождитесь кноца обработки файла, появится размеченное видео в правом окне. При желании вы можете его скачать
  ![image](https://github.com/DmitryChatBotov/traffic-sign-recognition/assets/41739221/682a884f-22f1-43fe-a2df-b0d22c48de49)


## Обоснованность выбранного решения

### Формат

В рамках проекта нужно было реализовать возможность запуска нейросетевой модели на мобильных устройствах (android, ios).
Для обеспечения этой возможности мы конвертировали модель PyTorch в формат ONNX.
ONNX – открытый стандарт для конвертации моделей машинного обучения из разных фреймворков в единый формат, а также для
обмена моделями между фреймворками.
С помощью библиотеки ONNXRuntime можно будет [запустить](https://onnxruntime.ai/docs/tutorials/mobile/)
сконвертированную модель детекции дорожных знаков прямо на телефоне под любой ОС, и она будет работать.

В таблице ниже приведены метрики производительности модели на CPU, чтобы примерно понимать, как быстро модель детекции
дорожных знаков в формате CPU будет работать на мобильных устройствах.



## Эксперименты

### Подбор гиперпараметров
Так как мы используем модели yolo от ultralytics, подбор гиперпараметров для обучения выполняется "из коробки". Нам не нужно проводить серию экспериментов по оптимизации параметров обучения, таких как подбор optimizer и прочих параметров.
Таким образом, мы подобрали только: 
 - Количество эпох на обучение. Мы ориентировались на то, на сколько хватит ресурсов colab и kaggle, но при этом старались получить наилучшее качество, какое можем получить.
 - Размер изображения установили 640 потому что мы искали баланс между ресурсами и быстродействием модели. Не выбрали размер 1280, потому что у нас сильно увеличилось время на инференс, а качество не сильно возрасло, по сравнению с обучением модели на размере изображения 640.

Важные метрики выделенны курсивом. Нам важно, чтобы у нас был как можно более высокий MAP50 и MAP50-95, а также скорость работы. Учитывая, что нужно дать возможность запустить модель на мобильном устройстве, рассмотрим еще метрику CPU FPS, которая показывает количество кадров в секунду при обработке на CPU.

| Модель  | Датасет                                                                                           | Precision(all) | Recall(all) | _MAP50(all)_ | _MAP50-95(all)_ | Кол-во эпох на обучение | Кол-во классов | Формат |  GPU Device | _FPS_  | _CPU FPS_ |batch_size |
|---------|---------------------------------------------------------------------------------------------------|----------------|-------------|--------------|-----------------|-------------------------|----------------|--------|---------------------|--------|-----------| --- |
| yolov8s | [Russian traffic sign images dataset](https://www.kaggle.com/datasets/watchman/rtsd-dataset/data) | 0.834          | 0.774       | **0.839**    | **0.621**       | 15                      | 117            | ONNX   | Nvidia GeForce 3050 | **36** | **15**    | 1          |
| yolov5s | [Russian traffic sign images dataset](https://www.kaggle.com/datasets/watchman/rtsd-dataset/data) | 0.739          | 0.46        | 0.533        | 0.385           | 15                      | 117            | ONNX   | Nvidia GeForce 3050 | 26     | 13        | 1          |

