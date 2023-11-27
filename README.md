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
TBD
