# celsus

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

celsus test task

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── textures       <- Textures used for dataset generation
│   └── datasets       <- Root of project datasets 
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes celsus a Python module
    ├── analytics               <- Package for analysis
    ├── cli_api                 <- CLI scripts that provide iterface for basic repo utils 
    ├── data                    <- Code for data rading/generation
    ├── nn                      <- Building block for models training
    ├── schemas                 <- Package where all schemas/intermediate data structures are defined
    ├── trian                   <- Package with training scipts defined
    └── consts.py               <- Project constants
    └── utils.py                <- Project utils
```

# Отчет по выполненным работам  

## Данные   

## Исходное задание 
```
Написать кастомный PyTorch Dataset, который выдаёт пару изображений (32×32) и бинарный лейбл (0/1):
На каждом изображении случайно размещается либо круг, либо квадрат в случайном месте.
Лейбл: 1, если фигуры одинаковые (оба круга или оба квадрата), 0 — если разные.
```

## Выполненные работы  

При генерации датасета учитывал следующее  

### Разнообразие датасета  

В качестве характеристик которые могут меняться у фигур выделил: размер, цвет (фигуры и фона), поворот.  
В качестве вариации цветов использовал контрастную палитру из matplotlib, и для дополнитиельного тестового датасета использовал абстрактные текстуры для покраски фигур.  
[демо семплов](notebooks/0.3-as-figure_sample_generation_demo.ipynb)

### Качество при маленьком целевом разрешении  

Чтобы установить ограничения на размер и повороты внимательно ознакомился как будут изменяться фигуры при их вариации.  Также тестировал использование antialising.   
В итоге было принято решение использовать antialiasing и ограничение минимального размера.  
[отчет_по_ограничениям_для_круга](notebooks/0.0-as-circle_AA.ipynb)  
[отчет_по_ограничениям_для_квадрата](notebooks/0.1-as-square_EDA.ipynb)  
[демо_генерации_фигур](notebooks/0.3-as-figure_sample_generation_demo.ipynb)  


### Итоги  

Получено 3 датасета, ниже приведены статистики по числу лейблов:  
train -  0-511, 1-489  
test - 0-106, 1-94  
val:  
    color - 0-56, 1-44  
    texture - 0-47, 1-53  