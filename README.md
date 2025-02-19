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

