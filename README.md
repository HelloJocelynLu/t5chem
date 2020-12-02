RxnTransformer
==============================

A few transformer-based models for chemical reactions predictions.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data (.gitignore)
    │   ├── C_N_yield        <- Buchwald–Hartwig reactions (3,955 reactions)
    │   ├── Ni_rxns          <- Ni-catalyzed dicarbofunctionalization. (710 reactions)
    │   ├── STEREO_separated <- USPTO reactions with stereochemistry. (1 million reactions)
    │   └── vocab            <- Vocabulary files. (type: torchtext.Vocab)
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models (.gitignore) <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data                            <- Scripts to parse data
    │   │   └── data_utils.py               <- MolTokenizer, dataset classes
    │   │   └── gen_translation_pretrain.py <- Generate pretraining data for translation task.
    │   │   └── run_stats.py                <- Run statistical analysis on a molecule data file, one line per molecule.
    │   │   └── make_dataset.py             <- Generate vocabulary file based on given file(s).
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts about models
    │   │   ├── models.py  <- T5 model adapted for regression (Removed)
    │   │   └── Trainer.py <- A modified Trainer class (:transformer:) that can save
    │   │                     best weights accordin to validation error.
    │   │
    │   ├── run_pretrain.py         <- Script to train mask-filling pretraining task.
    │   ├── run_trainer.py          <- Script to train molecular translation task.
    │   ├── iterative_selector.py   <- Script to select high-yield reactions iteratively.
    │   ├── rxn_selector.py         <- Script to train and selection out-of-sample reactions based on predicted yields.
    │   ├── run_predict.py          <- Script to test molecular translation, given a trained model.
    │   ├── evaluate_predictions.py <- Evaluate prediction results from T5ConditionalGeneration.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
