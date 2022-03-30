#  ReadMe
## Data Availability

All datasets involved in this work can be downloaded free of charge from [here](https://yzhang.hpc.nyu.edu/T5Chem/).

- USPTO\_TPL
- USPTO\_MIT 
- USPTO\_50k
- C–N Coupling
- USPTO\_500\_MT

The user should extract those compressed files before using by:
```
$ tar -xjvf /path/to/your/downloaded.tar.bz2
```

After unzip those data files, you should be good to go! If you check your data folder(s), you will see `train.source`, `train.target`, `val.source`, `val.target`, `test.source`, `test.target`. Then all you need to do is to pass the path to those data files to `--data_dir` argument during training.

## Data Usage
### USPTO\_TPL
The data set for the reaction type classification task was originally derived from the USPTO database by [Lowe](https://aspace.repository.cam.ac.uk/handle/1810/244727) and introduced by [Schwaller et al.](https://www.nature.com/articles/s42256-020-00284-w) This strongly imbalanced data set consists of 445,000 reactions divided into 1000 classes. Reactions were randomly split into 90% for training and validation and 10% for testing.
- Data folder: `data/USPTO_1k_TPL`
- Task type: classification

### USPTO\_MIT
This benchmark data set has been prepared by [Jin et al.](https://proceedings.neurips.cc/paper/2017/hash/ced556cd9f9c0c8315cfbe0744a3baf0-Abstract.html) based on the USPTO database of [Lowe](https://aspace.repository.cam.ac.uk/handle/1810/244727)  and includes both separated and mixed versions. It consists of 479,000 reactions: 409,000 for training, 30,000 for validation, and 40,000 for testing.
- Data folder: `data/USPTO_MIT/MIT_separated/` or `data/USPTO_MIT/MIT_mixed/`
- Task type: product

### USPTO\_50k
 The data set was a [filtered](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00564) version of Lowe’s patent data set. It contains only 50,000 reactions that have been classified into 10 broad reaction types. Here, 40,000, 5000, and 5000 reactions were used for training, validation, and testing, respectively.
- Data folder: `data/USPTO_50k/`
- Task type: reactants

### C–N Coupling
In 2018, [Ahneman et al.](https://www.science.org/doi/10.1126/science.aar5169) performed 4608 Pd-catalyzed Buchwald–Hartwig C–N cross coupling reactions. These nanomole-scale experiments were carried out on three 1536-well plates consisting of a full matrix of 15 aryl and heteroaryl halides, three bases, four ligands, and 23 isoxazole additives. Here, 3955 applicable reactions after removing control groups were used. 
- Data folder: 
    - Ten random split cross validation sets: `data/C_N_yield/MFF_FullCV_*/`
    - Four out-of-sample test sets: `data/C_N_yield/MFF_Test*/`
- Task type: regression

### USPTO\_500\_MT
In order to illustrate the multitasking capability of our unified model, we introduced a new data set that is applicable for multiple reaction prediction tasks, including forward reaction prediction, reactants prediction (single step retrosynthesis), reagents prediction, reaction yield prediction, and reaction type classification. It contains 116,360 for training, 12,937 for validation and 14,238 for testing.
- Task type: product
    - Data folder: `data/USPTO_500_MT/Product/`
- Task type: reactants
    - Data folder: `data/USPTO_500_MT/Reactants/`
- Task type: classification
    - Data folder: `data/USPTO_500_MT/Classification/`
- Task type: reagents
    - Data folder: `data/USPTO_500_MT/Reagents/`
- Task type: regression
    - Data folder: `data/USPTO_500_MT/Yield/`
- Task type: mixed (include all three sequence-to-sequence tasks)
    - Data folder: `data/USPTO_500_MT/mixed/`

There are also three extra files are kept as references: `train/val/test.csv`
As mentioned in the paper, USPTO\_500\_MT is derived from USPTO\_1k\_TPL. Therefore, our *.csv files keep the following columns as in original source:
- level\_0,index: old indexes, can be ignored
- original\_rxn,  mapped\_rxn: reaction smiles with atom mapping, with or without reactant/reagent separation
- fragments: fragment information, not using here
- source: patent id for reaction source
- year: publication year
- confidence: quality of this record, a score
- canonical\_rxn\_with\_fragment\_info, canonical\_rxn: reaction smiles without atom mapping
- ID,reaction_hash: can be used as identifiers to reactions
- **reactants, reagents, products**: reactants/reagents/product smiles
- retro\_template,template\_hash: Retrosynthesis template (SMARTS). The templates Schwaller et. al. used to assign different reaction classes for every reaction.
- selectivity,outcomes: most of them are 1.0. Not used in T5Chem
- **labels**: The reaction class. (500 classes in total with label [0-499])
- **Yield**: reaction yield data recovered as described in paper

Only those bold entries were used to construct USPTO\_500\_MT trained on T5Chem. It requires some simple dataframe operations, for example:
- **Forward reaction prediction**: source: `reactants.reagents>>` target: `product`  (see `data/USPTO_500_MT/Product/`)
- **Retrosynthesis**: source: `product` target: `reactants`  (see ` data/USPTO_500_MT/Reactants/`)
- **Reagent prediction**: source: `reactants>>product` target: `reagents`  (see `data/USPTO_500_MT/Reagents/`)
- **Classification**: source: `reactants.reagents>>product` target: `labels`  (see `data/USPTO_500_MT/Classification/`)
- **Yield prediction**:  source: `reactants.reagents>>product` target: `Yield` (see `data/USPTO_500_MT/Yield`)

For a quick start, one can safely use *.source and *.target files in the 6 sub-folders.