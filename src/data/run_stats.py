import os
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from itertools import chain
from tqdm import tqdm

lg = rdkit.RDLogger.logger()  
lg.setLevel(rdkit.RDLogger.CRITICAL) 

##### functions #####
def get_element(row):
    try:
        return set([a.GetSymbol() for a in row['mol'].GetAtoms()])
    except:
        return None

def getNumAtom(row):
    try:
        return Descriptors.HeavyAtomCount(row['mol'])
    except:
        return None

def get_rot_bond(row):
    try:
        return AllChem.CalcNumRotatableBonds(row['mol'])
    except:
        return None

def get_MW(row):
    try:
        return Descriptors.ExactMolWt(row['mol'])
    except:
        return None

def mol2inchi(row):
    try:
        return Chem.MolToInchi(row['mol'])
    except:
        return None

def get_can_smiles(row):
    try:
        return Chem.MolToSmiles(row['mol'])
    except:
        return None

element_types = []
##### read data #####
num_lines = sum(1 for line in open('pubchem/PCH.csv','r'))
for i, df in enumerate(tqdm(
        pd.read_csv('pubchem/PCH.csv', header=None, chunksize=1000),
        total=num_lines//1000+1)):
    df.columns = ['dataset', 'ID', 'smiles']
    if os.path.exists('pubchem/temp_csv/'+str(i)+'.csv'): continue
#    df.columns = ['smiles']

    PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles', molCol='mol')
    df = df[df['mol'].map(lambda x: x is not None)]

    ##### calculate #####
    df.loc[:,'ele_type'] = df.apply(get_element, axis=1)
    df.loc[:,'#atom'] = df.apply(getNumAtom, axis=1)
    df.loc[:,'#robond'] = df.apply(get_rot_bond, axis=1)
    df.loc[:,'MW'] = df.apply(get_MW, axis=1)
    df.loc[:,'inchi'] = df.apply(mol2inchi, axis=1)
    df.loc[:,'can_smiles'] = df.apply(get_can_smiles, axis=1)

    element_types.append(set(chain.from_iterable(df['ele_type'])))
    df[['inchi','can_smiles','#atom','#robond','MW']].to_csv('pubchem/temp_csv/'+str(i)+'.csv', index=None)
print(set(chain.from_iterable(element_types)))
