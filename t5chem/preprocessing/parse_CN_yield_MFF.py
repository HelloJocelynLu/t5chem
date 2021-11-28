import argparse
import pandas as pd
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from EFGs import standize

core_smiles = '[Pd-2]1(OS(=O)(=O)C(F)(F)F)[NH2+]c2ccccc2-c2ccccc21'
# Define reaction pattern
rxn = rdChemReactions.ReactionFromSmarts('[C:1][c:2]1[c:3][c:4][c:5]([N:6])[c:7][c:8]1.[#17,#35,#53][c:9]>>[C:1][c:2]1[c:3][c:4][c:5]([N:6][c:9])[c:7][c:8]1')


def GetComplex(ligand, core=core_smiles):
    '''Combine metal core (Pd-2] with ligand to get complex smiles'''
    core_mol = Chem.MolFromSmiles(core)
    ligand_mol = Chem.MolFromSmiles(ligand)
    new_mol = Chem.RWMol(Chem.CombineMols(core_mol, ligand_mol))
    target_idx = [atom.GetIdx() for atom in ligand_mol.GetAtoms() if atom.GetSymbol() == 'P']
    assert len(target_idx) == 1
    new_mol.AddBond(0,target_idx[0]+core_mol.GetNumAtoms(),Chem.BondType.SINGLE)
    new_mol.GetAtomWithIdx(target_idx[0]+core_mol.GetNumAtoms()).SetFormalCharge(1)
    Chem.SanitizeMol(new_mol)
    return Chem.MolToSmiles(new_mol)


def main(opts):
    raw_excel = pd.ExcelFile(opts.file_name)
    for name in raw_excel.sheet_names:
        save_file = open(opts.save+'_'+name+'.txt', 'w')
        cur_df = pd.read_excel(raw_excel, name)
        # Generate product
        prod_dict = {x:[] for x in cur_df['Aryl halide'].unique() if type(x)==str}
        for x in cur_df['Aryl halide'].unique():
            if type(x)==str:
                rts = [Chem.MolFromSmiles(x) for x in ['Cc1ccc(N)cc1',x]]
                prod_dict[x].extend([standize(x[0], asMol=True) for x in rxn.RunReactants(rts)])
        # Make sure we generated unique product
        for x in prod_dict:
            prod_dict[x] = list(set(prod_dict[x]))
            assert len(prod_dict[x])==1
            prod_dict[x] = prod_dict[x][0]
        for i in range(len(cur_df)):
            cur_rxn = cur_df.iloc[i]
            if type(cur_rxn['Aryl halide'])!=str: continue
            rts = standize(cur_rxn['Aryl halide'])+'.Cc1ccc(N)cc1'
            cat = GetComplex(cur_rxn['Ligand'], core=core_smiles)
            add = cur_rxn['Additive'] if type(cur_rxn['Additive'])!=float else ''
            rgs = [cat, add, cur_rxn['Base'], 'CS(C)=O']
            rgs = standize('.'.join([x for x in rgs if x]))
            print(rts+'>'+rgs+'>'+prod_dict[cur_rxn['Aryl halide']], cur_rxn['Output'], file=save_file)
        save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse C-N coupling yield.') 
    parser.add_argument("file_name", help="file name")
    parser.add_argument("save", help="file name to save as")
    args = parser.parse_args()
    main(args)
