import re
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from EFGs import standize


pat = r'\d+(?:\.\d+)?%'
p2f = lambda x: float(x.strip('%'))/100


def mols_from_smiles_list(all_smiles):
    '''Given a list of smiles strings, this function creates rdkit
    molecules'''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols


def replace_deuterated(smi):
    return re.sub('\[2H\]', r'[H]', smi)


def get_tagged_atoms_from_mols(mols):
    '''Takes a list of RDKit molecules and returns total list of
    atoms and their tags'''
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags


def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True  # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    if atom1.GetDegree() != atom2.GetDegree(): return True
    if atom1.IsInRing() != atom2.IsInRing(): return True
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2: return True

    return False


def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')
            and a.GetProp('molAtomMapNumber') == str(mapnum)][0]


def get_tetrahedral_atoms(reactants, products):
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            atom_tag = ar.GetProp('molAtomMapNumber')
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    if ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED or\
                            ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms


def set_isotope_to_equal_mapnum(mol):
    for a in mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))


def get_frag_around_tetrahedral_center(mol, idx):
    '''Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes'''
    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0\
               else '[#{}]'.format(a.GetAtomicNum()) for a in mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(mol, ids_to_include, isomericSmiles=True,
                                   atomSymbols=symbols, allBondsExplicit=True,
                                   allHsExplicit=True)


def check_tetrahedral_centers_equivalent(atom1, atom2):
    '''Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped'''
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx())
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False


def clear_isotope(mol):
    [a.SetIsotope(0) for a in mol.GetAtoms()]


def get_rxn_tag(reaction_smiles):
    '''Given a reaction, return a reaction tag.
    0: Reaction without any stereocenters involved
    1: Reaction with chirality involved, but not in reaction center
    2: Reaction with chirality involved, and in reaction center
    '''
    rt, pd = reaction_smiles.split('>>')
    reactants = mols_from_smiles_list(replace_deuterated(rt).split('.'))
    products = mols_from_smiles_list(replace_deuterated(pd).split('.'))
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)
    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)

    # Find differences
    changed_atoms = {} # actual reactant atom species

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):

        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atoms: # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms[reac_tag] = reac_atoms[j]
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms[reac_tag] = reac_atoms[j]
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atoms:
            if reac_tag not in prod_atom_tags:
                changed_atoms[reac_tag] = reac_atoms[j]

    # Atoms that change CHIRALITY (just tetrahedral for now...)
    tetra_exist = True
    tetra_atoms = get_tetrahedral_atoms(reactants, products)
    [set_isotope_to_equal_mapnum(reactant) for reactant in reactants]
    [set_isotope_to_equal_mapnum(product) for product in products]

    if not tetra_atoms:
        tetra_exist = False
        [clear_isotope(reactant) for reactant in reactants]
        [clear_isotope(product) for product in products]
        return 0

    for (atom_tag, ar, ap) in tetra_atoms:

        if atom_tag in changed_atoms:
            [clear_isotope(reactant) for reactant in reactants]
            [clear_isotope(product) for product in products]
            return 2
        else:
            unchanged = check_tetrahedral_centers_equivalent(ar, ap) and \
                    ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
            if not unchanged:
                # Make sure chiral change is next to the reaction center and not
                # a random specifidation (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    if neighbor.HasProp('molAtomMapNumber'):
                        nei_mapnum = neighbor.GetProp('molAtomMapNumber')
                        if nei_mapnum in changed_atoms:
                            tetra_adj_to_rxn = True
                            break
                if tetra_adj_to_rxn:
                    changed_atoms[atom_tag] = ar
                    [clear_isotope(reactant) for reactant in reactants]
                    [clear_isotope(product) for product in products]
                    return 2

    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]
    return 1


def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])


def FakeRxnChecker(rxn):
    '''
    Check if a reaction is valid.
    A reaction would be viewed as invalid if product is the same as one of reactants
    '''
    reactants, reagents, prods = rxn.strip().split('>')
    rts = [standize(x) for x in reactants.split('.')+reagents.split('.') if x]
    pds = [standize(x) for x in prods.split('.') if x]
    if set(rts)&set(pds):
        pds = '.'.join(set(pds).difference(rts))
        if (not rts) or (not pds):
            return True
    return False


def StripTrivalProd(rxn):
    '''
    Remove trival prod(s) that also appear in reactants
    '''
    reactants, reagents, prods = rxn.strip().split('>')
    if reagents:
        reactants = reactants + '.' + reagents
    rts = reactants.split('.')
    pds = prods.split('.')
    if set(rts)&set(pds):
        pds = list(set(pds).difference(rts))
    assert len(pds) == 1
    return rxn.rsplit('>',1)[0]+'>'+pds[0]


def GetYield(yield_list, tolerance=0.1):
    valid_yield = []
    for y in yield_list:
        # Remove np.nan
        if type(y)==str:
            per = re.findall(pat, y)
            # Remove yield data > 100%
            if per and p2f(per[-1]) <= 1:
                valid_yield.append(p2f(per[-1]))
    if not valid_yield:
        return False
    if len(valid_yield) == 1:
        return valid_yield[0]
    if np.abs(valid_yield[0]-valid_yield[1]) < tolerance:
        return np.average(valid_yield)
    return valid_yield[0]
