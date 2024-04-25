import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops, Descriptors
from molvs import standardize_smiles
from chembl_structure_pipeline import standardize_mol
from maxqsaring.logging_helper import create_logger
logger = create_logger(__name__)

rxnstr = '[*:1]~[S+](~[O-])~[*:2]>>[*:1]S(=O)[*:2]'
so_rxn_pattern = AllChem.ReactionFromSmarts(rxnstr)
so_pattern = Chem.MolFromSmarts('[S+](~[O-])')

def DebugAtomProps(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            logger.info(f'Atom: {atom.GetIdx()} {atom.GetSymbol()}')
            logger.info(f'TotH/ExpH/ImpH: {atom.GetTotalNumHs()} {atom.GetNumExplicitHs()} {atom.GetNumImplicitHs()}')
            logger.info(f'Formal Charge: {atom.GetFormalCharge()}')
            logger.info(f'Total Valence: {atom.GetTotalValence()}')
            logger.info(f'Other info: {atom.GetPropsAsDict()}')



def revise_natom(atom):
    if atom.GetFormalCharge()== 1:
        if  atom.GetTotalValence()<=3 or  \
            (atom.GetTotalValence()==4 and atom.GetNumExplicitHs()>=1):
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNumRadicalElectrons(0)


def revise_sopat(smi):
    '''use rdkit to transform [*:1][S+](~[O-])[*:2] to [*:1]S(=O)[*:2]'''
    mol = Chem.MolFromSmiles(smi)
    if not mol.HasSubstructMatch(so_pattern):
        return smi
    product = so_rxn_pattern.RunReactant(mol, 0)
    if len(product) >0:
        mol_new = product[0][0]
        new_smi = Chem.MolToSmiles(mol_new, isomericSmiles=False, kekuleSmiles=True)
        return new_smi
    else:
        return None


def clean_smi(smi: str):
    try:
        mol= Chem.MolFromSmiles(smi)
        assert (mol is not None) and (mol.GetNumHeavyAtoms() >0)
    except:
        raise ValueError(f'Error to read smiles-1: {smi}')
        
    new_smi = Chem.MolToSmiles(standardize_mol(mol), isomericSmiles=False, kekuleSmiles=True)
    mol= Chem.MolFromSmiles(new_smi)

    if mol is None:
        DebugAtomProps(mol)
        raise ValueError(f'Error to read smiles-2: {new_smi}')
        
    mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
    if len(mol_frags) >1:
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            revise_natom(atom)
    try:
        new_smi = Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)
    except:
        DebugAtomProps(mol)
        raise ValueError(f'Error to export smiles-3: {new_smi}')

    new_smi = revise_sopat(new_smi)
    return new_smi


def check_carbon(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() =='C':
            return True

    return False



def doCleanSmilesV1(insmiles: str, keep_parent_frag=True, modify_n_status=False, modify_s_pattern=False, max_mw=1000):
    '''steps to cleanize smiles'''
    try:
        mol = Chem.MolFromSmiles(insmiles)
        assert (mol is not None) and (mol.GetNumHeavyAtoms() > 0)
    except:
        logger.info(f'Error-[1]: fail to read smiles -> \n{insmiles}')
        return None, "ERROR-1"

    new_smi = Chem.MolToSmiles(standardize_mol(mol), isomericSmiles=False, kekuleSmiles=True)
    mol = Chem.MolFromSmiles(new_smi)

    if mol is None:
        logger.info(f'Error-[2]: fail to standardize mol -> \n{insmiles}')
        return None, "Error-2"


    if keep_parent_frag:
        mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
        if len(mol_frags) > 1:
            mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

    if not check_carbon(mol):
        logger.info(f'Error-[2]: mol has no carbons -> \n{insmiles}')
        return None, "Error-2"

    if Descriptors.ExactMolWt(mol)>max_mw:
        logger.info(f'Error-[2]: mol is large than {max_mw} -> \n{insmiles}')
        return None, "Error-2"

    if modify_n_status:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                revise_natom(atom)

    try:
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)
    except:
        logger.info(f'Error-[3]: fail to export new smiles -> \n{insmiles}')
        return None, "Error-3"

    if modify_s_pattern:
        new_smiles = revise_sopat(new_smiles)
        if new_smiles is None:
            logger.info(f'Error-[4]: fail to transform [S+][O-] -> \n{insmiles}')
            return None, "Error-4"

    return new_smiles, "Valid"


if __name__ == '__main__':
    smi_list=[
        'CC12CCC3C4=C(C=C(O)C=C4)CC(CCCCCCCCC[S+]([O-])CCCC(F)(F)C(F)(F)F)C3C1CCC2O',
        'CC1=C(OCC(F)(F)F)C=CN=C1C[S+]([O-])C1=NC2=CC=CC=C2N1',
    ]
    for smi_ in smi_list:
        # mol = Chem.MolToSmiles()
        new_smi_ = revise_sopat(smi_)
        print(new_smi_)