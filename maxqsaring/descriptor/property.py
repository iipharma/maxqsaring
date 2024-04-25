#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:05:49 2019

@author: wanxiang.shen@u.nus.edu

molecular physchemical properties
"""

                    
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import MolSurf
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from mordred import Calculator, descriptors

from collections import namedtuple
import math
import numpy as np



from rdkit import Chem
import math

    
    
    
def _CalculateBondNumber(mol,bondtype='SINGLE'):

    """
    ################################################################# 
    **Internal used only*
    
    Calculation of bond counts in a molecule. it may be 
    
    SINGLE, DOUBLE, TRIPLE and AROMATIC
    #################################################################
    """
    i=0;
    for bond in mol.GetBonds():

        if bond.GetBondType().name==bondtype:
            i=i+1
            
    return i


def CalculateUnsaturationIndex(mol):
    """
    #################################################################
    Calculation of unsaturation index.
    
    ---->UI
    
    Usage:
        
        result=CalculateUnsaturationIndex(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    nd=_CalculateBondNumber(mol,bondtype='DOUBLE')
    nt=_CalculateBondNumber(mol,bondtype='TRIPLE')
    na=_CalculateBondNumber(mol,bondtype='AROMATIC')
    res=math.log((1+nd+nt+na),2)
    
    return round(res,3)
    

def CalculateHydrophilicityFactor(mol):
    """
    #################################################################
    Calculation of hydrophilicity factor. The hydrophilicity 
    
    index is described in more detail on page 225 of the 
    
    Handbook of Molecular Descriptors (Todeschini and Consonni 2000).
    
    ---->Hy
    
    Usage:
        
        result=CalculateHydrophilicityFactor(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    nheavy=mol.GetNumHeavyAtoms()
    nc=0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==6:
            nc=nc+1
    nhy=0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==7 or atom.GetAtomicNum()==8 or atom.GetAtomicNum()==16:
            atomn=atom.GetNeighbors()
            for i in atomn:
                if i.GetAtomicNum()==1:
                    nhy=nhy+1
                    
    
    try:
        res=(1+nhy)*math.log((1+nhy),2)+nc*(1.0/nheavy*math.log(1.0/nheavy,2))+math.sqrt((nhy+0.0)/(nheavy^2))
    
    except: 
        res = 0
    return res
    
def FilterItLogS(mol):
    '''
    Fragement based solubity value: """Filter-it™ LogS descriptor.: based on a simple fragment-based method. 
    http://silicos-it.be.s3-website-eu-west-1.amazonaws.com/software/filter-it/1.0.2/filter-it.html#installation
    '''
    calc = Calculator(descriptors.LogS)
    return calc(mol).asdict().get('FilterItLogS')
    

############################################################
def nAcidicGroup(mol):
    calc = Calculator(descriptors.AcidBase.AcidicGroupCount)
    return calc(mol).asdict().get('nAcid')

def nBasicGroup(mol):
    calc = Calculator(descriptors.AcidBase.BasicGroupCount)
    return calc(mol).asdict().get('nBase')

def isAcidic(mol):
    if nAcidicGroup(mol) > nBasicGroup(mol):
        return 1
    else:
        return 0

def isBasic(mol):
    if nAcidicGroup(mol) < nBasicGroup(mol):
        return 1
    else:
        return 0
    
def isNeutral(mol):
    if nAcidicGroup(mol) == nBasicGroup(mol):
        return 1
    else:
        return 0    
    
############################################################


def CalculateAverageMolWeight(mol):
    N = mol.GetNumAtoms()
    return Chem.Descriptors.MolWt(mol) / N



_properties = { #weight
                'MolWeight':Descriptors.MolWt,
                'MolExactWeight':Descriptors.ExactMolWt,
                'MolAverageWeight':CalculateAverageMolWeight,
                'MolHeavyAtomWeight':Descriptors.HeavyAtomMolWt,

                # other from https://github.com/lanternpharma/tdc-bbb-martins/blob/main/code/generate_features.py
                'NumHDonors': Descriptors.NumHDonors,
                'NumHAcceptors': Descriptors.NumHAcceptors,
                'NumRotatableBonds': Descriptors.NumRotatableBonds,
                'NumAtoms': Chem.rdchem.Mol.GetNumAtoms,
                'NumHeavyAtoms': Chem.rdchem.Mol.GetNumHeavyAtoms,
                'FormalCharge':Chem.rdmolops.GetFormalCharge,
                'NumRings': Chem.rdMolDescriptors.CalcNumRings,

                #philibic
                'MolFilterItLogS': FilterItLogS,
                'MolSLogP': Crippen.MolLogP,
                'MolQedWeightsMax': QED.weights_max,
                'MolQedWeightsMean': QED.weights_mean,
                'MolQedWeightsNone':QED.weights_none,

                'MolRefractivity':Crippen.MolMR,
                'MolFractionCSP3':Descriptors.FractionCSP3,
                'MolHydrophilicityFactor':CalculateHydrophilicityFactor,
                'MolUnsaturationIndex':CalculateUnsaturationIndex,
    
                #denisty
                'MolFpDensityMorgan1': Descriptors.FpDensityMorgan1,
                'MolFpDensityMorgan3': Descriptors.FpDensityMorgan2,
                'MolFpDensityMorgan3': Descriptors.FpDensityMorgan3,
                'MolTPSA': MolSurf.TPSA,
    
                #two ekectrions
                'NumRadicalElectrons':Descriptors.NumRadicalElectrons,
                'NumValenceElectrons':Descriptors.NumValenceElectrons,
    
                #acidic, basic
                'NumAcidicGroup': nAcidicGroup,
                'NumBasicGroup': nBasicGroup,
                'MolisAcidicType': isAcidic,
                'MolisBasicType': isBasic,
                'MolisNeutralType': isNeutral,
            }

from collections import OrderedDict

def addRules(mol_dict: dict):
    if (mol_dict['MolWeight'] <= 500) and (mol_dict['MolSLogP'] <= 5) and (mol_dict['NumHDonors'] <= 5) and (mol_dict['NumHAcceptors'] <= 5) and (mol_dict['NumRotatableBonds'] <= 5):
        rule = 1
    else:
        rule = 0
    mol_dict['lipinski_rule'] = rule

    if (mol_dict['MolWeight'] >= 160) and (mol_dict['MolWeight'] <= 480) and (mol_dict['MolSLogP'] >= 0.4) and (mol_dict['MolSLogP'] <= 5.6) and (mol_dict['NumAtoms'] >= 20) and (mol_dict['NumAtoms'] <= 70) and (mol_dict['MolRefractivity'] >= 40) and (mol_dict['MolRefractivity'] <= 130):
        rule =1
    else:
        rule = 0
    mol_dict['ghose_filter'] = rule

    if (mol_dict['NumRotatableBonds'] <= 10) and (mol_dict['MolTPSA'] <= 140):
        rule = 1 
    else:
        rule = 0
    mol_dict['veber_filter'] = rule

    if (mol_dict['MolWeight'] <= 300) and (mol_dict['MolSLogP'] <= 3) and (mol_dict['NumHDonors'] <= 3) and (mol_dict['NumHAcceptors'] <= 3) and (mol_dict['NumRotatableBonds'] <= 3):
        rule = 1
    else:
        rule = 0
    mol_dict['rule_of_3'] = rule

    if (mol_dict['MolWeight'] >= 200) and (mol_dict['MolWeight'] <= 500) and (mol_dict['MolSLogP'] >= int(0 - 5)) and (mol_dict['MolSLogP'] <= 5) and (mol_dict['NumHDonors'] >= 0) and (mol_dict['NumHDonors'] <= 5) and (mol_dict['NumHAcceptors'] >= 0) and (mol_dict['NumHAcceptors'] <= 10) and (mol_dict['FormalCharge'] >= int(0-2)) and (mol_dict['FormalCharge'] <= 2) and (mol_dict['NumRotatableBonds'] >= 0) and (mol_dict['NumRotatableBonds'] <= 8) and (mol_dict['NumHeavyAtoms'] >= 15) and (mol_dict['NumHeavyAtoms'] <= 50):
        rule =1 
    else: 
        rule = 0
    mol_dict['REOS_filter'] = rule
    
    if (mol_dict['MolWeight'] < 400) and (mol_dict['NumRings'] > 0) and (mol_dict['NumRotatableBonds'] < 5) and (mol_dict['NumHDonors'] <= 5) and (mol_dict['NumHAcceptors'] <= 10) and (mol_dict['MolSLogP'] < 5):
        rule = 1
    else: 
        rule = 0
    mol_dict['drug_like'] = rule

    return mol_dict


def GetProperty(mol):
    """
    #################################################################
    Get the dictionary of Properties descriptors for given moelcule mol
    
    Usage:
        
        result=GetProperties(mol)
        
        Input: mol is a molecule object.

        Output: result is a dict form containing all constitutional values.
    #################################################################
    """
    result=OrderedDict()
    for key, func in _properties.items():
        result[key]=round(func(mol),5)
    return result


_PropertyNames = list(_properties.keys())

def gen_property_desc(mol):
    """
    #################################################################
    Get the dictionary of Properties descriptors for given moelcule mol
    
    Usage:
        
        result=GetProperties(mol)
        
        Input: mol is a molecule object.

        Output: result is a dict form containing all constitutional values.
    #################################################################
    """
    result=OrderedDict()
    for key, func in _properties.items():
        result[key]=round(func(mol),5)

    result = addRules(result)
    return list(result.values())
##########################################################

if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetProperty(m))
        
    pd.DataFrame(x)
