from enum import Enum

class FeatureBase(Enum):
    # path fp
    F1A = 'rdkitfp'
    F1B = 'rdkitfp_unbranch'
    
    # substucture from experts
    F2A1 = 'maccsfp_deepchem'
    F2A2 = 'maccsfp_rdkit'
    F2D  = 'pubchemfp'
    F2E  = 'avalonfp'
    F2F  = 'mold2'

    # atom pair
    F3A = 'atompairfp'
    F3B = 'mhfp6'
    F3C = 'map4'

    F4A = 'torsionfp'
    F5A = 'estatefp'

    F7A  = 'pharmafp_erg'
    F7A1 = 'pharmafp_erg_max15'
    F7A2 = 'pharmafp_erg_max15_bin'
    F7B1 = 'pharmafp_base_short'
    F7B2 = 'pharmafp_base_long'
    # F7C  = 'pharmafp_gobbi'
    F7D1 = 'pmapperfp_3p3b'
    # F7D2 = 'pmapperfp_2p4b'
    F7D3 = 'pmapperfp_2p8b'

    F8A = 'rdkit2d_chemprop'
    F8B = 'rdkit2d_deepchem'
    F8C = 'rdkit2dnorm_chemprop'
    F8D = 'mordredfp_2d_deepchem'

    # pybiomed
    F9B = 'moe'
    F9C = 'estate'
    F9D = 'charges'
    F9E1 = 'autocorr'
    F9E2 = 'autocorr_2d_rdkit'
    F9F = 'fragment'
    F9G = 'property'
    F9H = 'constitution'
    F9I = 'connectivity'
    F9J = 'topology'
    F9K = 'kappa'
    F9L = 'path'
    F9M = 'matrix'
    F9N = 'infocontent'
    # circle fp
    F10A1 = 'morganfp_deepchem'
    F10B1 = 'circlefp_deepchem_rad1'
    F10B2 = 'circlefp_deepchem_rad2'
    F10B3 = 'circlefp_deepchem_rad3'
    F10E1 = 'circlefp_feature_deepchem_rad1'
    F10F1 = 'morganfp_basic_rdkit_rad1'
    F10F2 = 'morganfp_basic_rdkit_rad2'
    F10F3 = 'morganfp_basic_rdkit_rad3'
    F10G1 = 'morganfp_counts_rdkit_rad1'
    F10G2 = 'morganfp_counts_rdkit_rad2'
    F10G3 = 'morganfp_counts_rdkit_rad3'
    F10H1 = 'morganfp_chiral_rdkit_rad1'
    F10H2 = 'morganfp_chiral_rdkit_rad2'
    F10H3 = 'morganfp_chiral_rdkit_rad3'
    F10I1 = 'morganfp_chiralCounts_rdkit_rad1'
    F10I2 = 'morganfp_chiralCounts_rdkit_rad2'
    F10I3 = 'morganfp_chiralCounts_rdkit_rad3'
    F10J1 = 'morganfp_feature_rdkit_rad1'
    F10J2 = 'morganfp_feature_rdkit_rad2'
    F10J3 = 'morganfp_feature_rdkit_rad3'
    F10K1 = 'morganfp_featureCounts_rdkit_rad1'
    F10K2 = 'morganfp_featureCounts_rdkit_rad2'
    F10K3 = 'morganfp_featureCounts_rdkit_rad3'

    # F11A1  = 'pretrain_gin_supervised_contextpred'
    # F11A2  = 'pretrain_gin_supervised_edgepred'
    # F11A3  = 'pretrain_gin_supervised_infomax'
    # F11A4  = 'pretrain_gin_supervised_masking'
    # F11A5  = 'pretrain_gin_supervised_edgepred_LogD'   # pretrain_gin_supervised_edgepred_Lipophilicity
    # F11A6  = 'pretrain_gin_supervised_masking_LogD'
    # F11A7  = 'pretrain_gin_supervised_contextpred_LogD'
    # F11A8  = 'pretrain_gin_supervised_infomax_LogD'
    # F11A9  = 'pretrain_gin_supervised_edgepred_ESOL'
    # F11A10 = 'pretrain_gin_supervised_edgepred_FreeSolv'
    # F11A11 = 'pretrain_gin_supervised_edgepred_LogpNew' # pretrain_gin_supervised_edgepred_LogpNew
    # F11A12 = 'pretrain_gin_supervised_edgepred_RTtime'   #pretrain_gin_supervised_edgepred_RetentionTime
    #
    # F11B1  = 'pretrain_chemprop_supervised_scaffold_LogD'  # pretrain_chemprop_supervised_scaffold_explipo
    # F11B2  = 'pretrain_chemprop_supervised_random_LogD'   #pretrain_chemprop_supervised_random_
    # F11B3  = 'pretrain_chemprop_supervised_scaffold_LogDrescoss'
    # F11B4  = 'pretrain_chemprop_supervised_random_LogDrescoss'
    # # F11B5  = 'pretrain_chemprop_supervised_scaffold_cyp2c9sub'
    # F11B6  = 'pretrain_chemprop_supervised_scaffold_bioav'
    # F11B7  = 'pretrain_chemprop_supervised_scaffold_bbbreg'
    # F11B8  = 'pretrain_chemprop_supervised_scaffold_ames'
    # F11B9  = 'pretrain_chemprop_supervised_random_RTtime'
    # F11B10 = 'pretrain_chemprop_supervised_random_pka'
    # F11B11 = 'pretrain_chemprop_supervised_random_LogP'
    # F11B12 = 'pretrain_chemprop_supervised_scaffold_LogP'
    #
    # F11B13 = 'pretrain_chemprop_supervised_random_ESOL'
    # F11B14 = 'pretrain_chemprop_supervised_scaffold_hepg2'
    # F11B15 = 'pretrain_chemprop_supervised_scaffold_hepg2_part'
    # F11B16 = 'pretrain_chemprop_supervised_scaffold_hepg2class'
    # F11B17 = 'pretrain_chemprop_supervised_scaffold_auxbbb'
    # F11B18 = 'pretrain_chemprop_supervised_scaffold_bbbclass'
    # F11B19 = 'pretrain_chemprop_supervised_scaffold_herg_clean'
    #
    #
    # F11C1 = 'pretrain_chemprop_supervised_scaffold_ld50'
    # F11C2 = 'pretrain_chemprop_supervised_scaffold_halflife'
    #
    # F11D1 = 'pretrain_grover_base'
    # F11D2 = 'pretrain_grover_large'
    #
    # # F11E1 = 'pretrain_gemv1_class'
    # # F11E2 = 'pretrain_gemv1_regr'
    #
    # F11F1 = 'pretrain_kBert_base'
    # F11F2 = 'pretrain_kBert_wcl'
    # F11F3 = 'pretrain_kBert_chiral'
    # F11F4 = 'pretrain_kBert_chiral_rs'


 

FeatKeyToName = {x.name: x.value for x in FeatureBase}
FeatNameToKey = {x.value: x.name for x in FeatureBase}

if __name__ == '__main__':
    print(len(FeatKeyToName))
    print(len(FeatNameToKey))