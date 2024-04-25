import os

from rdkit import Chem
from pathlib import Path
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from Mold2_pywrapper import Mold2

path_to_zipfile = Path(__file__).parent / 'descriptor/Mold2-exe.zip'
mold2 = Mold2.from_executable(str(path_to_zipfile))

logger = logging.getLogger('Featurizer')
logger.setLevel(logging.INFO)


class BaseMolFeature:
    def __init__(self, **kwargs):
        self.calc = self._load_func(**kwargs)

    def _load_func(self, **kwargs):
        raise NotImplementedError

    def _load_mol(self, moldata: str or Chem.Mol):
        """load mol"""
        if isinstance(moldata, str):
            try:
                mol = Chem.MolFromSmiles(moldata)
                assert mol.GetNumHeavyAtoms() > 0
            except Exception:
                logger.error(f'{moldata} is not a valid SMILES string.')
                return None
        elif isinstance(moldata, Chem.Mol):
            mol = moldata
        else:
            logger.error(f'moldata type is not allowed')
            return None

        if mol is None:
            logger.error(f'mol is None')
            return None

        return mol

    def featurize(self, moldata: str or Chem.Mol):
        """Calc mols """
        mol = self._load_mol(moldata)
        # fp = self.calc(mol)
        # return np.nan_to_num(np.array(fp, dtype=np.float32)).squeeze()
        try:
            fp = self.calc(mol)
            return np.nan_to_num(np.array(fp, dtype=np.float32)).squeeze()
        except Exception as e:
            logger.error(f'{e}')
            logger.error(f'calc feature error: {moldata}')
            return None



def clip_sparse(vect, nbits):
    vector = [0] * nbits
    for i, v in vect.GetNonzeroElements().items():
        vector[i] = min(v, 255)
    return vector


def mol_to_fp(fp_func: callable, mol: Chem.Mol):
    try:
        fp = fp_func(mol)
        return np.nan_to_num(np.array(fp, dtype=np.float32)).squeeze()
    except Exception:
        logger.error(f'calc feature error.')
        return None


class MolFPEncoder(BaseMolFeature):
    """Molecular fingerprints"""

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def _load_func(self, **kwargs):
        nBits = int(kwargs.get("nBits", 2048))
        radius = int(kwargs.get("radius", 3))
        minPathLen = int(kwargs.get("minPathLen", 1))
        maxPathLen = int(kwargs.get("maxPathLen", 30))

        if kwargs.get('fp_name') == "rdkitfp":  #
            from rdkit import Chem
            return lambda x: Chem.RDKFingerprint(x, minPath=1, maxPath=7, fpSize=nBits)

        if kwargs.get('fp_name') == "rdkitfp_unbranch":  #
            from rdkit import Chem
            return lambda x: Chem.RDKFingerprint(x, minPath=1, maxPath=7, branchedPaths=False, fpSize=nBits)

        if kwargs.get('fp_name') == "maccsfp":  #
            from rdkit.Chem import rdMolDescriptors
            return lambda x: rdMolDescriptors.GetMACCSKeysFingerprint(x)

        if kwargs.get('fp_name') == "atompairfp":  #
            from rdkit.Chem.AtomPairs import Pairs
            return lambda x: list(
                Pairs.GetHashedAtomPairFingerprint(x, minLength=minPathLen, maxLength=maxPathLen, nBits=nBits))

        if kwargs.get('fp_name') == "torsionfp":  #
            from rdkit.Chem.AtomPairs import Torsions
            return lambda x: list(Torsions.GetHashedTopologicalTorsionFingerprint(x, nBits=nBits))

        if kwargs.get('fp_name') == "estatefp":  # 79 atom types
            from rdkit.Chem.EState import Fingerprinter
            return lambda x: Fingerprinter.FingerprintMol(x)[0]

        if kwargs.get('fp_name') == "avalonfp":
            from rdkit.Avalon.pyAvalonTools import GetAvalonFP
            return lambda x: GetAvalonFP(x, nBits=512)

        if kwargs.get('fp_name') == "avalonfp_count":
            from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
            return lambda x: GetAvalonCountFP(x, nBits=512)

        if kwargs.get('fp_name').startswith("pharmafp_erg"):
            fp_name = kwargs.get('fp_name')
            from rdkit.Chem import AllChem
            # maxpath = float(os.environ.get('MAX_ERG_PATH', 15))
            # print("curent value", maxpath)
            if "_max15" in fp_name:
                return lambda x: AllChem.GetErGFingerprint(x, fuzzIncrement=0, maxPath=20, minPath=1)
            elif "_max15_bin" in fp_name:
                return lambda x: np.array(AllChem.GetErGFingerprint(x, fuzzIncrement=0, maxPath=20, minPath=1), dtype=np.bool)
            else:
                return lambda x: AllChem.GetErGFingerprint(x, fuzzIncrement=0.3, maxPath=21, minPath=1)

        if kwargs.get('fp_name').startswith("pharmafp_base"):
            from rdkit.Chem import ChemicalFeatures
            from rdkit.Chem.Pharm2D.SigFactory import SigFactory
            from rdkit.Chem.Pharm2D import Generate
            fdef = Path(__file__).parent / 'descriptor/mnimalfatures.fdef'
            featFactory = ChemicalFeatures.BuildFeatureFactory(str(fdef))

            fpname = kwargs.get('fp_name')
            if '_long' in fpname:
                MysigFactory = SigFactory(featFactory,
                                          trianglePruneBins=False,
                                          minPointCount=2,
                                          maxPointCount=2)
                MysigFactory.SetBins([(i, i + 1) for i in range(20)])
            elif '_short' in fpname:
                MysigFactory = SigFactory(featFactory,
                                          trianglePruneBins=False,
                                          minPointCount=2,
                                          maxPointCount=2)
                MysigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
            else:
                raise ValueError(f'{fpname} is not allowed.')
            MysigFactory.Init()
            return lambda x: Generate.Gen2DFingerprint(x, MysigFactory)

        if kwargs.get('fp_name').startswith("pmapperfp"):
            fp_name = kwargs.get('fp_name')
            from rdkit.Chem import ChemicalFeatures
            from rdkit.Chem.Pharm2D.SigFactory import SigFactory
            from rdkit.Chem.Pharm2D import Generate
            fdef = Path(__file__).parent / 'descriptor/pmapper_features.fdef'
            featFactory = ChemicalFeatures.BuildFeatureFactory(str(fdef))

            if "_3p3b" in fp_name:
                MysigFactory = SigFactory(featFactory,
                                          trianglePruneBins=False,
                                          minPointCount=3,
                                          maxPointCount=3)
                MysigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
            elif "_2p4b" in fp_name:
                MysigFactory = SigFactory(featFactory,
                                          trianglePruneBins=False,
                                          minPointCount=2,
                                          maxPointCount=2)
                MysigFactory.SetBins([(0, 2), (2, 5), (5, 8), (8, 20)])
            elif "_2p8b" in fp_name:
                MysigFactory = SigFactory(featFactory,
                                          trianglePruneBins=False,
                                          minPointCount=2,
                                          maxPointCount=2)
                MysigFactory.SetBins(
                    [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12)])

            else:
                raise ValueError(f'{fp_name} is not allowed.')

            MysigFactory.Init()
            return lambda x: Generate.Gen2DFingerprint(x, MysigFactory)

        if kwargs.get('fp_name') == 'pharmafp_gobbi':
            from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
            return lambda x: Generate.Gen2DFingerprint(x, Gobbi_Pharm2D.factory)

        if kwargs.get('fp_name') == "mhfp6":
            from mhfp.encoder import MHFPEncoder
            encoder = MHFPEncoder(n_permutations=nBits)
            return lambda x: encoder.fold(encoder.encode_mol(x, radius=3, rings=True, kekulize=True, min_radius=1),
                                          nBits)

        if kwargs.get('fp_name') == "map4":
            from .descriptor.map4 import MAP4Calculator
            encoder = MAP4Calculator(dimensions=nBits, radius=2, is_counted=False, is_folded=True,
                                     fold_dimensions=nBits)
            return lambda x: encoder.calculate(x)

        if kwargs.get('fp_name') == "pubchemfp":
            from .descriptor.pubchemfp import GetPubChemFPs
            return lambda x: GetPubChemFPs(x)

        if kwargs.get('fp_name') == "rdkit2d_chemprop":
            from descriptastorus.descriptors import rdDescriptors
            encoder = rdDescriptors.RDKit2D()
            return lambda x: encoder.processMol(x, smiles="")

        if kwargs.get('fp_name') == "rdkit2d_deepchem":
            import deepchem
            encoder = deepchem.feat.RDKitDescriptors()
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == "mordredfp_2d_deepchem":
            import deepchem
            encoder = deepchem.feat.MordredDescriptors(ignore_3D=True)
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == "maccsfp_deepchem":
            import deepchem
            encoder = deepchem.feat.MACCSKeysFingerprint()
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == "morganfp_deepchem":
            import deepchem
            encoder = deepchem.feat.CircularFingerprint()
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == f"circlefp_deepchem_rad{radius}":
            import deepchem
            if radius == 1:
                encoder = deepchem.feat.CircularFingerprint(radius=radius, size=1024)
            else:
                encoder = deepchem.feat.CircularFingerprint(radius=radius, size=2048)
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == "circlefp_feature_deepchem_rad1":
            import deepchem
            encoder = deepchem.feat.CircularFingerprint(radius=1, size=1024, features=True)
            return lambda x: encoder.featurize(x)

        if kwargs.get('fp_name') == "rdkit2dnorm_chemprop":
            from descriptastorus.descriptors import rdNormalizedDescriptors
            encoder = rdNormalizedDescriptors.RDKit2DNormalized()
            return lambda x: encoder.processMol(x, smiles="")

        if kwargs.get('fp_name') == "maccsfp_rdkit":
            from rdkit.Chem import MACCSkeys
            return lambda x: MACCSkeys.GenMACCSKeys(x)

        if kwargs.get('fp_name') == f"morganfp_basic_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: rd.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits)

        if kwargs.get('fp_name') == f"morganfp_counts_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: clip_sparse(rd.GetHashedMorganFingerprint(x, radius=radius, nBits=nBits), nBits)

        if kwargs.get('fp_name') == f"morganfp_chiral_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: rd.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits, useChirality=True)

        if kwargs.get('fp_name') == f"morganfp_chiralCounts_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: clip_sparse(
                rd.GetHashedMorganFingerprint(x, radius=radius, nBits=nBits, useChirality=True), nBits)

        if kwargs.get('fp_name') == f"morganfp_feature_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: rd.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits,
                                                              invariants=rd.GetFeatureInvariants(x))

        if kwargs.get('fp_name') == f"morganfp_featureCounts_rdkit_rad{radius}":
            from rdkit.Chem import rdMolDescriptors as rd
            return lambda x: clip_sparse(
                rd.GetHashedMorganFingerprint(x, radius=radius, nBits=nBits, invariants=rd.GetFeatureInvariants(x)),
                nBits)

        if kwargs.get('fp_name') == "mold2":
            return lambda x: mold2.calculate([x], show_banner=False).values[0].astype(float)

        if kwargs.get('fp_name') == "autocorr_2d_rdkit":
            from rdkit.Chem import rdMolDescriptors
            return lambda x: rdMolDescriptors.CalcAUTOCORR2D(x)

        if kwargs.get('fp_name') == "moe":
            from .descriptor.moe import gen_moe_desc
            return lambda x: gen_moe_desc(x)

        if kwargs.get('fp_name') == "estate":
            from .descriptor.estate import gen_estate_desc
            return lambda x: gen_estate_desc(x)

        if kwargs.get('fp_name') == "charges":
            from .descriptor.charge import gen_charge_desc
            return lambda x: gen_charge_desc(x)

        if kwargs.get('fp_name') == "autocorr":
            from .descriptor.autocorr import gen_autocorr_desc
            return lambda x: gen_autocorr_desc(x)

        if kwargs.get('fp_name') == "fragment":
            from .descriptor.fragment import gen_fragment_desc
            return lambda x: gen_fragment_desc(x)

        if kwargs.get('fp_name') == "property":
            from .descriptor.property import gen_property_desc
            return lambda x: gen_property_desc(x)

        if kwargs.get('fp_name') == "constitution":
            from .descriptor.constitution import gen_constitution_desc
            return lambda x: gen_constitution_desc(x)

        if kwargs.get('fp_name') == "connectivity":
            from .descriptor.connectivity import gen_connectivity_desc
            return lambda x: gen_connectivity_desc(x)

        if kwargs.get('fp_name') == "topology":
            from .descriptor.topology import gen_topology_desc
            return lambda x: gen_topology_desc(x)

        if kwargs.get('fp_name') == "kappa":
            from .descriptor.kappa import gen_kappa_desc
            return lambda x: gen_kappa_desc(x)

        if kwargs.get('fp_name') == "path":
            from .descriptor.path_desc import gen_path_desc
            return lambda x: gen_path_desc(x)

        if kwargs.get('fp_name') == "matrix":
            from .descriptor.matrix import gen_matrix_desc
            return lambda x: gen_matrix_desc(x)

        if kwargs.get('fp_name') == "infocontent":
            from .descriptor.infocontent import gen_infocontent_desc
            return lambda x: gen_infocontent_desc(x)

        if kwargs.get('fp_name').startswith("pretrain_gin"):
            from .pretrain.GIN.encoder import DglGinPretrainRepr
            from .pretrain.GIN.dataset import SmilesDataset, gin_collate_fn
            from torch.utils.data import DataLoader as thDataLoader
            import torch
            model = DglGinPretrainRepr(kwargs.get('fp_name').split('_', 1)[-1])

            PARAMS = {
                'batch_size': 100,
                'shuffle': False,
                'num_workers': 4,
                'collate_fn': gin_collate_fn
            }

            def inference(smiList):
                dataset = SmilesDataset(smiList)
                valid_fps = [None for _ in range(len(smiList))]
                valid_ids = dataset.valid_ids
                dataloader = thDataLoader(dataset, **PARAMS)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        batch_feats = model(batch)
                        for idx, feat in enumerate(batch_feats.detach().cpu().numpy()):
                            feat_idx = batch_idx * PARAMS['batch_size'] + idx
                            valid_fps[valid_ids[feat_idx]] = feat

                # del model
                return valid_fps

            return lambda x: inference(x)

        if kwargs.get('fp_name').startswith("pretrain_chemprop"):
            from .pretrain.chemprop.encoder import ChempropPretrainRepr
            from chemprop.data import get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
            import torch
            model = ChempropPretrainRepr(kwargs.get('fp_name').split('_', 1)[-1])

            PARAMS = {
                'batch_size': 50,
                'shuffle': False,
                'num_workers': 4,
            }

            def inference(smiList):
                valid_fps = [None for _ in range(len(smiList))]
                dataset = get_data_from_smiles(
                    smiles=[[s] for s in smiList],
                    skip_invalid_smiles=False,
                    features_generator=None
                )
                valid_ids = []
                valid_data = []
                for full_index in range(len(dataset)):
                    if all(mol is not None for mol in dataset[full_index].mol):
                        valid_ids.append(full_index)
                        valid_data.append(dataset[full_index])

                valid_ids = np.array(valid_ids)
                dataset = MoleculeDataset(valid_data)
                dataloader = MoleculeDataLoader(dataset, **PARAMS)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        batch_feats = model(batch)
                        for idx, feat in enumerate(batch_feats.detach().cpu().numpy()):
                            feat_idx = batch_idx * PARAMS['batch_size'] + idx
                            valid_fps[valid_ids[feat_idx]] = feat
                # del model
                return valid_fps

            return lambda x: inference(x)

        if kwargs.get('fp_name').startswith('pretrain_grover'):
            from .pretrain.GROVER.dataset import SmilesDataset, grover_collate_fn
            from .pretrain.GROVER.encoder import GroverPretrainRepr
            from torch.utils.data import DataLoader as thDataLoader
            import torch
            model = GroverPretrainRepr(kwargs.get('fp_name').split('_', 1)[-1])

            PARAMS = {
                'batch_size': 100,
                'shuffle': False,
                'num_workers': 4,
                'collate_fn': grover_collate_fn
            }

            def inference(smiList):
                dataset = SmilesDataset(smiList)
                valid_fps = [None for _ in range(len(smiList))]
                valid_ids = dataset.valid_ids
                dataloader = thDataLoader(dataset, **PARAMS)
                with torch.no_grad():
                    for batch_idx, batch_item in enumerate(dataloader):
                        _, batch, features_batch, _, _ = batch_item
                        batch_feats = model(batch, features_batch)
                        for idx, feat in enumerate(batch_feats.detach().cpu().numpy()):
                            feat_idx = batch_idx * PARAMS['batch_size'] + idx
                            valid_fps[valid_ids[feat_idx]] = feat

                # del model
                return valid_fps

            return lambda x: inference(x)

        if kwargs.get('fp_name').startswith('pretrain_gemv1'):
            from .pretrain.GEM.encoder import Gemv1PretrainRepr
            from .pretrain.GEM.dataset import SmilesDataset, gem_collate_fn
            from pgl.utils.data import Dataloader as phDataloader
            import torch
            model = Gemv1PretrainRepr(kwargs.get('fp_name').split('_', 1)[-1])
            PARAMS = {
                'batch_size': 100,
                'shuffle': False,
                'num_workers': 4,
                'collate_fn': gem_collate_fn
            }

            def inference(smiList):
                dataset = SmilesDataset(smiList)
                valid_fps = [None for _ in range(len(smiList))]
                valid_ids = dataset.valid_ids
                dataloader = phDataloader(dataset, **PARAMS)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        batch_feats = model(batch)
                        for idx, feat in enumerate(batch_feats.numpy()):
                            feat_idx = batch_idx * PARAMS['batch_size'] + idx
                            valid_fps[valid_ids[feat_idx]] = feat

                # del model
                return valid_fps

            return lambda x: inference(x)

        if kwargs.get('fp_name').startswith('pretrain_kBert'):
            from .pretrain.kmolbert.encoder import KMolBertPretrainRepr
            from .pretrain.kmolbert.dataset import SmilesDataset, kmolbert_collate_fn
            import torch
            from torch.utils.data import DataLoader
            model = KMolBertPretrainRepr(kwargs.get('fp_name').split('_', 1)[-1])
            PARAMS = {
                'batch_size': 100,
                'shuffle': False,
                'num_workers': 4,
                'collate_fn': kmolbert_collate_fn
            }
            def inference(smiList):
                dataset = SmilesDataset(smiList)
                valid_fps = [None for _ in range(len(smiList))]
                valid_ids = dataset.valid_ids
                dataloader = DataLoader(dataset, **PARAMS)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        batch_feats = model(*batch)
                        for idx, feat in enumerate(batch_feats.detach().cpu().numpy()):
                            feat_idx = batch_idx * PARAMS['batch_size'] + idx
                            valid_fps[valid_ids[feat_idx]] = feat

                # del model
                return valid_fps

            return lambda x: inference(x)

        raise ValueError(f'fp_name {kwargs.get("fp_name", None)} is not allowed.')
