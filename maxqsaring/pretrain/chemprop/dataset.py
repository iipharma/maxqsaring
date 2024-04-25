from typing import List
from chemprop.data import MoleculeDatapoint, MoleculeDataset

class SmilesDataset(MoleculeDatapoint):
    def __init__(self, data: List[MoleculeDatapoint]):
        super().__init__(data)
