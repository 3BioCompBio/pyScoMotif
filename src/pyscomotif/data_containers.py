from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class Residue():
    """Used to store all the data needed about a residue to then calculate the geometric descriptors of pairs of residues."""
    resname: str # Ex: 'A' (Alanine)
    C_alpha: npt.NDArray[np.float64]
    sidechain_CMR: npt.NDArray[np.float64]
    vector: npt.NDArray[np.float64]

    def __getitem__(self, key: str) -> Any:
        """To enable bracket notation to get an attribute value (Ex: Residue['resname'])"""
        return getattr(self, key)

@dataclass
class Residue_pair_data():
    """Used to store the data associated with a pair of residues. Typically used as a value in a dictionary where the keys are pairs of residues."""
    C_alpha_distance: float
    sidechain_CMR_distance: float
    vector_angle: float