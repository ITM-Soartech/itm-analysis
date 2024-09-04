
from typing import List
import numpy as np
from itm_schema.kdma_ids import KDMAId
from itm_schema.ml_pipeline import (
    AlignmentTarget,
    KDMAMeasurement,
    KDMAProfile,
    KernelDensity,
)



def create_single_value_profile(
        value: float,
        profile_name: str,
        kde_bandwidth: float,
        kdma_ids: List[KDMAId]) -> KDMAProfile:
    """
    Create a KDMAProfile from a float slider value

    Parameters
    ----------

    value: float
        The slider value
    profile_name: str
        What you want the "RDM" name to be for the created profile
    kde_bandwidth: float
        number to use as the KDE bandwidth
        changing this value impacts the shape of the KDE curve formed

    Returns
    -------
    KDMAProfile
        KDMA profile generated from the single value

    """
    # The KDE sample consistes of only the provided value
    sample = np.full((100, 1), value)

    # Create KDMA measurements for each KDMA
    kdma_measurements = {}
    for kdma_id in kdma_ids:
        kdma_measurements[kdma_id] = KDMAMeasurement(
                kdma_id=kdma_id,
                value=value,
                kde=KernelDensity(bandwidth=kde_bandwidth).fit(sample),
                hist=None,
            )

    profile = KDMAProfile(
            dm_id=profile_name,
            kdma_measurements=kdma_measurements,
        )

    return profile


# Create an alignment target schema object from a list of KDMAProfiles 
# Does not matter if the KDMAProfile was generated from a real RDM or from a float
def create_alignment_target(profiles: List[KDMAProfile]):
    """
    Create an alignment target schema object from a list of KDMAProfiles 

    Parameters
    ----------
    profiles: List[KDMAProfile]
        List of KDMAProfiles to generate this target from

    Returns
    -------
    AlignmentTarget
    """
    alignment_target = AlignmentTarget(target={})

    for kdma_profile in profiles:
        alignment_target.target[kdma_profile.dm_id] = kdma_profile

    return alignment_target
