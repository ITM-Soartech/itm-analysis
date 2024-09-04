from typing import Optional

from . import similarity_functions as sim_fns

from itm_schema.ml_pipeline import (
    KDMAProfile,
    KDMAAlignment,
    RDMAlignment,
    AlignmentTarget,
    AlignmentPackage,
)

# construct a Alignment Package
def build_alignment_package(
        kdma_profile: KDMAProfile,
        alignment_target: AlignmentTarget,
        metric: str = 'JS',
        kde_sample_size: int = 100,
        alignment_target_short_name: Optional[str] = None,
    ) -> AlignmentPackage:

    """
    Compute and construct an AlignmentPackage object for a KDMA profile an alignment target

    Parameters
    ----------
    kdma_profile: KDMAProfile
        Decision profile for the decision maker
    alignment_target:
        Alignment target
    alignment_algorithm_name: str
        For now the only valid alignmentAlgorithmName for the graphs is "JS". 
        Support is partially in place for "HD" = HellingerDistance and "KL" = Kullback-Leibler Distance, but
            1) No time to fully pass the params through for the graphing code to use
            2) HD produces values greater than 1 so unclear how it should be inverted. 
    kde_sample_size: int
        the sampling size that will be used for sampling the kdes when comparing them. 
    alignment_target_short_name: str
        will be appeneded to the alignment_target folder name in the artifacts directory structure so you can compute more
        than one alignment target per ADM and have them interleave nicely in the same directory structure. If we decide this is useful, we may
        want to add it to the AlignmentTarget object in the schema.

    Returns
    -------

    AlignmentPackage
    """

    if alignment_target_short_name is None:
        alignment_target_short_name = 'target'

    rdm_alignments_list = []
    mean_list = []

    for id, profile in alignment_target.target.items():
        kdmaAlignment = KDMAAlignment(kdma_alignments={})
        for adm_msr, target_msr in zip(kdma_profile.kdma_measurements.values(), profile.kdma_measurements.values()):
            h_dist =  sim_fns.hellinger_similarity(adm_msr.kde, target_msr.kde, kde_sample_size)
            kl_dist = sim_fns.kl_similarity(adm_msr.kde, target_msr.kde, kde_sample_size)
            js_dist = sim_fns.js_similarity(adm_msr.kde, target_msr.kde, kde_sample_size)
            match(metric):
                case "HD":
                    kdmaAlignment.kdma_alignments[target_msr.kdma_id] = h_dist
                case "KL":
                    kdmaAlignment.kdma_alignments[target_msr.kdma_id] = kl_dist
                case "JS":
                    kdmaAlignment.kdma_alignments[target_msr.kdma_id] = js_dist
                case _:
                    raise ValueError(f'Invalid algorithm name: {metric}')
        # this is a rollup of the overall alignment of the ADM to this single RDM
        mean = sum(kdmaAlignment.kdma_alignments.values()) / len(kdmaAlignment.kdma_alignments)
        mean_list.append(mean)
        rdmAlignment = RDMAlignment(rdm_id=id, individual_alignment=mean, alignment_detail=kdmaAlignment)
        rdm_alignments_list.append(rdmAlignment)
    
    # this is a rollup of the alignment of the ADM to the entire alignment target
    overall_mean = sum(mean_list) / len(mean_list)
    keys = []
    for at in alignment_target.target.keys():
        keys.append(at)
    # print(f"Alignment Package: {kdma_profile.dm_id}, {keys}, {metric}, {kde_sample_size}")
    return AlignmentPackage(overall_alignment=overall_mean, rdm_alignments=rdm_alignments_list, aligner_id=kdma_profile.dm_id, aligner_profile=kdma_profile, alignment_target=alignment_target)
