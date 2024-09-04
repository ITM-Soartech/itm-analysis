from collections import defaultdict
import numpy as np
from sklearn.neighbors import KernelDensity

from ta3_schema.scenario import Scenario
from ta3_schema.scene import Scene

from itm_schema.ml_pipeline import KDMAMeasurement, KDMAProfile, KDMAId, SimpleHistogram
from itm_schema.pydantic_schema import KDMA, ProbeResponse

KDE_MAX_VALUE = 1.0 # Value ranges from 0 to 1.0
KDE_BANDWIDTH = 0.75 * (KDE_MAX_VALUE / 10.0)


# For each probe, we store a mappping of KDMA associations for each possible choice
ProbeId = str
ChoiceId = str
ProbeKDMAAssociations = dict[ChoiceId, dict[KDMAId, KDMA]]
ScenarioKDMAAssociations = dict[ProbeId, ProbeKDMAAssociations]

def probe_kdma_associations(scene: Scene, probe_id: ProbeId) -> ProbeKDMAAssociations:
    """
    For a given probe id in a scene, return a mapping og all kdma associations
    for all possible choices for this probe

    Returns
    -------
    ProbeKDMAAssociations
        A dictionary of kdma associations for each choice for this probe
    """
    kdma_associations: ProbeKDMAAssociations = defaultdict(dict)
    
    # Get KDMAs for this probe by checking each action
    for action in scene.action_mapping:
        # We only care about actions for this probe
        if action.probe_id == probe_id:
            if action.kdma_association:
                for kdma in action.kdma_association:
                    kdma_associations[action.choice][kdma] = (KDMA(
                        kdma=KDMAId(kdma),
                        value=action.kdma_association[kdma]))

    return dict(kdma_associations)


def scenario_kdma_associations(scenario: Scenario) -> dict[ProbeId, ProbeKDMAAssociations]:
    """
    Returns
    -------
    associations: ProbeKDMAAssociation
    
        Usage
            associations[probe_id][choice_id][kdma_id]
            
        For example
            associations['probe-1.1']['choice-0']['maximization']
    """
    probes: dict[ProbeId, ProbeKDMAAssociations] = {}

    for scene in scenario.scenes:
        # We only care about scenes that have probes
        if scene.probe_config:
            for probe in scene.probe_config:                                
                probes[probe.probe_id] = probe_kdma_associations(scene, probe.probe_id)
    return probes

#associations = scenario_kdma_associations(scenario)
#associations['probe-1.1']['choice-0']['maximization']



def kdma_measurement_from_sample(kdma_id: KDMAId, X: list[float], max_value=KDE_MAX_VALUE):
    """
    Generates a random KDMA Measurement based on a 
    normally distributed random sample
    
    The normal distribution is centered on `norm_loc` with a
    a scale of `norm_scale`
    """
    X = np.array(X) # convert to numpy (if not already)
        
    # Generate the histogram
    bin_values, bin_edges = np.histogram(X, bins=np.linspace(0, max_value, 10))
    hist = SimpleHistogram(
        bin_values=bin_values.tolist(), 
        bin_edges=bin_edges.tolist())

    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_BANDWIDTH).fit(X[:, np.newaxis])

    return KDMAMeasurement(
        kdma_id=kdma_id,
        value=X.mean(),
        kde=kde,
        hist=hist)


# this version works with decisions that are coming from a list of KDMA measurements that corresponds to
# outputs a KDMA profile
def create_mean_kdma_profile(dm_id: str, responses: list[ProbeResponse], associations: ScenarioKDMAAssociations) -> KDMAProfile:
    # Get the KDMA values for each response
    response_kdma_values: dict[KDMAId, list[float]] = defaultdict(list)
    for response in responses:
        response_kdma_associations = associations[response.response.probe_id][response.response.choice]
        for kdma_id, kdma_value in response_kdma_associations.items():
            response_kdma_values[kdma_id].append(kdma_value.value)

    kdma_measurements: dict[KDMAId, KDMAMeasurement] = {}
    for kdma_id, kdma_values in response_kdma_values.items():
        kdma_measurements[kdma_id] = kdma_measurement_from_sample(kdma_id, kdma_values)

    return KDMAProfile(
        dm_id=dm_id,
        kdma_measurements=kdma_measurements,
    )

#profile = create_mean_kdma_profile(session_id, responses, associations)
