# TODO clean up code/add docstrings/make formatting consistent with rest of 
# program metrics pipeline
# # Library calls for 2 feature KDE measurement, alignment, construct validity checks & visualization

# +
import numpy as np
import pandas as pd
import math
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import KernelDensity
from scipy.integrate import trapezoid

from itm.kdma_profile import kdma_measurement_from_sample
from itm.program_metrics_pipeline.preprocessing import normalize_kdma_associations
from itm_schema.ml_pipeline import KDMAMeasurement, KDMAProfile, KDMAId

from copy import deepcopy
from typing import Optional

# +
# 2 feature variants of the KDE creating & alignment functions. 
# Testing here, once they are working they would be moved to the python libs
# The expectation is that we may shift to n features once we figure out how many features per KDMA are useful. 

# So this isn't splattered all over the code as hardcoded numbers
# The default is the same bandwidth values used for the 1 feature KDE
def get_default_2feature_bandwidth(max_value=1.0):
    bandwidth = (max_value / 10) * 0.75 
    return bandwidth
    
def make_2feature_kde(X: list[float], Y: list[float], bandwidth, max_value=1.0):
    # Convert input lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Concatenate X and Y to form a 2D array where each row is (X[i], Y[i])
    data = np.column_stack((X, Y))

    # Fit Kernel Density Estimation
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data)

    return kde

def compute_2feature_kde_ess(X: list[float], Y: list[float], bandwidth, max_value=1.0):
    
    kde = make_2feature_kde(X, Y, bandwidth, max_value)

    # Convert input lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Concatenate X and Y to form a 2D array where each row is (X[i], Y[i])
    data = np.column_stack((X, Y))    

    # Evaluate the KDE on the data points
    log_densities = kde.score_samples(data)
    densities = np.exp(log_densities)

    # Compute weights as inverse of densities
    weights = 1.0 / densities

    # Normalize weights
    normalized_weights = weights / np.sum(weights)

    # Calculate ESS
    ess = 1.0 / np.sum(normalized_weights ** 2)
    
    return ess   

# As a starting point confidence is a weighted average (50/50) of 
# 1) percentage of the number of sufficient obsercvations needed. Default value of 50 was computed using Silverman's Rule + Bandwidth sensitivity analysis. See https://soartech.sharepoint.us/sites/DARPA-ITM/_layouts/OneNote.aspx?id=%2Fsites%2FDARPA-ITM%2FSiteAssets%2FDARPA%20ITM%20Notebook&wd=target%28Design.one%7C0C008ED3-2F55-4548-B538-F6BC2269FE28%2FEstimating%20Probe%20Sample%20Size%20Needed%7CB68E1739-B6BA-42A7-B4EA-E33DB620EE1F%2F%29 onenote:https://soartech.sharepoint.us/sites/DARPA-ITM/SiteAssets/DARPA%20ITM%20Notebook/Design.one#Estimating%20Probe%20Sample%20Size%20Needed&section-id={0C008ED3-2F55-4548-B538-F6BC2269FE28}&page-id={B68E1739-B6BA-42A7-B4EA-E33DB620EE1F}&end
# 2) percentage of the actual observations the ess achieves 
def compute_2feature_kde_confidence(ess, actual_observation_count, desired_observation_count=50,
                       ess_high_threshold=90, ess_low_threshold=50,
                       observation_count_high_threshold=90, observation_count_low_threshold=50,
                       ess_weight=.5, observation_count_weight=.5,
                       confidence_high_threshold=90, confidence_low_threshold=50):

    confidence_reasons=[]
    confidence_rating = 0

    #print(f"actual observations {actual_observation_count}")
    #print(f"desired_observations {desired_observation_count}")
    observation_percentage = (actual_observation_count / desired_observation_count) * 100
    #print(f"observation_percentage {observation_percentage}")
    if observation_percentage >= observation_count_high_threshold:
        confidence_reasons.append("Sufficient Observation Count")
    elif observation_percentage >= observation_count_low_threshold:
        confidence_reasons.append(f"Greater than {observation_count_low_threshold}% of Sufficient Observation Count")
    else:
        confidence_reasons.append(f"Less than {observation_count_low_threshold}% of Sufficient Observation Count")
                           
    ess_percentage = (ess / actual_observation_count) * 100

    #print(f"ess_percentage {ess_percentage}")  
    if ess_percentage >= ess_high_threshold:
        confidence_reasons.append("High ESS")
    elif ess_percentage >= ess_low_threshold:
        confidence_reasons.append("Medium ESS")
    else:
        confidence_reasons.append("Low ESS")

    # Now just a simple weighted average 
    confidence_percentage = ess_percentage*ess_weight + observation_percentage*observation_count_weight
    #print(confidence_percentage)
                           
    # convert back to a number between zero & 1 instead of a percentage                
    confidence_rating = confidence_percentage/100
    
    return confidence_rating, confidence_reasons  
                           
   
#The expectation is that X is the KDMA per probe scores rescaled to be relative to the global reference frame
#and Y is the KDMA per probe scores rescaled to be relative to the local reference frame
# For now I'm not propagating the default for computing confidence into this call's parameters yet so it will be easier to get this integrated into the server for the DRE. 
def kdma_measurement_from_sample_2feature(kdma_id: KDMAId, X: list[float], Y: list[float], bandwidth, max_value=1.0):
    """
    Generates a random KDMA Measurement based on a 
    normally distributed random sample
    
    The normal distribution is centered on `norm_loc` with a
    a scale of `norm_scale`
    """
    X = np.array(X)  # convert to numpy (if not already)

    kde = make_2feature_kde(X, Y, bandwidth, max_value)
    ess = compute_2feature_kde_ess(X, Y, bandwidth, max_value)
    confidence, confidence_reasons = compute_2feature_kde_confidence(ess, len(X))
    
    return KDMAMeasurement(
        kdma_id=kdma_id,
        value=-1,
        kde=kde, # TODO need to update this to newest schema version- ie use kdes: dict[str, KernelDensity]
        confidence=confidence,
        confidence_reasons=confidence_reasons,
        num_observations = len(X),
        kde_ess= ess,
        hist=None)

def _normalize_2feature(x, y, z):
    """
    Normalize 2D probability distribution z such that its integral over domain (x, y) is one.

    Parameters
    ----------
    x: ndarray
        domain over which discrete probability distribution z is defined (x-coordinates).
    
    y: ndarray
        domain over which discrete probability distribution z is defined (y-coordinates).

    z: ndarray
        2D probability distribution at each point in (x, y). z is proportional to the
        probability density of the distribution at (x, y).

    Returns
    --------
    pdf: ndarray
        array with same shape as z that gives normalized probability density function
        values at each point (x, y).
    """
    # Compute the area under the surface
    dx = x[1] - x[0]  # assuming uniform spacing
    dy = y[1] - y[0]  # assuming uniform spacing
    area = trapezoid(trapezoid(z, x, axis=0), y, axis=0)
    
    # Normalize z by the computed area
    pdf = z / area

    return pdf

def _kde_to_pdf_2feature(kde, grid_size=100, normalize=True):

    # Create a 2D grid of points within the normalized range [0, 1] x [0, 1]
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy_grid = np.vstack([X.ravel(), Y.ravel()]).T  

    # Evaluate the KDE on the xy_grid
    pf = np.exp(kde.score_samples(xy_grid))
    
    if normalize:
        # Reshape the grid and the probability values
        x_vals = np.linspace(0, 1, grid_size)
        y_vals = np.linspace(0, 1, grid_size)
        pf_reshaped = pf.reshape(grid_size, grid_size)
        
        # Normalize the 2D KDE values
        pf = _normalize_2feature(x_vals, y_vals, pf_reshaped).ravel()
        
    return pf

def js_similarity_2feature(kde1, kde2, grid_size=100):

    # Compute the PDFs of the two 3D KDEs on the 2D grid
    pdf_kde1 = _kde_to_pdf_2feature(kde1, grid_size)
    pdf_kde2 = _kde_to_pdf_2feature(kde2, grid_size)

    if np.allclose(pdf_kde1, pdf_kde2):
        # If two kdes are functionally identical but off by a 10 to the minus 6 or so floating point amount
        # jensenshannon can hit floating point roundoff problems and return a nan instead of a zero. 
        # This is not a 2D kde issue, it can happen in 1D as well. 
        # To see it happen in a simple 1D case these values will cause the error case
        # if they are used as the input to fitting two kdes to compare with this function
        # (6 zeros and 6 1s in each array, but not at the same array positions)
        #kde1_data_values = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
        #kde2_data_values = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        # To avoid introducing nans by hitting this case, we'll set very close to zero cases to zero. 
        print("KDE1 and KDE2 are so close to identical Jensenshannon can produce a nan. Assigning them to be identical.")
        js = 0.0
    else:
        js = jensenshannon(pdf_kde1, pdf_kde2) 

    # We invert the value because the spec agreed to with the other ITM performs has
    # 0 = unaligned, 1 = full aligned which is the opposite of what Jensenshannon produces. 
    return 1 - js


# -

# The scenarios passed in are considered the reference frame
def normalizeProbeDataToReferenceFrame(all_data, scenarios):
    print(f"Normalizing probe data to reference frame scenarios {scenarios}")
    
    # Filter to the scenarios selected (consider them the reference frame)
    probe_data = all_data['probes'].set_index('ScenarioID').loc[scenarios].reset_index()
    
    # normalize kdma associations with respect to reference frame
    probe_data = pd.concat(
        [
            normalize_kdma_associations(x, 'RefFrame')
            for _, x in probe_data.groupby('KDMA')
        ],
        ignore_index=True
    )
    #probe_data.head(1)    
    return probe_data


# +
# Get the fingerprint data for each of the specified individuals
# Note, only data in data tables is combined, so this function can be used to get the data for a list
# of ids that will be used to create profiles for individuals, or for a list of ids that will be later turned into a single population profile. 
# If the dropNans flag is true, when merging the session & probe data, if no probe data is found for an item in the session data, 
# that item is dropped so only valid rows are returned (not rows that have a Nan for score or KDMA). This
# happens when filtering has been applied prior to this call (e.g., such as selecting only a subset of probes to include in analysis). 
def getDMFingerprintData(probe_data, session_data, scenarios, dmIDs, dropNans = True):

    # Keep just the specified users
    try:
        print(session_data['UserID'].unique())
        session_data = session_data.set_index('UserID').loc[dmIDs].reset_index()
    except KeyError as e:
        print("KeyError: "+str(e))

    # Join the two parts of the data together into one DF
    eval_data = session_data.merge(probe_data,
        how='left', # TODO verify that after changing this to 'inner', dropNans block is not needed
                    # (but resulting dataframe is still the same)
        on=['ScenarioID','ProbeID','ChoiceID']
    )

    if(dropNans == True):
        # This happens if the probe data has been filtered so a fingerprint can be created using a subset of probes, regardless of 
        # how many probes the decision maker actually experienced in a session. 
        # The merge of the session & probe dataframes will inject NANs for all items in the session that 
        # are not in the probe data. We want to filter these out. We can do so easily by dropping the rows without a score (which comes from the probe data if found). 
        
        # Drop rows where either Score or KDMA are NaN (if one but not both is NaN it is a data entry error when coding in the probes). I'm not explicitly checking for that
        # edge case now, but we could in future. Right now I just drop those from consideration. 
        eval_data = eval_data.dropna(subset=['Score', 'KDMA'])

    return eval_data
    
# TODO applies to both getIndividualProfilesInReferenceFrame() and
# getPopulationProfileInReferenceFrame: The keys for KDEs are
#   - globalnormx_localnormy
#   - globalnorm
#   - localnorm
#   - rawscores
#    See the start in getIndividualProfilesInReferenceFrame() for details

# Get the individual KDMA profiles for each UserID present in the data that is passed in
# Assumes the data has already been filtered to only the observations that should
# be included in the profile. 
def getIndividualProfilesInReferenceFrame(eval_data, kdma_id: KDMAId, bandwidth) -> dict[str, KDMAProfile]:
    print("Creating profiles")

    profiles = {}
    for user_id, subdf in eval_data.groupby('UserID'):
        
        kdma_measurements = {
            # We don't want the truely global norm that includes all scenarios.
            # We want the norm that considers a subset of the scenarios to be the 
            # "global space". That value is stored in ScoreRefFrameNorm
            # If that does not exist when you get here, it is because you have not called
            # normalizeProbeDataToReferenceFrame to the set of scenarios you want to be
            # considered the "global space". The value stored in ScoreGlobalNorm 
            # considers all scenarios (including those you may not want) to be the global space.

            'globalnormx_localnormy': kdma_measurement_from_sample_2feature(
                kdma_id, subdf['ScoreRefFrameNorm'], subdf['ScoreLocalNorm'], bandwidth
            ),

            'globalnorm': kdma_measurement_from_sample(
                kdma_id, subdf['ScoreRefFrameNorm'], bandwidth
            ),

            'localnorm': kdma_measurement_from_sample(
                kdma_id, subdf['ScoreLocalNorm'], bandwidth
            ),

            'rawscores': kdma_measurement_from_sample(
                kdma_id, subdf['Score'], bandwidth
            ),
        }

        alignment_measurement = deepcopy(kdma_measurements['globalnormx_localnormy'])
        alignment_measurement.kdes = {k: v.kde for k,v in kdma_measurements.items()}
        alignment_measurement.kde = None

        profile = KDMAProfile(
            dm_id=user_id,
            kdma_measurements={kdma_id: alignment_measurement},
        )
        print(f"Creating profile for {user_id}")
        profiles[user_id] = profile
    print("Finished creating profiles")
    return profiles


# Get a population KDMA profiles that combines all UserIDs present in the data that is passed in
# Assumes the data has already been filtered to only the observations that should
# be included in the profile. 
# the passed in pop_id will be used as the identifier for the population profile
def getPopulationProfileInReferenceFrame(eval_data, kdma_id: KDMAId, bandwidth, pop_id):

    # We don't want the truely global norm that includes all scenarios. 
    # We want the norm that considers a subset of the scenarios to be the 
    # "global space". That value is stored in ScoreRefFrameNorm
    # If that does not exist when you get here, it is because you have not called
    # normalizeProbeDataToReferenceFrame to the set of scenarios you want to be
    # considered the "global space". The value stored in ScoreGlobalNorm 
    # considers all scenarios (including those you may not want) to be the global space.
    kdma_measurements = {
        'globalnormx_localnormy': kdma_measurement_from_sample_2feature(
            kdma_id, eval_data['ScoreRefFrameNorm'], eval_data['ScoreLocalNorm'], bandwidth
        ),
    
        'globalnorm': kdma_measurement_from_sample(
            kdma_id, eval_data['ScoreRefFrameNorm'], bandwidth
        ),

        'localnorm': kdma_measurement_from_sample(
            kdma_id, eval_data['ScoreLocalNorm'], bandwidth
        ),

        'rawscores': kdma_measurement_from_sample(
            kdma_id, eval_data['Score'], bandwidth
        ),
    }

    alignment_measurement = deepcopy(kdma_measurements['globalnormx_localnormy'])
    alignment_measurement.kdes = {k: v.kde for k,v in kdma_measurements.items()}
    alignment_measurement.kde = None

    print(f"Creating profile for {pop_id}")
    profile = KDMAProfile(
        dm_id=pop_id,
        kdma_measurements={kdma_id: alignment_measurement},
    )

    print("Finished creating profile")
    return profile

# +
from itertools import combinations
from itm.alignment.similarity_functions import js_similarity

# Note so far we only support computing confidence for the specified KDMA, not all KDMAs found in a profile. 
def compute_alignment_confidence_2feature(alignerProfile, targetProfile, kdma_id: KDMAId):
    
    # for now, all I'm doing is averaging the confidence values for the aligner/target
    confidence_mean = (alignerProfile.kdma_measurements[kdma_id].confidence + 
                       targetProfile.kdma_measurements[kdma_id].confidence) / 2

    return confidence_mean

 

# Compare similarity of 2 2Feature KDDEs. 
# This computes the alignment for each aligner/target pair found in the input arguments. 
def computeAlignment_2feature(
        alignerProfiles: dict[str, KDMAProfile],
        targetProfiles: dict[str, KDMAProfile],
        kdma_id: str,
        visualize = False,
        artifactBasePath = None,
        batchName = None,
        clipthreshold=0,
        blurthreshold=0,
        measurement_key='globalnormx_localnormy'):

    alignment_results = []  # List to store alignment results

    kdma_id = KDMAId(kdma_id)
    
    # Compute & visualize alignment of each aligner vs each target
    for ((a_u, a_p)) in alignerProfiles.items():
        for((t_u,t_p)) in targetProfiles.items():
            
            if (a_p.kdma_measurements[kdma_id].kdes is None or
                t_p.kdma_measurements[kdma_id].kdes is None):
                raise ValueError("Alignment must be computed using KDEs")

            # print('---')
            # print(a_p.kdma_measurements[kdma_id])
            # print('---')

            if (measurement_key not in a_p.kdma_measurements[kdma_id].kdes or
                measurement_key not in t_p.kdma_measurements[kdma_id].kdes):
                raise ValueError(f"Alignment missing measurement type {measurement_key}")

            a_kde = a_p.kdma_measurements[kdma_id].kdes[measurement_key]
            t_kde = t_p.kdma_measurements[kdma_id].kdes[measurement_key]
    
            alignment = js_similarity_2feature(a_kde, t_kde, grid_size=100)

            confidence = compute_alignment_confidence_2feature(a_p, t_p, kdma_id)
            
            t_confidence_reasons_str = "; ".join(t_p.kdma_measurements[kdma_id].confidence_reasons)
            a_confidence_reasons_str = "; ".join(a_p.kdma_measurements[kdma_id].confidence_reasons)
            confidence_reasons = f"Aligner({a_confidence_reasons_str}) Target({t_confidence_reasons_str})"
             
            # Store the result
            alignment_results.append({'Target': t_u, 'Aligner': a_u, f'Alignment_{kdma_id.value}': alignment, 
                    f'AlignmentConfidence_{kdma_id.value}': confidence, 
                    f'AlignmentConfidenceReasons_{kdma_id.value}': confidence_reasons,
                    f'TargetProfileConfidence_{kdma_id.value}': t_p.kdma_measurements[kdma_id].confidence,
                    f'AlignerProfileConfidence_{kdma_id.value}': a_p.kdma_measurements[kdma_id].confidence,
                    f'TargetESS_{kdma_id.value}': f'{t_p.kdma_measurements[kdma_id].kde_ess}',
                    f'AlignerESS_{kdma_id.value}': f'{a_p.kdma_measurements[kdma_id].kde_ess}',
                    f'TargetNumObservations_{kdma_id.value}': t_p.kdma_measurements[kdma_id].num_observations,
                    f'AlignerNumObservations_{kdma_id.value}': a_p.kdma_measurements[kdma_id].num_observations,
                 })

            if visualize == True:
                print(f"{a_u} V {t_u} = {alignment}")
                visualize_fingerprint_alignment(
                    a_u, t_u,
                    a_kde, t_kde,
                    kdma_id,
                    alignment,
                    path=get_alignment_path(artifactBasePath, batchName), 
                    grid_size=100,
                    batchName=batchName,
                    clipthreshold=clipthreshold, blurthreshold=blurthreshold)

    
    alignment_df = pd.DataFrame(alignment_results)
    print("Finished with alignment")
    return alignment_df



# +
# Visualization functions for visualizing 2 Feature KDEs & KDE alignments
# After these are solidified, they would also be moved to the python libs and out of the jupyter notebook
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap
import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")

def get_base_path(basepath, batchName):
    path = f"{basepath}batch_{batchName}/"
    create_folder_if_not_exists(path)
    return path
    
def get_aligner_path(basepath, batchName, folder="Aligner"):
    path = f"{basepath}batch_{batchName}/{folder}/"
    create_folder_if_not_exists(path)
    return path

def get_target_path(basepath, batchName, folder="Target"):
    path = f"{basepath}batch_{batchName}/{folder}/"
    create_folder_if_not_exists(path)
    return path

def get_target_path_subfolder(path, targetName):
    path = f"{path}/{targetName}/"
    create_folder_if_not_exists(path)
    return path

def get_alignment_path(basepath, batchName, folder="Alignment"):
    path = f"{basepath}batch_{batchName}/{folder}/"
    create_folder_if_not_exists(path)
    return path    
    
def visualize_fingerprint_terrain(path, dmName, profile, kdma_id, grid_size=100, batchName="", key='globalnormx_localnormy'):
    
    kde_model = profile.kdma_measurements[kdma_id].kdes[key]
    kde_ess = profile.kdma_measurements[kdma_id].kde_ess
    num_observations = profile.kdma_measurements[kdma_id].num_observations

    filename = path+"fingerprint_3D_"+dmName+".png"
    title = f"{dmName} ESS ({kde_ess:.2f}) Observations ({num_observations:.2f})\nBatch({batchName})"
    # Create a grid of points within range [0, 1] 
    # Still need this for the axis labeling
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate the KDE model at the grid points
    kde_values = _kde_to_pdf_2feature(kde_model, grid_size, normalize=True).reshape(grid_size, grid_size)

    # Plotting the normalized KDE as a terrain-like surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z_normalized, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, kde_values, cmap='viridis', edgecolor='none')

    ax.set_title(title)
    ax.set_xlabel(f'{kdma_id} (Across Probes)')
    ax.set_ylabel(f'{kdma_id} (Within Probes)')
    ax.set_zlabel('Alignment Probability Density')

    plt.savefig(filename, bbox_inches='tight', dpi=600)
    print("Wrote " + filename)
    plt.close()

def visualize_fingerprint_heatmap(path, dmName, profile, kdma_id, grid_size=100, batchName="", key='globalnormx_localnormy'):

    kde_model = profile.kdma_measurements[kdma_id].kdes[key]
    kde_ess = profile.kdma_measurements[kdma_id].kde_ess
    num_observations = profile.kdma_measurements[kdma_id].num_observations
    
    filename = path+"fingerprint_Heatmap_"+dmName+".png"
    title = f"{dmName} ESS ({kde_ess:.2f}) Observations ({num_observations:.2f})\nBatch({batchName})"

    pdf_values = _kde_to_pdf_2feature(kde_model, grid_size, normalize=True).reshape(grid_size, grid_size)

    plt.figure(figsize=(8, 6))
    plt.imshow(pdf_values, extent=(0, 1, 0, 1), cmap='viridis', origin='lower')
    plt.colorbar(label='Density')
    plt.title(title)
    plt.xlabel(f'{kdma_id} (Across Probes)')
    plt.ylabel(f'{kdma_id} (Within Probes)')
    
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    print("Wrote " + filename)
    plt.close()

# Valid colorscheme names are Aligner and Target
def visualize_fingerprint_contour(
        dmName,
        profile,
        kdma_id,
        ax=None,
        path=None,
        grid_size=100,
        batchName="",
        colorScheme="Aligner",
        key='globalnormx_localnormy'):

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    kde_model = profile.kdma_measurements[kdma_id].kdes[key]
    kde_ess = profile.kdma_measurements[kdma_id].kde_ess
    num_observations = profile.kdma_measurements[kdma_id].num_observations
    
    title = f"{dmName} ESS ({kde_ess:.2f}) Observations ({num_observations:.2f})\nBatch({batchName})"

    pdf_kde = _kde_to_pdf_2feature(kde_model, grid_size, normalize=True).reshape(grid_size, grid_size)

    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Set background color to grey
    ax.set_facecolor('grey')
    
    if colorScheme == "Target":
        plt.contour(X, Y, pdf_kde, levels=10, cmap='cool', alpha=1.0)
    else:
        plt.contour(X, Y, pdf_kde, levels=10, cmap='Wistia', alpha=1.0)    
        
    plt.title(title)
    plt.xlabel(f'{kdma_id} (Across Probes)')
    plt.ylabel(f'{kdma_id} (Within Probes)')

    if path is not None:
        filename = path+"fingerprint_2D_"+dmName+".png"
        plt.savefig(filename, bbox_inches='tight', dpi=600)
        print("Wrote " + filename)
        plt.close()

    return ax


def visualize_fingerprint_alignment(
        alignerName,
        targetName,
        aligner_kde_model,
        target_kde_model,
        kdma_id,
        alignment_score,
        ax=None,
        path=None,
        grid_size=100,
        batchName:Optional[str]="",
        clipthreshold = 0,
        blurthreshold=0):

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    title = f"{alignerName} V {targetName}"
    if batchName != "":
        title += f" Batch({batchName})"

    # This is so I get target - aligner
    target_kde= target_kde_model
    aligner_kde = aligner_kde_model

    pdf_target_kde = _kde_to_pdf_2feature(target_kde, grid_size, normalize=True).reshape(grid_size, grid_size)
    pdf_aligner_kde = _kde_to_pdf_2feature(aligner_kde, grid_size, normalize=True).reshape(grid_size, grid_size)

    # Create a custom colormap for the heatmap
    # Yellow = aligner, Cyan = target
    colors = [(0, 'yellow'), (0.5, 'black'), (1, 'cyan')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    

    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    ax.contour(X, Y, pdf_target_kde, levels=10, cmap='cool', alpha=1.0)
    ax.contour(X, Y, pdf_aligner_kde, levels=10, cmap='Wistia', alpha=1.0)    
    

    # Add a color overlay that turns the difference into a heatmap
    
    difference = pdf_target_kde - pdf_aligner_kde
    min_difference = np.min(difference)
    #print(min_difference)
    max_difference = np.max(difference)
    #print(max_difference)    
        
    # Rescale the difference to the range [-1, 1] so it will be the same for all images
    normalized_difference = 2 * ((difference - min_difference) / (max_difference - min_difference)) - 1

    # Adding a threshold so I can mute the "noise" of coloring the "flat" area of the terrain as "different". If it is an area without
    # peaks in both pdfs, we want it to be grey. But because it is a density function, there are differences that
    # are not meaningful to color. That is the "plateau" of each image has a different hight that isn't meaningful to call attention to.
    # So clip the "plateaus" to be grey. 
    # Apply the clip threshold
    clipped_difference = np.copy(normalized_difference)
    clipped_difference[np.abs(normalized_difference) < clipthreshold] = 0

    # Then to keep it from looking pixelated soften the edges on the clipping boundary a bit
    # Apply the blur threshold to soften the edges
    blur_mask = (np.abs(normalized_difference) >= clipthreshold) & (np.abs(normalized_difference) < (clipthreshold + blurthreshold))
    soften_factor = (np.abs(normalized_difference[blur_mask]) - clipthreshold) / blurthreshold
    clipped_difference[blur_mask] *= soften_factor
    
    ax.imshow(clipped_difference, extent=(0, 1, 0, 1), origin='lower', cmap=custom_cmap, alpha=0.5)
    
    #plt.colorbar(ax=ax, label='Difference in Density')
    
    ax.set_title(title)
    ax.set_xlabel(f'{kdma_id} (Across Probes)')
    ax.set_ylabel(f'{kdma_id} (Within Probes)')

    # Overlay alignment score in the center of the image
    ax.text(0.95, 0.05, f'{alignment_score:.2f}', fontsize=72, ha='right', va='bottom', color='white', bbox=dict(facecolor='black', alpha=0.5))

    if path is not None:
        filename = get_target_path_subfolder(path, targetName)+"fingerprint_alignment_"+alignerName+"_V_"+targetName+".png"
        plt.savefig(filename, bbox_inches='tight', dpi=600)
        print("Wrote " + filename)
        plt.close()


