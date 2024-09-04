"""
Collection of functions for preprocessing steps of the analysis pipeline. Computations
that generate anything other than the final outputs of the pipeline (metrics
and/or visualizations) can be implemented here.
"""

import numpy as np
import pandas as pd


def normalize_kdma_associations(
        df: pd.DataFrame,
        group_suffix: str,
        score_key: str = 'Score',
        safe: bool=True,
        inplace: bool = False,
        ) -> pd.DataFrame | None:
    """
    Given a dataframe containing probe kdma associations, normalize each association
    with a linear transform that makes the range of values in the dataframe go from 0
    to 1. If all associations are the same, the normalized values will be set to -1.

    The normalization is:
        normalized score = (scores-max(scores))/(max(scores)-min(scores))
        if max(scores) != min(scores), else normalized_score = -1

    Adds the following columns to the dataframe:
        {score_key}{group_suffix}Norm: normalized scores ranging from 0 to 1 (or -1
            if all scores are the same)
        {score_key}{group_suffix}Min: min value used for normalization
        {score_key}{group_suffix}Max: max value used for normalization
    

    Parameters
    -----------
    df: pd.DataFrame
        dataframe containing values to normalize
    
    group_suffix: str
        Suffix to use in generated columns. For ITM program metric pipeline,
        should follow PascalCase convention.
    
    score_key: str
        Key in df to retrieve kdma associations from
    
    inplace: bool
        if True, df is updated in place and the return of the function is None
        Otherwise, df is not updated, and 

    safe: bool
        if True, validation checks (such as making sure there is only 1 kdma present
        in the dataframe) are performed and an error is raised if the validation fails.
        if False, no checks are applied.


    Returns
    ------------
    df_processed: pd.DataFrame or None
        dataframe with added columns, or None if inplace==True
    
    Examples
    ------------
    >>> import pandas as pd
    >>> from itm.program_metrics_pipeline import preprocessing as pre
    >>> df = pd.DataFrame({'Score': [0.1, 0.3, 0.5, 0.7, 0.9]})
    >>> pre.normalize_kdma_associations(df, group_suffix='Global', inplace=True)
    >>> print(df)
    
       Score  ScoreGlobalNorm  ScoreGlobalMin  ScoreGlobalMax
    0    0.1             0.00             0.1             0.9
    1    0.3             0.25             0.1             0.9
    2    0.5             0.50             0.1             0.9
    3    0.7             0.75             0.1             0.9
    4    0.9             1.00             0.1             0.9

    """

    # validation checks
    if safe:
        if ('KDMA' in df.columns) and (len(df['KDMA'].unique()) > 1):
            # normalizing values from different kdmas does not make sense
            raise ValueError('Scores for multiple kdmas found in df')

    # columns added to dataframe after normalization        
    norm_score_key, norm_min_key, norm_max_key = [
        f'{score_key}{group_suffix}Norm', f'{score_key}{group_suffix}Min',
        f'{score_key}{group_suffix}Max'
    ]

    if not inplace: # prevent modifying input dataframe
        df = df.copy()


    vmin, vmax = (f(df[score_key]) for f in (np.min, np.max))
    
    if vmin == vmax:
        # cannot normalize by 0 range
        df[norm_score_key] = -1
    else:
        # linear transformation to scale ranging from 0 to 1
        df[norm_score_key] = (df[score_key] - vmin)/(vmax - vmin)
    
    # record min and max values used for normalization
    df[[norm_min_key, norm_max_key]] = vmin, vmax


    if inplace:
        return None
    
    return df
        
