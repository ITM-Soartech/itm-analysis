# TODO getting all data from mongo (ie for ta1 server applications) is really
# inefficient- queries should be optimized to only include scenario/user/session
# data relevant to the current call.
import json
from os import PathLike
import pandas as pd
from pathlib import Path
from pymongo.database import Database
from itm.program_metrics_pipeline.preprocessing import normalize_kdma_associations
from itm.program_metrics_pipeline.utils import (
    enforce_dtypes,
    mongo_id_to_timestamp,
    snake_to_pascal_case
)


from typing import Iterable, Optional
import yaml


def get_dm_group_df(path: PathLike) -> pd.DataFrame:
    """
    Load decision maker groups from yaml file into a dataframe.

    Decision maker groups are collections of user IDs used for the analysis.

    The yaml file should be structured as a dictionary that maps
    a string group name to a list[string] of UserIDs of members of the group.
    For example:
    {
        'ADMs': ['adm-1','adm-2'],
        'RDMs': ['user1','user2'],
        'Risk_Takers': ['adm-1','user1']
    }

    The resulting dataframe has 2 columns- 'UserID' and 'GroupID'. Each row contains
    a single UserID/GroupID association.

    For example:
       'UserID' | 'GroupID'
       --------------------
       'adm-1'  | 'ADMs'
       'adm-1'  | 'Risk_Takers'
       'user1'  | 'RDMs'
        ...     |  ...

    Parameters
    ----------
    path: PathLike
        path to yaml file containing group info
    
    Returns
    ---------
    df_groups: DataFrame
        dataframe containing columns for UserID (str) and GroupID (str)
    """
    # parse yaml into dict
    dm_groups = yaml.safe_load(Path(path).read_text())

    # parse each group as a separate dataframe
    tmp = []
    for group_id,user_ids in dm_groups.items():
        df_tmp = pd.DataFrame({'UserID': user_ids})
        df_tmp['GroupID'] = group_id
        tmp.append(df_tmp)
    
    # concatenate all group dataframes together to get final dataframe
    df_groups = pd.concat(tmp, ignore_index=True)
    
    return df_groups


def get_dm_response_df(
        path: Optional[PathLike] = None,
        db: Optional[Database] = None,
        user_ids: Optional[Iterable[str]] = None,
        session_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load probe responses from decision makers from file or mongodb instance
    and assemble into dataframe.

    Either path or db must be not None. If both are specified,
    path is used (probe response data is loaded from disk).

    The resulting dataframe has the following columns:
        UserID: string user ID of decision maker
        SessionID: string session ID associated with response
        ScenarioID: string scenario ID associated with response
        ProbeID: string Probe ID associated with response
        ChoiceID: string Choice ID associated with response
        Justification: string text explaining response
        Timestamp, SessionStartTimestamp: numeric, seconds since linux
            epoch when response was recorded or session was started
    
    Parameters
    ------------
    path: PathLike, optional
        PathLike, directory containing json files probe response data for each user
    
    db: pymongo Database, optional
        ta1 server-formatted database to load probe response data from
    
    user_ids, session_ids: Iterable[str], optional
        optional lists of user and session IDs to include data for.
        If set to None, all of the probe responses are loaded.
        If user_ids is not None, all sessions for each included user are
        added to the response dataframe.
        If session_ids is not None, all specified sessions are added.
    
    Returns
    -------------
    response_df: DataFrame
        Dataframe of probe response data in the format described above.
    """
    
    if path is not None: # simply load data from files
        response_df = pd.concat(
            # keep_default_dates=False to load timestamps as numeric instead
            # of datetime (ie avoid having to deal with timezone issues)
            [pd.read_json(f, keep_default_dates=False)
            for f in sorted(Path(path).glob('*.json'))],
            ignore_index=True
        )
    
    elif db is not None: # load data from mongo
        # start by loading session data- fewer columns
        # session data- contains userID-sessionID pairings and session start time
        session_data = list(db['sessions'].find())

        session_df = pd.DataFrame(session_data)

        session_df['SessionStartTimestamp'] = session_df['_id'].apply(
            lambda x: mongo_id_to_timestamp(str(x))
        )
        # mongo's _id field no longer needed after getting timestamp
        # format columns as PascalCase for consistency among dataframes
        session_df = session_df.drop(columns='_id')
        session_df.rename(
            columns={c: snake_to_pascal_case(c) for c in session_df.columns},
            inplace=True
        )

        # if specified, generate list of sessions to keep 
        if user_ids or session_ids:
            include_sessions = set()
            if user_ids:
                include_sessions = include_sessions.union(
                    session_df.set_index('UserID').loc[list(user_ids), 'SessionID']
                )
            if session_ids:
                include_sessions = include_sessions.union(
                    session_ids
                )

            # remove unwanted sessions to avoid unecessary processing
            session_df = session_df.set_index('SessionID').loc[
                include_sessions
            ].reset_index()


        # now, load actual user responses
        response_df = pd.DataFrame(list(db['probe_responses'].find()))
        # use timestamp derived from mongodb _id instead of the saved
        # timestamp so it is in same format as timestamp from sessions
        # (ie already in seconds from epoch, no timezone issues)
        response_df['Timestamp'] = response_df['_id'].apply(
            lambda x: mongo_id_to_timestamp(str(x))
        ).astype(float)
        response_df = response_df.drop(columns='_id')

        print(f'{response_df.columns=}')
        # column names need to be the same so we can join to session_df
        response_df.rename(
            columns={
                col: snake_to_pascal_case(col) for col in response_df.columns
            } | {'choice': 'ChoiceID'},
            inplace=True
        )

        if user_ids or session_ids: # apply optional filter
            response_df = response_df.set_index('SessionID').loc[
                include_sessions
            ].reset_index()
        
        # merge session_df onto response_df to associate UserID and start
        # timestamp with responses
        response_df = response_df.merge(session_df, on=['SessionID'], how='inner')

        # order columns in convenient way
        cols = response_df.columns
        new_col_order = ['UserID',
            'SessionID',
            'ScenarioID',
            'ProbeID',
            'ChoiceID',
            'Timestamp'
        ]

        # add any additional columns (justification, any other future columns
        # that may get added)
        new_col_order.extend(
            set(cols).difference(new_col_order)
        )
        response_df = response_df[new_col_order]
    else:
        raise ValueError('path or db must be set')

    # ensure columns have consistent dtypes
    enforce_dtypes(response_df)
    return response_df


def get_qualtrics_df(survey_id: str, reload: bool=False) -> pd.DataFrame:
    """
    Load qualtrics responses into a formatted dataframe.

    The dataframe contains all raw responses from qualtrics surveys, as well as:
        maximization scores/variants generated from
            itm.data_pipeline.parse_qualtrics_for_maximization_score
            demographics schema objects generated from dgs_itm.get_demographics
        
    
    Parameters
    -------------
    survey_id: str
        survey id used to get qualtrics data from qualtrics.load_survey_data()
    
    reload: bool
        if True, qualtrics data will be downloaded (overwriting existing values)
        before being loaded into python.
    
    Returns
    -----------
    df_qualtrics: DataFrame
        Dataframe with all raw qualtrics survey responses and processed values
        (maximization scores, demographics, etc.)
    """
    # only import qualtrics_data if this function is called, so other data
    # can be retrieved in environments where qualtrics_data and 
    # related modules cannot be imported
    import itm.qualtrics_data as qualtrics
    from itm import demographics as dgs_itm
    from itm.data_pipeline import parse_qualtrics_for_maximization_score
    if reload:
        qualtrics.reload()
    
    # dataframe of responses to each qualtrics survey question
    raw_responses = qualtrics.load_survey_data(survey_id, filter_incomplete=False)
    
    # process responses to compute maximization scorres
    df_maximization = parse_qualtrics_for_maximization_score(raw_responses)

    # remove duplicate columns that are in both maximization_responses and
    # raw_responses before merging them together
    merge_cols = ['UserID'] + list(
        raw_responses.columns.difference(df_maximization.columns)
    )
    
    # extract demographics object
    # note- some fields in demographics are arrays, and cannot be fit into a single
    # row after unflattening the data structure. For now, just put the whole 
    # demographic object into the dataframe
    split = qualtrics.split_surveys(raw_responses)
    # TODO demographics cannot conveniently be unpacked into a dataframe because
    # some of the entries are arrays. However, it looks like the arrays from our
    # current data collect are all length 1, so they could be unpacked. I'm leaving
    # this for now, in case we anticipate having multiple entries in these arrays,
    # but if we don't expect this to be the case, we should change from arrays
    # to single fields and then unpack into a dataframe.
    demographics = dgs_itm.get_demographics(split['demographics'])
    df_demographics = pd.DataFrame(
        [
            {'UserID': k, 'Demographics': v }
            for k, v in demographics.population.items()
        ]
    )

    # concatenate all qualtrics info into a single dataframe
    df_qualtrics = df_maximization.merge(
        raw_responses[merge_cols], how='left', on='UserID'
    )
    df_qualtrics = df_qualtrics.merge(df_demographics, how='left', on='UserID')

    # ensure column dtypes are consistent for merging
    enforce_dtypes(df_qualtrics)
    return df_qualtrics


def get_scenario_and_probe_dfs(
        path: Optional[PathLike] = None,
        db: Optional[Database] = None,
        use_schema=False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads scenario data and extracts dataframes containing scenario and probe response
    association data.

    Either path or db must be not None. If both are specified,
    path is used (scenarios are loaded from disk).

    The first dataframe (df_scenarios) has two columns, one for the string scenario ID
    and one for the scenario schema object, or dictionary of scenario data, if 
    use_schema==False.

    The second (df_probes) has the following columns:
        ScenarioID, ProbeID, ChoiceID, KDMA: string names
            for scenario/probe/choice/kdma IDS
        
        SceneID: integer index of scene in scenario
        ActionType: string action type associated with probe response
        Score: float KDMA score associated with probe response.
    
    Parameters
    -----------
    path: PathLike, optional
        PathLike, directory containing json files probe response data for each scenario
    
    db: pymongo Database, optional
        ta1 server-formatted database to load scenario data from
    
    use_schema: bool
        default=False
        If True, scenarios are loaded as schema objects, which are generally more 
        convenient to work with.
        Otherwise, scenarios are loaded as dicts, which removes any dependence
        on the schema requirements (some field names, required fields, enums, etc),
        other than the general structure of retrieving probe response kdma associations.
    
    Returns
    -----------
    df_scenarios, df_probes: pd.DataFrame
        dataframes with the format described above.
    
    """
    # TODO replace with plain json load
    # load scenarios as dict, and only convert to schema object at the end if use_
    #scenario_data = ta1_data.load_scenarios_from_directory(path)
    
    # maps string scenario id to dictionary of scenario data
    if path: # simply load from disk
        scenario_data: dict[str, dict] = {}
        for p in sorted(Path(path).glob('*.json')):
            data = json.loads(p.read_text())
            scenario_data[data['id']] = data
    else: # get from mongo
        scenario_data = {
            x['id']: x for x in db['scenarios'].find(projection={'_id': False})
        }
        

    
    # simply maps scenario id to the schema scenario object
    df_scenarios = pd.DataFrame(
        [{'ScenarioID': x['id'], 'Scenario': x} for x in scenario_data.values()]
    )
    if use_schema:
        # only load schema if specified to allow the pipeline to run
        # even when breaking schema changes are made
        from ta3_schema import Scenario
        df_scenarios['Scenario'] = df_scenarios['Scenario'].apply(
            Scenario.model_validate
        )
    
    # also create a dataframe that maps scenario/probe/choiceID/KDMA type to
    # the score associated for that KDMA with that choice
    probe_df_rows = []
    for scenario in scenario_data.values():
        for scene in scenario['scenes']:
            for action_map in scene['action_mapping']:
                # scene ends the scenario or otherwise has no probes
                if action_map is None or action_map['kdma_association'] is None:
                    continue
                for kdma_name, kdma_association in action_map[
                    'kdma_association'
                ].items():
                    # following conventions, df columns should be PascalCase

                    # TA3 renamed index to id, this supports both
                    if 'index' in scene:
                        scene_id = scene['index']
                    else:
                        scene_id = scene['id']

                    probe_df_rows.append(
                        {
                            'ScenarioID': scenario['id'],
                            'SceneID': scene_id,
                            'ProbeID': action_map['probe_id'],
                            'ChoiceID': action_map['choice'],
                            'ActionType': action_map['action_type'],
                            'KDMA': kdma_name,
                            'Score': kdma_association
                        }
                    )
    
    df_probes = pd.DataFrame(probe_df_rows)

    # add local/global normalization to probe kdma associations
    tmp = []
    for _, probe_df_global in df_probes.groupby('KDMA'):
        # global norms- scale kdma association scores from 0 to 1
        # based on the min and max values across all scenarios
        normalize_kdma_associations(probe_df_global, 'Global', inplace=True)

        # local norms- scale kdma association scores from 0 to 1
        # based on the min and max values within the same probe
        for _, probe_df_local in probe_df_global.groupby(
            ['ScenarioID', 'ProbeID']
        ):
            normalize_kdma_associations(probe_df_local, 'Local', inplace=True)
            tmp.append(probe_df_local)
    df_probes = pd.concat(tmp, ignore_index=True)
    
    # make sure columns are formatted correctly for joins
    returns = (df_probes, df_scenarios)
    for df in returns:
        enforce_dtypes(df)
    
    return returns

# TODO update docstring, include all params
# TODO add flag (default True) to only load the first complete set of respones
# for each scenario (ie handle edge case where users have multiple incomplete sessions)
# TODO add options to override defaults (ie pass in a path to a custom)
#     file of decision maker groups)
def get_all_data(
        path: Optional[PathLike] = None,
        db: Optional[Database] = None,
        qualtrics_survey_id: Optional[str]=None,
        load_user_groups: bool = True,
        load_sessions: bool = True,
        load_qualtrics: bool = True,
        load_scenario_data: bool = True,
        use_scenario_schema: bool = True
    ) -> dict[str, pd.DataFrame]:
    """
    Load all data used for ITM analysis pipeline.

    By default, all data is loaded. Optional parameters can be passed to exclude
    some of the data (ie load scenario data but not qualtrics data)

    The data is formatted as a dictionary mapping string keys to dataframe values.
    The dictionary is analogous to a sql database. Each key is like a table name,
    and the dataframe values are like tables. Keys should be in camelCase, and
    columns in the dataframe values should be in PascalCase.

    For more information on the content/format of the information in each dataframe,
    see the documentation for the associated  function used to retrieve the data
    (listed in the code below.)

    Parameters
    -----------
    path: PathLike
        Parent directory of cached data, should contain:
          - scenarios/: directory containing json files with data for each scenario
          - responses/: directory containing json files with data for each user
          - dm_groups.yaml: file that defines groups to use for pipeline analysis
        
    survey_id: str
        string qualtrics survey ID used to retrieve survey data.

    load_user_groups, load_session_data, load_qualtrics_data, load_scenario_data: bool
        Default True
        Flags for optionally disabling the loading of certain dataframes.
    
    use_scenario_schema: bool
        Default True
        If True, scenarios in all_data['scenarios'] will be loaded as
        ta3_schema.Scenario objects, which are generally more convenient to work with.
        
        Otherwise, scenarios are loaded as dictionaries. This allows for decoupling
        of the pipeline code with the schema. Note that the general structure for
        probe response kdma associations is still expected, but other schema changes
        such as adding or removing fields, changes to enums, etc will not cause errors.
    
    Returns
    -----------
    all_data: dict[str, pd.DataFrame]
        aggregated data in format described above.

    """
    # if path is defined, make sure it is a pathlib Path object
    path = None if path is None else Path(path)
    def _path_or_none(child_name: PathLike) -> Optional[Path]:
        # helper function to get subdirectories in get_all_data,
        # or return None if parent path is None (ie data loaded from mongo)
        nonlocal path # parent path
        return None if path is None else path / child_name

    all_data = {}
    
    # collections of UserIDs that are convenient to group together for analysis
    if load_user_groups:
        all_data['groups'] = get_dm_group_df(path=_path_or_none('dm_groups.yaml'))

    # qualtrics survey responses and aggregated data (ie maximization scores)
    if load_qualtrics:
        all_data['qualtrics'] = get_qualtrics_df(qualtrics_survey_id)
    
    # probe response data for users
    if load_sessions:
        all_data['sessions'] = get_dm_response_df(path=_path_or_none('responses'), db=db)

    if load_scenario_data:
        # kdma associations for all probes
        probe_df, scenario_df = get_scenario_and_probe_dfs(
            path=_path_or_none('scenarios'), db=db, use_schema=use_scenario_schema
        )
        all_data['probes'] = probe_df
        # maps scenario id to scenario
        all_data['scenarios'] = scenario_df
    
    return all_data
