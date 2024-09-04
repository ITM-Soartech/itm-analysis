"""
Misc utils for pipeline
TODO rename if you can think of a more descriptive name.
"""
import pandas as pd

# pandas infers some data types on load
# this can convert some UserID and ProbeID columns to numeric values instead of strings,
# causing problems with the pipeline, especially on joins
# Define a set of column names and datatypes that will be enforced where applicable
# in all_data elements
_COL_DTYPES = {
    # some of these are probably unecessary (ie session id should always be
    # interpreted as string anyways) but since errors from wrong dtypes are
    # annoying/tricky to debug, we err on the side of caution
    'UserID': str,
    'SessionID': str,
    'ScenarioID': str,
    'ProbeID': str,
    'ChoiceID': str,
}

def enforce_dtypes(df: pd.DataFrame) -> None:
    """
    Enforces standardized datatypes in dataframe columns.

    For example, ensures the UserID dtype is always string, 
    even if entries are strings of numeric characters. This avoids
    unexpected behavior when trying to join dataframes with matching
    columns that were interpreted differently (ie str(UserID) will not
    match int(UserID))

    Parameters
    -----------
    df, pd.DataFrame
    dataframe that will be modified inplace

    Returns
    -----------
    None
    Modifies input dataframe in place.
    """
    global _COL_DTYPES
    for col in df.columns:
        if col in _COL_DTYPES:
            df[col] = df[col].astype(_COL_DTYPES[col])


def mongo_id_to_timestamp(entry: str) -> float:
    """
    Convert timestamp embedded in mongodb _id object to float timestamp
    (seconds since epoch)

    Can be converted into nice datetime object with:
    .. code-block:: python

        from datetime import datetime, UTC
        time = datetime.fromtimestamp(seconds_since_epoch, UTC)


    Parameters
    -----------
    entry: str
        "_id" property of mongodb document
    
    Returns
    -----------
    timestamp: str
        formatted timestamp
    """
    # see https://stackoverflow.com/questions/67927913/python-method-to-
    #     convert-objectid-of-mongodb-as-string-to-timestamp
    return float(int(entry[:8], base=16))

def snake_to_pascal_case(text: str) -> str:
    """
    Convert snake_case text to PascalCase.
    
    Used to maintain consistentcy across all dataframes.

    Special case: ID is capitalized. For example,
    user_id is converted to UserID instead of UserId.
    
    Parameters
    ----------
    text: str
        input text
    
    Returns
    -------
    text_pc: str
        text in PascalCase
    """
    # snake_case has segments divided by underscores
    split = text.split('_')
    # make first letter of each segment capitalized
    split = [seg[0].upper() + seg[1:] for seg in split]
    # handle special case- segments starting with "Id"
    # are converted to "ID"
    new_split = []
    for seg in split:
        if len(seg) >= 2 and seg[:2] == "Id":
            seg = "ID" + seg[2:]
        new_split.append(seg)
    # join list of segments back into single string
    text_pc = ''.join(new_split)
    return text_pc