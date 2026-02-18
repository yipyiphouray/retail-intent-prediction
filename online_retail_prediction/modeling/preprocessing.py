import pandas as pd
import numpy as np

"""Preprocessing module for retail intent prediction data."""

id_to_country = {
    1: "Australia",
    2: "Austria",
    3: "Belgium",
    4: "British Virgin Islands",
    5: "Cayman Islands",
    6: "Christmas Island",
    7: "Croatia",
    8: "Cyprus",
    9: "Czech Republic",
    10: "Denmark",
    11: "Estonia",
    12: "unidentified",
    13: "Faroe Islands",
    14: "Finland",
    15: "France",
    16: "Germany",
    17: "Greece",
    18: "Hungary",
    19: "Iceland",
    20: "India",
    21: "Ireland",
    22: "Italy",
    23: "Latvia",
    24: "Lithuania",
    25: "Luxembourg",
    26: "Mexico",
    27: "Netherlands",
    28: "Norway",
    29: "Poland",
    30: "Portugal",
    31: "Romania",
    32: "Russia",
    33: "San Marino",
    34: "Slovakia",
    35: "Slovenia",
    36: "Spain",
    37: "Sweden",
    38: "Switzerland",
    39: "Ukraine",
    40: "United Arab Emirates",
    41: "United Kingdom",
    42: "USA",
    43: "biz (*.biz)",
    44: "com (*.com)",
    45: "int (*.int)",
    46: "net (*.net)",
    47: "org (*.org)"
}

id_to_category = {
    1: "trousers",
    2: "skirts",
    3: "blouses",
    4: "sale"
}

id_to_color = {
    1: "beige",
    2: "black",
    3: "blue",
    4: "brown",
    5: "burgundy",
    6: "gray",
    7: "green",
    8: "navy blue",
    9: "of many colors",
    10: "olive",
    11: "pink",
    12: "red",
    13: "violet",
    14: "white"
}

id_to_position = {
    1: "top left",
    2: "top in the middle",
    3: "top right",
    4: "bottom left",
    5: "bottom in the middle",
    6: "bottom right"
}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess retail data for modeling.
    
    Args:
        df: Input DataFrame with raw retail data.
        
    Returns:
        Cleaned and preprocessed DataFrame.
    """
    
    # Drop 'year' column since it has only one unique value
    df.drop(columns=['year'], inplace=True)
    
    # month & day column in Integer: 4-8 & 1-31
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    
    # order column in Integer: 0-1
    df['order'] = df['order'].astype(int)
    
    # map country column to actual country names
    df['country'] = df['country'].map(id_to_country)
    # df = pd.get_dummies(df, columns=['country']) # create one hot encoding for the country column
    
    # session ID column in Integer
    df['session ID'] = df['session ID'].astype(int)
    
    # map main category to actual category 
    df['main_category'] = df['page 1 (main category)'].map(id_to_category)
    df.drop(columns = ['page 1 (main category)'], inplace = True)
    
    # Page 2 clothing model: 217 unique values, no mapping provided, so we will drop this column for now
    #df.drop(columns = ['page 2 (clothing model)'], inplace = True)
    
    # map color column to actual color names
    df['colour'] = df['colour'].astype(int)
    df['colour'] = df['colour'].map(id_to_color)
    
    # map model photography column to actual photography types
    df['model photography'] = df['model photography'].astype(int)
    df['model photography'] = df['model photography'].map({1: "en face", 2: "profile"})
    
    # convert price to int
    df['price'] = df['price'].astype(int)
    
    # rename price 2 to higher_than_average, 1 as True, 2 as false
    df['higher_than_average'] = df['price 2'].map({1: 1, 2: 0})
    df.drop(columns=['price 2'], inplace=True)
    
    # convert page to int
    df['page'] = df['page'].astype(int)
    return df


# main 
def main():
    # Example usage
    raw_data_path = "data/raw/e-shop clothing 2008.csv"
    df_raw = pd.read_csv(raw_data_path,sep = ";")
    
    df_processed = preprocess_data(df_raw)
    
    # Save processed data
    processed_data_path = "data/processed/e-shop clothing 2008.csv"
    df_processed.to_csv(processed_data_path, index=False)


if __name__ == "__main__":
    main()