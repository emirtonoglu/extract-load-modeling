import json
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder


def _connect_mongo(host, port, db):
    """
    This function returns mongodb connection
    :param host:string
    :param port: string
    :param db: string
    :return: connection
    """
    conn = MongoClient(host, port)
    return conn[db]


def create_covid_dataframe(df):
    """
    You can use this function to create covid dataframe after reading from mongodb. Because some days don't contains
    all columns from column_list.txt file. So If your model can get error because of absent columns.
    :param df: Pandas dataframe which is initialized from mongodb and con have absent columns
    :return: new dataframe with all columns.
    """
    with open('model/column_list.txt', 'r') as file:
        columns = [column.strip() for column in file.readlines()]
    empty_df = pd.DataFrame(columns=columns)
    return df.reindex(columns=empty_df.columns)


def read_mongo(db, collection, query={}, host='localhost', port=27017, no_id=True):
    """
    This function read data from mongodb collection. Also you can give query parameter which benefits to query filtering
    data. Mongodb collection's reside as json. When you read data from mongodb to python, It will be converted to
    dictionary. So you should convert it from dictionary to pandas dataframe. Also If you want, you can add a condition 
    to delete "_id" column. 
    :param db: Database name in mongodb
    :param collection: Collection name in mongodb database
    :param query: Mongodb query to filter in dictionary type
    :param host: mongodb server ip/host. Default "localhost"
    :param port: mongodb server port. Default "27017"
    :param no_id: mongodb returns data with "_id" column. If you think that It is not necessary, you can set True.
    Default true.
    :return: Pandas Dataframe
    """
    db = _connect_mongo(host=host, port=port, db=db)
    coll = db.client[db.name][collection]
    cursor = coll.find(query)
    documents = list(cursor)
    df = pd.DataFrame(documents)

    return df


def write_mongo(db, collection, score, host='localhost', port=27017):
    """
    This function write pandas dataframe to mongodb.
    :param db: Database name in mongodb
    :param collection: Collection name in mongodb database
    :param score: Pandas dataframe which contains scores and input variables
    :param host: mongodb server ip/host. Default "localhost"
    :param port: mongodb server port. Default "27017"
    """

    db = _connect_mongo(host=host, port=port, db=db)
    coll = db.client[db.name][collection]
    records = score.to_dict(orient='records')
    records  = json.dumps(records)
    coll.insert_many(str(records))


def feature_formatting(df):
    """
    In this function you should make some changes on columns. First, you should change date column type to datetime.
    After that, non-categorical columns types can be updated like converting float.

    Example
    -------
    When you convert dictionary to pandas dataframe, Column types can be like below
    22  stringency_index                 171 non-null    object
    23  total_cases                      193 non-null    object
    24  date                             194 non-null    object

    Your new type should be similar below.

    22  stringency_index                 171 non-null    float64
    23  total_cases                      193 non-null    float64
    24  date                             194 non-null    datetime64[ns]

    :param df: Dataframe
    :return:
    """
    df = df[['continent', 'location', 'date', 'total_cases', 'new_cases',
             'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed',
             'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
             'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million',
             'reproduction_rate', 'icu_patients', 'icu_patients_per_million', 'hosp_patients',
             'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
             'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
             'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed',
             'new_tests_smoothed_per_thousand',
             'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated',
             'people_fully_vaccinated',
             'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred',
             'people_vaccinated_per_hundred',
             'people_fully_vaccinated_per_hundred', 'new_vaccinations_smoothed_per_million', 'stringency_index',
             'population',
             'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita',
             'cardiovasc_death_rate', 'diabetes_prevalence', 'handwashing_facilities',
             'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']]
    for i in df.loc[:, df.columns != 'tests_units'].columns[3:]:
        df[i] = df[i].astype(float)
    df.replace({None: 0}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index('date', inplace=True)
    return df

def feature_generation(df):
    """
    In this function you will generate your new columns which are evaluated in modeling phase. Which means that if
    you need new columns for your model. You can generate them here.

    Hint**: If you generate avg, sum, etc. columns for a couple of weeks or days for your model, you should read required
    date range. For example, If we have average of last 3 day, you should read 27, 28, 29th days from mongodb to score
    30th day. Be careful about that.

    :param df: Pandas Dataframe
    :return:
    """
    df.reset_index(inplace=True)

    df['new_cases_avg_3g'] = df.groupby('location')['new_cases'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)  # Last 3 days avg
    df['daily_death_ratio'] = round(df['new_deaths'] / df['total_deaths'], 5)
    df['total_cases/total_tests'] = round(df['total_cases'] / df['total_tests'], 5)
    df['hosp_patient_ratio'] = round(df['hosp_patients'] / df['total_cases'], 5)
    df['icu_patient_ratio'] = round(df['icu_patients'] / df['hosp_patients'], 5)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def label_encoding(df):
    """
    This function converts char columns to numeric values. This can be helpful to run models.

    Example
    -------
    continent     location       date
      Asia       Afghanistan     2021-05-01
      Asia       Afghanistan     2021-05-02
      Asia       Afghanistan     2021-05-03
      Asia       Afghanistan     2021-05-04


   continent_num  location_num   date
           2             0       2021-05-01
           2             0       2021-05-02
           2             0       2021-05-03

    :param df: Pandas dataframe
    :return: Labeled Pandas Dataframe
    """
    df = df[df['continent'] != 0]
    encoder_continent = LabelEncoder()
    encoder_location = LabelEncoder()

    encoder_continent.fit(df['continent'])
    encoder_location.fit(df['location'])

    df['continent_num'] = encoder_continent.transform(df['continent'])
    df['location_num'] = encoder_location.transform(df['location'])
    df = df.drop(columns=['location', 'continent'])
    df = pd.concat([df, pd.get_dummies(df.tests_units)], 1).drop(columns=['tests_units'])

    return df


def get_scores(df):
    """
    This function will generates your scores. You will read your model from pickle file and them apply it onto the
    dataframe to get next day covid case number. It will return new dataframe with score column.
    :param df: Input data for model as dataframe
    :return: Output data with scores as dataframe
    """
    with open('model/finalized_model.pickle', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(df)
    df['score'] = predictions

    return df



if __name__ == '__main__':
    """
    This your main function and flow will be here. 
    1. Read country info and covid data from mongodb. These data should be loaded previously with ETL.
    2. Merge these two dataset
    3. Apply Feature formatting
    4. Apply Feature generation
    5. Apply Label Encoding
    6. Find Scores
    7. Write them to Mongodb
    """
    db = "CovidData"
    covid_collection = "CovidInfo"
    country_collection = "CountryInfo"
    new_collection = "TotalInfo"
    query = {"date": {"$gte": "2022-04-01", "$lte": "2022-11-07"}}
    df1 = read_mongo(db, covid_collection, query, host='localhost', port=27017, no_id=True)
    df1 = create_covid_dataframe(df1)
    df2 = read_mongo(db, country_collection, query={}, host='localhost', port=27017, no_id=True)
    df = pd.merge(df1, df2, on="location", how="left")
    df = feature_formatting(df)
    df = feature_generation(df)
    df = label_encoding(df)
    df.set_index('date', inplace=True)
    score = get_scores(df)
    print(df['score'].tail(7))
    write_mongo(db, new_collection, score, host='localhost', port=27017)




