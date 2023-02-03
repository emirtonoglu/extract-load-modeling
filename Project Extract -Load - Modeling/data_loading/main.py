from pyspark import SparkConf
from pyspark.sql import SparkSession
from extract import extract_covid_data
from transformation import transform_data
from load import load_data
from datetime import datetime
import argparse



def date_converter():
    """
    !!!DO NOT TOUCH!!!
    This function returns argument date parameter to datetime object.
    :param parser: parser
    :return: datetime
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--date', action='store', type=str, required=False)
    args = parser.parse_args()
    if args.date is not None:
        print("Date is {}".format(args.date))
        return datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        return None


def get_spark_utils():
    """
    !!!DO NOT TOUCH!!!
    This function returns spark context object and spark session object.
    These are the entry point into all functionality in Spark.
    :return: SparkContext, SparkSession
    """
    conf = SparkConf().setAppName("Covid"). \
        set("spark.executor.memory", "7G"). \
        set("spark.driver.memory", "7G"). \
        set("spark.driver.maxResultSize", "7G"). \
        set("spark.mongodb.input.uri", "mongodb://127.0.0.1"). \
        set("spark.mongodb.output.uri", "mongodb://127.0.0.1"). \
        set("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"). \
        set("spark.sql.debug.maxToStringFields", 1000)
    spark = SparkSession.builder.master("local[*]").config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    return sc, spark


if __name__ == '__main__':
    """
    This is your main function and this contains your flow. date_converter and get_spark_utils functions provide
    necessary variables for you like spark context, spark session and datatime variable for daily filter. You should 
    call extract, transform and load functions respectively from their modules.
    
    Hint**: You may convert extracted data to RDD after that convert it to Dataframe.
    
    """
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.json'
    datetime_date = date_converter()
    list_data = extract_covid_data(url)
    sc, spark = get_spark_utils()
    data = spark.createDataFrame(list_data)
    covid_df, country_df = transform_data(data, datetime_date=None)
    load_data(covid_df, country_df)


