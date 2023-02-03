from pyspark.sql.functions import map_concat, lit, create_map, explode, col, map_keys, desc


def generate_covid_info(raw_data_df):
    """
    This function extract data column which is dictionary from raw data and then convert it to new dataframe.
    You should also add location column to data column dictionary. It should be similar to like below.

    +-----------------------------------------------------------------------------------+
    |{date -> 2020-02-24, total_cases -> 1.0, new_cases -> 1.0, location -> Afghanistan}|
    |{date -> 2020-02-25, total_cases -> 1.0, new_cases -> 0.0, location -> Afghanistan}|
    |{date -> 2020-02-26, total_cases -> 1.0, new_cases -> 0.0, location -> Afghanistan}|
    |{date -> 2020-02-27, total_cases -> 1.0, new_cases -> 0.0, location -> Afghanistan}|
    |{date -> 2020-02-28, total_cases -> 1.0, new_cases -> 0.0, location -> Afghanistan}|
    +-----------------------------------------------------------------------------------+
    Hint**: You can use explode, map_concat, create_map and lit functions from spark.

    After this evaluation, you should convert it to dataframe. Actually If our dataset has a specific schema,
    we can do easily. But in the time, data column has been changed and new variables added to dictionary.
    So first of all you should find all key values from the above row maps and then you can use them to generate new
    column for our final dataframe. So the output something like that

    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |total_deaths|total_cases_per_million|new_cases_per_million|total_cases|      date|   location|
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |      2642.0|                 1549.0|                4.573|    60300.0|2021-05-03|Afghanistan|
    |    122554.0|               3415.529|                5.412|  4578852.0|2021-05-03|     Africa|
    |      2399.0|              45616.791|               13.205|   131276.0|2021-05-03|    Albania|
    |      3280.0|               2798.497|                4.447|   122717.0|2021-05-03|    Algeria|
    |       127.0|             172070.148|              168.252|    13295.0|2021-05-03|    Andorra|
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    Hint**: You might collect data to the driver using action functions. Also to create new column you can use
    col function with getItem function. You should evaluate column list with that approach and then you can
    extract column values from above row maps.

    :param raw_data_df: Dataframe
    :return: Dataframe
    """

    df = raw_data_df.select(explode(raw_data_df.data).alias("data"), "location")
    df = df.withColumn("location", create_map(lit("location"), col("location")))
    df = df.select(map_concat(df.data, df.location).alias("data"))
    df.show(truncate=False)

    keysDF = df.select(explode(map_keys(df.data))).distinct()
    keysList = keysDF.rdd.map(lambda x: x[0]).collect()
    keyCols = list(map(lambda x: col("data").getItem(x).alias(str(x)), keysList))
    df = df.select(*keyCols)
    df.sort(desc("total_deaths")).limit(10).show(truncate=False)

    return df


def generate_country_info(raw_data_df):
    """
    This function filters non data columns from the raw data.

    :param raw_data_df: Dataframe
    :return: Dataframe
    """
    raw_data_df = raw_data_df.filter(raw_data_df.location.isNotNull())
    return raw_data_df


def transform_data(covid_data_rdd, datetime_date=None):
    """
    This function converts your raw to new shape. Your raw data actually contains two important knowledge which are
    Country Info and Covid Info. You will use spark functions to evaluate these information and then generate two
    dataframe belonging to them. For all that you will create a condition according to date. Because for the first
    run you are going to make initial load to mongodb. After that you are going to load your data in daily runs.
    Only for initial load you can generate country info dataframe using generate_country_info function and generate
    covid info dataframe using generate_covid_info function. But It is not necessary to evaluate country info dataframe
    for daily runs. For daily runs also url returns you all data, so you should filter daily data according to datetime
    field.

    Example
    -------
    For Initial Load
    :input:
    +-------------+-------------+---------------------+---------+--------------------+
    |aged_65_older|aged_70_older|cardiovasc_death_rate|continent|                data|
    +-------------+-------------+---------------------+---------+--------------------+
    |        2.581|        1.337|              597.029|     Asia|[{date -> 2020-02...|
    |         null|         null|                 null|     null|[{date -> 2020-02...|
    |       13.188|        8.643|              304.195|   Europe|[{new_tests -> 8....|
    |        6.211|        3.857|              278.364|   Africa|[{date -> 2020-02...|
    |         null|         null|              109.135|   Europe|[{date -> 2020-03...|
    +-------------+-------------+---------------------+---------+--------------------+

    :output:
    Covid Info Dataframe
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |total_deaths|total_cases_per_million|new_cases_per_million|total_cases|      date|   location|
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |      2642.0|                 1549.0|                4.573|    60300.0|2020-12-24|Afghanistan|
    |    122554.0|               3415.529|                5.412|  4578852.0|2021-02-12|     Africa|
    |      2399.0|              45616.791|               13.205|   131276.0|2021-04-03|    Albania|
    |      3280.0|               2798.497|                4.447|   122717.0|2021-05-09|    Algeria|
    |       127.0|             172070.148|              168.252|    13295.0|2021-01-03|    Andorra|
    +------------+-----------------------+---------------------+-----------+----------+-----------+

    Country Info Dataframe
    +-------------+-------------+---------------------+---------+-------------------+--------------+
    |aged_65_older|aged_70_older|cardiovasc_death_rate|continent|diabetes_prevalence|gdp_per_capita|
    +-------------+-------------+---------------------+---------+-------------------+--------------+
    |        2.581|        1.337|              597.029|     Asia|               9.59|      1803.987|
    |         null|         null|                 null|     null|               null|          null|
    |       13.188|        8.643|              304.195|   Europe|              10.08|     11803.431|
    |        6.211|        3.857|              278.364|   Africa|               6.73|     13913.839|
    |         null|         null|              109.135|   Europe|               7.97|          null|
    +-------------+-------------+---------------------+---------+-------------------+--------------+

    For Daily Load:
    :input:
    +-------------+-------------+---------------------+---------+--------------------+
    |aged_65_older|aged_70_older|cardiovasc_death_rate|continent|                data|
    +-------------+-------------+---------------------+---------+--------------------+
    |        2.581|        1.337|              597.029|     Asia|[{date -> 2020-02...|
    |         null|         null|                 null|     null|[{date -> 2020-02...|
    |       13.188|        8.643|              304.195|   Europe|[{new_tests -> 8....|
    |        6.211|        3.857|              278.364|   Africa|[{date -> 2020-02...|
    |         null|         null|              109.135|   Europe|[{date -> 2020-03...|
    +-------------+-------------+---------------------+---------+--------------------+

    :output:
    Covid Info Dataframe
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |total_deaths|total_cases_per_million|new_cases_per_million|total_cases|      date|   location|
    +------------+-----------------------+---------------------+-----------+----------+-----------+
    |      2642.0|                 1549.0|                4.573|    60300.0|2021-05-03|Afghanistan|
    |    122554.0|               3415.529|                5.412|  4578852.0|2021-05-03|     Africa|
    |      2399.0|              45616.791|               13.205|   131276.0|2021-05-03|    Albania|
    |      3280.0|               2798.497|                4.447|   122717.0|2021-05-03|    Algeria|
    |       127.0|             172070.148|              168.252|    13295.0|2021-05-03|    Andorra|
    +------------+-----------------------+---------------------+-----------+----------+-----------+

    :param covid_data_rdd: RDD
    :param datetime_date: Datetime
    :return: Dataframe, Dataframe
    """

    covid_df = generate_covid_info(covid_data_rdd)
    country_df = generate_country_info(covid_data_rdd)

    return covid_df, country_df
