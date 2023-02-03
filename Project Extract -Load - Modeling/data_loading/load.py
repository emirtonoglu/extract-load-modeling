def load_data(covid_data, country_info):
    """
    This function gets covid info dataframe and country info dataframe. It loads data to mongodb in separate collections
    but in the same database. You should be careful about the write mode and you can find the sample code in this url.
    https://docs.mongodb.com/spark-connector/current/python/write-to-mongodb/

    :param covid_data: Dataframe of Covid Data
    :param country_info: Dataframe of Country Info
    """

    country_info.write.format("mongo").mode("append").option("database", "CovidData"). \
        option("collection", "CountryInfo").save()
    covid_data.write.format("mongo").mode("append").option("database", "CovidData"). \
        option("collection", "CovidInfo").save()
