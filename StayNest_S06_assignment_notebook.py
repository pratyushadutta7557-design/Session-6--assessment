# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # StayNest - Session 6 Assignment (PySpark Deep Dive)
# MAGIC Work through the 8 tasks below in order. Read the Assignment Questions PDF for the
# MAGIC full detail and acceptance criteria. Fill in each `# TODO` cell, run it, and keep the
# MAGIC output visible. Run on Databricks Free Edition (serverless).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 0 - Setup (already done for you)
# MAGIC Upload `bookings.csv`, `hotels.csv`, `customers.csv` to a Volume, then set `BASE`
# MAGIC to that path and run this cell. Counts should be 12000 / 200 / 2000.

# COMMAND ----------

# Point BASE at YOUR Volume path
BASE = "/Volumes/workspace/default/staynest"

print(spark.version)

read_csv = lambda name: (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{BASE}/{name}.csv"))

bookings_df   = read_csv("bookings")
hotels_df     = read_csv("hotels")
customers_df  = read_csv("customers")

print(f"bookings: {bookings_df.count()}, "
      f"hotels: {hotels_df.count()}, "
      f"customers: {customers_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Read and inspect
# MAGIC Show the schema, a few sample rows, the row count, and summary stats for the
# MAGIC numeric columns of `bookings_df`.

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    DoubleType,
    IntegerType
)

bookings_schema = StructType([
    StructField("booking_id", IntegerType(), nullable=False),
    StructField("customer_id", IntegerType(), nullable=False),
    StructField("hotel_id", IntegerType(), nullable=False),
    StructField("booking_date", DateType(), nullable=False),
    StructField("city", StringType(), nullable=False),
    StructField("nights", IntegerType(), nullable=False),
    StructField("amount", DoubleType(), nullable=False),
    StructField("status", StringType(), nullable=False)
])

booking_df_prod = (
    spark.read
    .schema(bookings_schema)
    .option("header", True)
    .csv(f"{BASE}/bookings.csv")
)

booking_df_prod.printSchema()
booking_df_prod.count()
booking_df_prod.describe().show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Select and filter
# MAGIC From `bookings_df`, select a few useful columns and return the **completed**
# MAGIC bookings with `amount` over 10000 in the cities Goa or Mumbai. Use `col()`, combine
# MAGIC conditions with `&`, and use `.isin(...)`.

# COMMAND ----------

# TODO
from pyspark.sql.functions import col

bookings_selected = (
    bookings_df
    .select(
        col("booking_id"),
        col("city"),
        col("amount")
    )
    .filter(
        (col("city").isin("Goa", "Mumbai")) &
        (col("status") == "completed") &
        (col("amount") > 10000)
    )
)

bookings_selected.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Derived columns
# MAGIC Add: `amount_with_gst` (amount plus 12% tax), a `value_tier`
# MAGIC (premium / standard / budget) using `when`/`otherwise`, and a `booking_month`
# MAGIC from `booking_date`.

# COMMAND ----------

# TODO
from pyspark.sql.functions import when,month
bookings_df=bookings_df.withColumn(
    "amount_with_gst",
    col("amount")*1.12
    )

bookings_df=bookings_df.withColumn(
    "value_tier",
    when(col("amount")>12000,"Premium")
    .when(col("amount")>8000,"Standard")
    .otherwise("Budget")
    )

bookings_df=bookings_df.withColumn(
    "bookings_month",
    month(col("booking_date")
    )

    )
bookings_df.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Aggregations
# MAGIC For **completed** bookings, group by `city` and return: number of bookings, total
# MAGIC revenue, average amount, biggest booking, and the count of unique customers.
# MAGIC Order by revenue, highest first.

# COMMAND ----------

from pyspark.sql.functions import col, count, sum, avg, max, countDistinct

city_revenue = (
    bookings_df
    .filter(col("status") == "completed")
    .groupBy("city")
    .agg(
        count("booking_id").alias("no_of_bookings"),
        sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_revenue"),
        max("amount").alias("biggest_order"),
        countDistinct("customer_id").alias("unique_customers")
    )
    .orderBy(col("total_revenue").desc())
)

display(city_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Joins
# MAGIC Inner-join bookings to hotels to enrich each booking. Do a left join too. Use
# MAGIC `left_anti` to check for orphaned bookings (expect 0). Then do a three-way join
# MAGIC with customers.

# COMMAND ----------

# TODO

# Inner join: bookings + hotels
bookings_hotels_df = bookings_df.join(
    hotels_df,
    on="hotel_id",
    how="inner"
)

print("Inner join count:", bookings_hotels_df.count())
bookings_hotels_df.show()

# Left join: bookings + hotels
bookings_hotels_left_df = bookings_df.join(
    hotels_df,
    on="hotel_id",
    how="left"
)

print("Left join count:", bookings_hotels_left_df.count())

# Left anti join: find orphaned bookings
orphans_df = bookings_df.join(
    hotels_df,
    on="hotel_id",
    how="left_anti"
)

print("Orphaned bookings:", orphans_df.count())

# Three-way join: bookings + hotels + customers
bookings_hotels_customers_df = (
    bookings_df
    .join(
        hotels_df,
        on="hotel_id",
        how="inner"
    )
    .join(
        customers_df,
        on="customer_id",
        how="inner"
    )
)

print("Three-way join count:", bookings_hotels_customers_df.count())
bookings_hotels_customers_df.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Spark SQL + a window function
# MAGIC Register temp views and use `spark.sql` to get revenue by hotel `category` for
# MAGIC completed bookings. Then use a window function to rank the **top 3 hotels by
# MAGIC revenue within each city**.

# COMMAND ----------

bookings_df.createOrReplaceTempView("bookings")

hotels_df.createOrReplaceTempView("hotels")

customers_df.createOrReplaceTempView("customers")

revenue_by_category = spark.sql("""
    SELECT
        hotels.category,
        SUM(bookings.amount) AS total_revenue
    FROM bookings
    INNER JOIN hotels
        ON bookings.hotel_id = hotels.hotel_id
    GROUP BY hotels.category
    ORDER BY total_revenue DESC
""")

revenue_by_category.show()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import sum, dense_rank

# Give aliases to the DataFrames
b = bookings_df.alias("b")
h = hotels_df.alias("h")

# Join and calculate revenue per hotel per city
hotel_revenue = (
    b
    .join(
        h,
        col("b.hotel_id") == col("h.hotel_id"),
        "inner"
    )
    .groupBy(
        col("h.city"),
        col("h.hotel_name")
    )
    .agg(
        sum(col("b.amount")).alias("revenue")
    )
)

# Window function: rank hotels within each city by revenue
window_spec = (
    Window
    .partitionBy("city")
    .orderBy(col("revenue").desc())
)

top_3_hotels_per_city = (
    hotel_revenue
    .withColumn(
        "rank",
        dense_rank().over(window_spec)
    )
    .filter(col("rank") <= 3)
    .select(
        col("city"),
        col("hotel_name"),
        col("revenue"),
        col("rank")
    )
    .orderBy(
        col("city"),
        col("rank")
    )
)

display(top_3_hotels_per_city)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Write the result
# MAGIC Write your city-revenue result as **Parquet**, and also as a **Delta table** with
# MAGIC `saveAsTable`. Read the Delta table back to confirm.

# COMMAND ----------

# Write city revenue result as Parquet
city_revenue.write.mode("overwrite").parquet(
    f"{BASE}/output/city_revenue_parquet"
)

print(f"Written to {BASE}/output/city_revenue_parquet")


# Write city revenue result as Delta table
city_revenue.write.mode("overwrite").format("delta").option("mergeSchema", "true").saveAsTable(
    "workspace.default.city_revenue"
)

print("Registered as workspace.default.city_revenue")


# Read Delta table back to confirm
spark.table("workspace.default.city_revenue").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - One chained pipeline
# MAGIC In a single chain: keep completed bookings, join hotels, keep hotels with
# MAGIC `star_rating >= 4.0`, group by `city`, sum revenue, order descending, take the
# MAGIC top 5. End with one `.show()`.

# COMMAND ----------

top5_cities = (
    bookings_df
    .filter(col("status") == "completed")
    .join(
        hotels_df.drop("city"),
        on="hotel_id",
        how="inner"
    )
    .filter(col("star_rating") >= 4.0)
    .groupBy("city")
    .agg(
        sum("amount").alias("total_revenue")
    )
    .orderBy(
        col("total_revenue").desc()
    )
    .limit(5)
)

top5_cities.show()