import sparknlp

spark = sparknlp.start()

from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, array_contains
from pyspark.sql import Row

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

reviews_df = spark.read\
                .option("header", "true")\
                .json("hdfs:///user/apathak2/input/yelp_academic_dataset_review.json")\
                .withColumnRenamed("text", "review")

reviews_df.show(truncate=50)


documentAssembler = DocumentAssembler() \
    .setInputCol("review") \
    .setOutputCol("document")
useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")
sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter") \
    .setInputCols(["sentence_embeddings"]) \
    .setThreshold(0.7) \
    .setOutputCol("sentiment")
pipeline = Pipeline().setStages([
    documentAssembler,
    useEmbeddings,
    sentiment
])

pipelineModel = pipeline.fit(reviews_df)

result = pipelineModel.transform(reviews_df)

result.select("review", "sentiment.result").show()

result.show()

# Load business.json file
business_df = spark.read\
                .option("header", "true")\
                .json("yelp_academic_dataset_business.json")\
                .withColumnRenamed("name", "business_name")

business_df.show(truncate=50)

# -------Show the top 10 most positive businesses------
# Filter for positive sentiment using array_contains
positive_reviews = result.filter(array_contains(col("sentiment.result"), "positive"))

# Group by business ID and count positive reviews
business_positive_counts = positive_reviews.groupBy("business_id").count()

# Order by count in descending order
top_business_ids = business_positive_counts.orderBy("count", ascending=False)

# Take the top 10 business IDs
top_10_business_ids = top_business_ids.limit(10)

# Show the top 10 business IDs
top_10_business_ids.show(truncate=False)

# Join the top 10 business IDs with the business DataFrame to get business names
top_business_names = top_10_business_ids.join(business_df, "business_id", "inner")

# Select relevant columns
top_business_names = top_business_names.select("business_id", "business_name", "count")

# -------Show the top 10 most negative businesses------
# Filter for negative sentiment using array_contains
negative_reviews = result.filter(array_contains(col("sentiment.result"), "negative"))

# Group by business ID and count negative reviews
business_negative_counts = negative_reviews.groupBy("business_id").count()

# Order by count in ascending order
worst_business_ids = business_negative_counts.orderBy("count", ascending=False)

# Take the worst 10 business IDs
worst_business_ids = worst_business_ids.limit(10)

# Show the worst 10 business IDs
worst_business_ids.show()

# Join the top 10 business IDs with the business DataFrame to get business names
worst_business_names = worst_business_ids.join(business_df, "business_id", "inner")

# Select relevant columns
worst_business_names = worst_business_names.select("business_id", "business_name", "count")

# --------Get the sentiments--------
# Join the review DataFrame with the business DataFrame on the business_id column
joined_df = result.join(business_df, result.business_id == business_df.business_id, "inner")

# Select relevant columns for analysis
selected_df = joined_df.select("state", "sentiment.result")

# Count positive and negative sentiments for each state
sentiment_counts = selected_df.groupBy("state", "result").count()

# Pivot the DataFrame to have positive and negative sentiments as separate columns
pivoted_df = sentiment_counts.groupBy("state").pivot("result").agg({"count": "sum"}).na.fill(0)


# Convert the 'stars' column to integer type
result = result.withColumn("true_sentiment", when(col("stars").isin([1, 1.5, 2]), 0).when(col("stars").isin([2.5, 3, 3.5]), 1).otherwise(2))

# Convert the 'true_sentiment' column to integer type
result = result.withColumn("true_sentiment", col("true_sentiment").cast("int"))

# Show the updated DataFrame
result.show(5)

result = result.withColumn("predicted_sentiment", when(array_contains(col("sentiment.result"),"negative"), 0).when(array_contains(col("sentiment.result"),"neutral"), 1).otherwise(2))

# Convert the 'true_sentiment' column to integer type
result = result.withColumn("predicted_sentiment", col("predicted_sentiment").cast("double"))

# Show the updated DataFrame
result.show(5)
# Convert the 'true_sentiment' column to integer type
result = result.withColumn("true_sentiment", col("true_sentiment").cast("double"))
result = result.withColumnRenamed("predicted_sentiment", "prediction")

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="true_sentiment", metricName="f1")
f1 = evaluator.evaluate(result)
print("F1 score:", f1)


metrics_df = spark.createDataFrame(Row(metric="f1_score", value=float(f1)))
metrics_df.write.format("csv").option("header", "true").coalesce(1).save("hdfs:///user/apathak2/output/pretrained_metrics.csv")


spark.stop()