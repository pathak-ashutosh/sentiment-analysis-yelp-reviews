from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, when, desc, lower, expr, udf, split, explode
from pyspark.sql.types import StringType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer, HashingTF, IDF, CountVectorizer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import re

spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.memory", "30g") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.executor.cores", 2) \
    .getOrCreate()

# Read in json files using spark.read.json
yelpReviewDF = spark.read.json('hdfs:///user/apathak2/input/yelp_academic_dataset_review.json')
yelpBusinessDF = spark.read.json('hdfs:///user/apathak2/input/yelp_academic_dataset_business.json')

yelpReviewDF.show(5)
yelpBusinessDF.show(5)

yelpReviewDF = yelpReviewDF.dropna()

print(yelpBusinessDF.count())
print(yelpReviewDF.count())

# Number of Unique Businesses in Review Data
unique_businesses_reviews = yelpReviewDF.select("business_id").distinct().count()
print('Number of Unique Businesses in Review Data: ', unique_businesses_reviews)

# Number of Unique Businesses in Business Data
unique_businesses_business = yelpBusinessDF.select("business_id").distinct().count()
print('Number of Unique Businesses in Business Data: ', unique_businesses_business)


"""**Count of the unique business categories**"""

# Extract and explode categories
categories_df = yelpBusinessDF.select("categories").na.drop().withColumn("category", explode(split("categories", ", ")))

# Count the occurrences of each category
category_counts = categories_df.groupBy("category").count().orderBy("count", ascending=False)

# Select the 'stars' column
stars_df = yelpBusinessDF.select("stars")

# Group by 'state' and count the number of businesses
business_by_state = yelpBusinessDF.groupBy("state").count().orderBy(desc("count"))

"""**Sentiment Analysis**"""

# Recode 1 - 3 stars as 0 (negative review)
# Recode 3.5 - 5 stars as 1 (positive review)
yelpReviewDF = yelpReviewDF.withColumn("sentiment", when(col("stars").isin([1, 1.5, 2, 2.5, 3]), 0).otherwise(1))

# Convert the 'sentiment' column to integer type
yelpReviewDF = yelpReviewDF.withColumn("sentiment", col("sentiment").cast("int"))

# Convert strings to lowercase
yelpReviewDF = yelpReviewDF.withColumn("pre_process", lower(col("text")))

# Replace contractions using expr function
contractions_replacements = [
    ("\\'d", " would"),
    ("\\'t", " not"),
    ("\\'t", " not"),
    ("\\'re", " are"),
    ("\\'s", " is"),
    ("\\'ll", " will"),
    ("\\'t", " not"),
    ("\\'ve", " have"),
    ("\\'m", " am")
]

for original, replacement in contractions_replacements:
    yelpReviewDF = yelpReviewDF.withColumn("pre_process",
                                        expr(f"regexp_replace(pre_process, '{original}', '{replacement}')"))

# Remove non-alpha characters
alpha_regex_udf = udf(lambda x: re.sub('[^A-Za-z]+', ' ', x), StringType())
yelpReviewDF = yelpReviewDF.withColumn("pre_process", alpha_regex_udf(col("pre_process")))

# Remove extra spaces between words
spaces_regex_udf = udf(lambda x: re.sub(' +', ' ', x), StringType())
yelpReviewDF = yelpReviewDF.withColumn("pre_process", spaces_regex_udf(col("pre_process")))

"""**Count Vectorizer**"""

# Load the stop words list
stop_words = StopWordsRemover.loadDefaultStopWords("english")

# Tokenizer
tokenizer = RegexTokenizer(inputCol="pre_process", outputCol="words", pattern="\\W")
yelpReviewDF = tokenizer.transform(yelpReviewDF)

# Remove stop words
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
yelpReviewDF = stop_words_remover.transform(yelpReviewDF)

# Lemmatization in PySpark (using CountVectorizer as a workaround)
count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="lemmatized_words", vocabSize=2)
model = count_vectorizer.fit(yelpReviewDF)
yelpReviewDF = model.transform(yelpReviewDF)

# Split the data
train_data, test_data = yelpReviewDF.randomSplit([0.8, 0.2], seed=42)
training_data, valid_data = train_data.randomSplit([0.8, 0.2], seed=42)

# Linear SVC
linear_svc = LinearSVC(featuresCol="lemmatized_words", labelCol="sentiment")

# Initialize ParamGrid
paramGrid = ParamGridBuilder() \
    .addGrid(linear_svc.maxIter, [10, 100, 1000]) \
    .addGrid(linear_svc.regParam, [0.01, 0.1]) \
    .build()

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", metricName="f1")
cv = CrossValidator(estimator=linear_svc, estimatorParamMaps=paramGrid, evaluator=evaluator,
    parallelism=2)
cvModel = cv.fit(training_data)

best_model = cvModel.bestModel

# Print the best parameters
print("Best Parameters:")
print("regParam:", best_model.getOrDefault("regParam"))
print("maxIter:", best_model.getOrDefault("maxIter"))

# Make predictions on validation data
predictions = best_model.transform(valid_data)
f1_score = evaluator.evaluate(predictions)
print(f"F1 score: {f1_score}")

# Show the DataFrame with predictions
predictions.select("text", "sentiment", "prediction").show()

# Make predictions on test data
test_predictions = best_model.transform(test_data)
f1_score = evaluator.evaluate(test_predictions)
print(f"F1 score: {f1_score}")

# Show the DataFrame with predictions
test_predictions.select("text", "sentiment", "prediction").show()

"""**TF-IDF**"""

# Load the stop words list
stop_words = StopWordsRemover.loadDefaultStopWords("english")

# Tokenizer
tokenizer = RegexTokenizer(inputCol="pre_process", outputCol="tokens", pattern="\\W")
yelpReviewDF = tokenizer.transform(yelpReviewDF)

# Remove stop words
stop_words_remover = StopWordsRemover(inputCol="tokens", outputCol="clean_tokens")
yelpReviewDF = stop_words_remover.transform(yelpReviewDF)

# TF-IDF Vectorization
hashing_tf = HashingTF(inputCol="clean_tokens", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Create the pipeline
pipeline = Pipeline(stages=[hashing_tf, idf])

# Fit the pipeline to the data
model = pipeline.fit(yelpReviewDF)

# Transform the data
yelpReviewDF = model.transform(yelpReviewDF)

"""**Undersampling**"""

# Separate positive and negative reviews
positive_reviews = yelpReviewDF.filter(col('sentiment') == 1)
negative_reviews = yelpReviewDF.filter(col('sentiment') == 0)

# Count the number of positive and negative reviews
num_positive = positive_reviews.count()
num_negative = negative_reviews.count()

# Determine the minority class (positive or negative) and its count
if num_positive < num_negative:
    minority_reviews = positive_reviews
    majority_reviews = negative_reviews
    minority_count = num_positive
else:
    minority_reviews = negative_reviews
    majority_reviews = positive_reviews
    minority_count = num_negative

# Calculate the number of rows in the majority_reviews DataFrame
num_majority = majority_reviews.count()

# Calculate the number of rows in the minority_reviews DataFrame
num_minority = minority_reviews.count()

# Undersample the majority class to balance the dataset
undersampled_majority = majority_reviews.sample(withReplacement=False, fraction=minority_count / num_majority)

# Concatenate the undersampled majority class with the minority class
undersampled_result = minority_reviews.union(undersampled_majority)

print(undersampled_result.filter(col('sentiment') == 1).count())
print(undersampled_result.filter(col('sentiment') == 0).count())

# Split the data
train_data, test_data = undersampled_result.randomSplit([0.8, 0.2], seed=42)
training_data, valid_data = undersampled_result.randomSplit([0.8, 0.2], seed=42)

# Linear SVC
linear_svc = LinearSVC(featuresCol="features", labelCol="sentiment")

# Initialize ParamGrid
paramGrid = ParamGridBuilder() \
    .addGrid(linear_svc.maxIter, [10, 100, 1000]) \
    .addGrid(linear_svc.regParam, [0.01, 0.1]) \
    .build()

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", metricName="f1")
cv = CrossValidator(estimator=linear_svc, estimatorParamMaps=paramGrid, evaluator=evaluator,
    parallelism=2)
cvModel_tfidf = cv.fit(training_data)

best_model_tfidf = cvModel_tfidf.bestModel
# Print the best parameters
print("Best Parameters:")
print("regParam:", best_model_tfidf.getOrDefault("regParam"))
print("maxIter:", best_model_tfidf.getOrDefault("maxIter"))

# Make predictions
predictions_tfidf = best_model_tfidf.transform(valid_data)
f1_score = evaluator.evaluate(predictions_tfidf)
print(f"F1 score: {f1_score}")

# Show the DataFrame with predictions
predictions_tfidf.select("text", "sentiment", "prediction").show(5)

# Make predictions
predictions_tfidf_test = best_model_tfidf.transform(test_data)
f1_score_tfidf = evaluator.evaluate(predictions_tfidf_test)
print(f"F1 score: {f1_score_tfidf}")

# Show the DataFrame with predictions
predictions_tfidf_test.select("text", "sentiment", "prediction").show()

metrics_data = [
    Row(metric="best_cv_regParam", value=float(best_model.getOrDefault("regParam"))),
    Row(metric="best_cv_maxIter", value=float(best_model.getOrDefault("maxIter"))),
    Row(metric="cv_f1_score", value=float(f1_score)),
    Row(metric="best_tfidf_regParam", value=float(best_model_tfidf.getOrDefault("regParam"))),
    Row(metric="best_tfidf_maxIter", value=float(best_model_tfidf.getOrDefault("maxIter"))),
    Row(metric="tfidf_f1_score", value=float(f1_score_tfidf))
]

metrics_df = spark.createDataFrame(metrics_data)
metrics_df.write.format("csv").option("header", "true").coalesce(1).save("hdfs:///user/apathak2/output/scratch_metrics.csv")


spark.stop()