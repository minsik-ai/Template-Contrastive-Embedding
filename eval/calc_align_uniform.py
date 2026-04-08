import os
import csv
import sys
import numpy as np
from argparse import ArgumentParser

from alexa_dr_semantic_similarity.utils.spark_builder import SparkBuilder
import pyspark.sql.functions as func
from pyspark.sql.types import FloatType, StringType, ArrayType
from pyspark.sql.functions import avg

# sys.path.insert(0, '/home/ec2-user/workspaces/semantic-similarity/src/AlexaDRSemSimBERT/src/')

def l2_norm(x):
    return [float(item) for item in x / x.norm(p=2)]

def align_metric(x, y, alpha=2):
    x, y = np.array(x), np.array(y)
    return float((np.linalg.norm(x - y, ord=2) ** alpha).mean())

def uniform_metric(x, x_pos, t=2):
    x, x_pos = np.array(x), np.array(x_pos)
    return float(np.log(np.exp((np.linalg.norm(x - x_pos, ord=2) ** 2) * -t).mean()))

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument("--matched-utterance-path", type=str, required=True)
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--local-model-path", type=str, required=True)
    # parser.add_argument("--output-dir", type=str, required=True)
    # parser.add_argument('--read-embeddings', action='store_true')
    # TODO : inference scaling
    return parser

def main(
        input_data: str,
        local_model_path: str
):
    # Input data should be formed as (utterance, pattern, intent)
    with open(input_data) as in_f:
        reader = csv.reader(in_f, delimiter=',')
        for row in reader:


def main(
        matched_utterance_path: str,
        output_dir: str,
        read_embeddings: bool
):
    embedding_output = os.path.join(output_dir, "embedding_vector")

    print("Creating Spark")
    spark = SparkBuilder.get_or_create(log_level="ERROR")
    sc = spark.sparkContext
    sc.addPyFile("/home/ec2-user/workspaces/semantic-similarity/src/AlexaDRSemSimBERT/src/alexa_dr_semsim_bert.zip")
    print("Creating Spark done")

    # No filtering since we only need the labels
    mu_df = spark.read.load(matched_utterance_path).select("utterance", "pattern", "domain", "intent")

    mu_df = mu_df.groupBy("utterance") \
        .agg(func.collect_set("domain").alias("domains"), func.collect_set("intent").alias("intents"))

    mu_df.cache()

    mu_data = mu_df.rdd.map(lambda x: (x["utterance"], set(x["domains"]), set(x["intents"]))).collect()
    mu_data = sc.broadcast({i[0]: (i[1], i[2]) for i in mu_data})

    if read_embeddings:
        print(f"Reading embeddings from path : {embedding_output}")
        encoded_utt = spark.read.parquet(embedding_output)
        encoded_utt.cache()

    norm2_udf = func.udf(l2_norm, ArrayType(FloatType()))

    encoded_utt = encoded_utt.withColumn("id", func.monotonically_increasing_id()) \
        .withColumn("norm_emb", norm2_udf("embedding_vector"))

    a = encoded_utt.selectExpr(f"utterance as utt1", "norm_emb as emb1", "id as id1")
    b = encoded_utt.selectExpr(f"utterance as utt2", "norm_emb as emb2", "id as id2")

    def _is_same_intent(key1: str, key2: str) -> str:
        _, intent_s1 = mu_data.value.get(key1, (set(), set()))
        _, intent_s2 = mu_data.value.get(key2, (set(), set()))
        is_same_intent = len(intent_s1.intersection(intent_s2)) > 0
        return f"{is_same_intent}"

    data_df = a.join(b, func.col("id1") < func.col("id2"))

    same_intent_udf = func.udf(_is_same_intent, StringType())
    align_udf = func.udf(align_metric, FloatType())
    uniform_udf = func.udf(uniform_metric, FloatType())

    print(data_df.count())

    data_df = data_df
        # .repartition(1000)
    # New caching
    data_df.cache()

    print("Strating comparison")

    align_df = data_df \
        .filter(same_intent_udf("utt1", "utt2") == "True") \
        .withColumn("align", align_udf("emb1", "emb2"))

    print(align_df.count())

    # Takes forever....
    uniform_df = data_df \
        .repartition(10000) \
        .withColumn("uniform", uniform_udf("emb1", "emb2")) \

    print(uniform_df.count())

    print("Results - uniform & align")
    align_avg = align_df.select(avg('align')).collect()[0]
    print(f"Align : {align_avg}")
    uniform_avg = uniform_df.select(avg('uniform')).collect()[0]
    print(f"Uniform : {uniform_avg}")

if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)