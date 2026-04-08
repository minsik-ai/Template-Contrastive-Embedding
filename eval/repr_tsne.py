from argparse import ArgumentParser

from alexa_dr_semantic_similarity.utils.spark_builder import SparkBuilder
import pyspark.sql.functions as func
from pyspark.sql.types import FloatType, StringType, ArrayType
from pyspark.sql.functions import col

def flatten(l):
    return [item for sublist in l for item in sublist]

def l1_norm(x):
    return [float(item) for item in x / x.norm(p=1)]


def l2_norm(x):
    return [float(item) for item in x / x.norm(p=2)]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--matched-utterance-path", type=str, required=True)
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--intent-count", type=int, default=5)
    parser.add_argument("--norm", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--intent-mode", action="store_true")
    return parser


def main(
        matched_utterance_path: str,
        embedding_path: str,
        intent_count: int,
        norm: int,
        output: str,
        intent_mode: bool
        ):
    assert norm in {0, 1, 2}

    print("Creating Spark")
    spark = SparkBuilder.get_or_create(log_level="ERROR")
    # sc = spark.sparkContext
    # sc.addPyFile("/home/ec2-user/workspaces/semantic-similarity/src/AlexaDRSemSimBERT/src/alexa_dr_semsim_bert.zip")
    print("Creating Spark done")

    print("Loading matched utterances and domains")
    mu_df = spark.read.load(matched_utterance_path).select("utterance", "pattern", "domain", "intent")

    # Filter on top intent_count with unique utterances
    targets = mu_df.select("utterance", "intent") \
        .distinct() \
        .groupby("intent").count() \
        .sort(col('count').desc()) \
        .take(intent_count)

    # Show intent count
    print(f"Targets : {targets}")

    # List of intents
    intents = [row.intent for row in targets]
    print(f"Intents : {intents}")

    encoded_utt = spark.read.parquet(embedding_path)

    print(f"Utt count : {encoded_utt.count()}")

    encoded_utt = encoded_utt\
        .join(mu_df, encoded_utt["utterance"] == mu_df["utterance"], "inner")\
        .select(mu_df.utterance, "pattern", "domain", "intent", "embedding_vector", "ptn_vector", "utt_vector")

    print(f"Inner Joined : {encoded_utt.count()}")

    encoded_utt = encoded_utt\
        .filter(encoded_utt.intent.isin(intents))

    print(f"Intent eligible : {encoded_utt.count()}")

    import pyspark.sql.functions as F
    slot_udf = F.udf(lambda x: "}" in x)
    encoded_utt = encoded_utt \
        .withColumn("slot_in", slot_udf(col("pattern")))
    print(f"Slot pattern only : {encoded_utt.head()}")

    encoded_utt = encoded_utt \
        .filter(encoded_utt["slot_in"] == True)
    print(f"Slot pattern only : {encoded_utt.count()}")

    encoded_utt.cache()

    if norm == 0:
        norm_udf = func.udf(lambda x: [float(it) for it in x], ArrayType(FloatType()))
    elif norm == 1:
        norm_udf = func.udf(l1_norm, ArrayType(FloatType()))
    else:
        norm_udf = func.udf(l2_norm, ArrayType(FloatType()))

    # Depending on count.... Do random selection
    print(f"Utterance count : {encoded_utt.count()}")
    if not intent_mode:
        encoded_utt = encoded_utt.withColumn("norm_utt_emb", norm_udf("utt_vector"))
        encoded_utt = encoded_utt.withColumn("norm_ptn_emb", norm_udf("ptn_vector"))
    encoded_utt = encoded_utt.withColumn("norm_emb", norm_udf("embedding_vector"))

    import pandas as pd

    utt_values = encoded_utt.select("utterance").withColumnRenamed("utterance", "value").toPandas()
    ptn_values = encoded_utt.select("pattern").withColumnRenamed("pattern", "value").toPandas()
    scale_rows = encoded_utt.select("norm_emb").toPandas()

    if not intent_mode:
        cols = {0: "Pattern", 1: "Utterance"}
        rename = {"norm_ptn_emb": "norm_emb", "norm_utt_emb": "norm_emb"}
        ptn_rows = encoded_utt.select("norm_ptn_emb").toPandas().rename(columns=rename)
        utt_rows = encoded_utt.select("norm_utt_emb").toPandas().rename(columns=rename)
        rows = [ptn_rows, utt_rows]
        values = [ptn_values, utt_values]
        labels = [[f"{idx}:{cols[idx]}"] * len(ser) for idx, ser in enumerate(rows)]
        X = pd.concat(rows, ignore_index=True, sort=False)
        Y = pd.Series(flatten(labels))
        Vals = pd.concat(values, ignore_index=True, sort=False)
        print("Label mode proc")
    else:
        X = scale_rows
        Y = encoded_utt.select("intent").toPandas()
        Vals = utt_values
        print("Intent mode proc")

    for idx in range(768):
        X[f'norm_emb_{idx}'] = X['norm_emb'].map(lambda x: x[idx])
    X = X.drop('norm_emb', axis=1)

    print(X.head())
    print(Y.head())
    print(Vals.head())

    import numpy as np

    print(f"X shape : {X.shape}")
    print(f"Y shape : {Y.shape}")
    print(f"Vals shape : {Vals.shape}")
    print(f"Intents : {np.unique(Y)}")

    from sklearn.manifold import TSNE

    n_components = 2
    tsne = TSNE(n_components)
    tsne_res = tsne.fit_transform(X)
    print(f"TSNE shape : {tsne_res.shape}")

    # Testing
    print(f"Start writing")
    np.savetxt(f"{output}/tsne_X.csv", tsne_res, fmt='%f')
    np.savetxt(f"{output}/Y.csv", Y, fmt="%s")
    np.savetxt(f"{output}/Vals.csv", Vals, fmt="%s", delimiter=";")

if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)