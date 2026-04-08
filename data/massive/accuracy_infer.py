from argparse import ArgumentParser
import spacy
import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics as sklm
from simcse import SimCSE

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-d", "--data-path", type=str, required=True)
    parser.add_argument("--local-model-path", type=str, required=True)
    parser.add_argument("--search-target", type=str, required=True)
    parser.add_argument("--pattern-target", type=str, required=False)
    parser.add_argument("--use-pair-for-target", action="store_true")
    parser.add_argument("--pattern-path-override", type=str, required=False)
    parser.add_argument("--aux-ratio", type=float, default=0.0)
    parser.add_argument("--pattern-threshold", type=float, default=float("-inf"))
    parser.add_argument("--pattern-topk", type=int, default=1)
    parser.add_argument("--utterance-topk", type=int, default=1)
    parser.add_argument("--pattern-avg", action="store_true")
    parser.add_argument("--model-type", type=str, required=False)
    parser.add_argument("--pooler", type=str, default="cls")
    parser.add_argument("--proto", action="store_true")
    # TODO : Spherical sum?
    # TODO : Prototypical network
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--unialign", action="store_true")
    # TODO : Pattern strategy?
    # TODO : Adjusting Pattern count?
    # TODO : Separate dataset for patterns?
    return parser


def _pattern_proc(annot):
    return re.sub("\n", "", re.sub("\[[^\[\]]+]", "{SLOT}", annot))


def _cos_sim(vec_A, vec_B):
    return cosine_similarity(vec_A, vec_B)

import torch

from transformers import AutoModel, AutoTokenizer

# from openai import OpenAI
#
# oai = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"]
# )

from dotenv import load_dotenv
load_dotenv()

from google import genai
goog = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

from google.genai import types

class LLM:

    def __init__(self, model_name):
        self.model_name = model_name

    def get_embeddings(self, texts):
        # response = oai.embeddings.create(
        #     model=self.model_name,
        #     input=texts,
        #     encoding_format="float",
        #     # dimensions=768
        # )
        # idxs = [item.index for item in response.data]
        # if not all(idxs[i] <= idxs[i + 1] for i in range(len(idxs) - 1)):
        #     raise ValueError(f"Indexes not sorted : {idxs}")
        # print(idxs)
        # return [torch.tensor(item.embedding) for item in response.data]
        # print(texts)
        response = goog.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        # for embed in response.embeddings:
        #     print(str(embed)[:50])
        return [torch.tensor(embed.values) for embed in response.embeddings]

    def encode(self, utts, batch_size=1):
        utt_batchs = [utts[i: min(i + batch_size, len(utts))] for i in range(0, len(utts), batch_size)]
        results = []
        # print(utts)
        for batch in utt_batchs:
            results.extend(self.get_embeddings(batch))
        return torch.stack(results, 0)


class DSE:

    def __init__(self, model_path):
        # Load the model and tokenizer
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Calculate the sentence embeddings by averaging the embeddings of non-padding words
        with torch.no_grad():
            embeddings = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = embeddings[0]*attention_mask
            embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            print(f"text  1 : {texts[0]}, size : {embeddings[0].size()}")
            if len(texts) != len(embeddings):
                raise ValueError(f"Wrong embedding lengths : {len(embeddings)}")
            return embeddings

    def encode(self, utts, batch_size=1):
        utt_batchs = [utts[i: min(i + batch_size, len(utts))] for i in range(0, len(utts), batch_size)]
        results = []
        for batch in utt_batchs:
            results.extend(self.get_embeddings(batch))
        return torch.stack(results, 0)

def _infer(model, aux_model, utts, aux_ratio, batch, nested_ptns=None, norm=0):
    utt_vecs = model.encode(utts, batch_size=batch).detach()
    ptn_vecs = None
    if nested_ptns:
        ptn_length = len(nested_ptns[0])
        import itertools
        fltn_ptn = list(itertools.chain(*nested_ptns))
        import torch
        if aux_ratio != 0:
            ptn_vecs = aux_model.encode(fltn_ptn, batch_size=batch).detach()
            ptn_vecs = [ptn_vecs[idx:idx+ptn_length] for idx in range(0, len(ptn_vecs), ptn_length)]
            ptn_vecs = [torch.mean(p, 0) for p in ptn_vecs]
    if norm != 0:
        from torch.nn.functional import normalize
        utt_vecs = [normalize(vec, p=norm, dim=0) for vec in utt_vecs]
        if aux_ratio != 0:
            ptn_vecs = [normalize(vec, p=norm, dim=0) for vec in ptn_vecs]
    if aux_ratio != 0:
        final_vecs = [torch.add((1.0 - aux_ratio) * utt_vecs[i], aux_ratio * ptn_vecs[i] if nested_ptns is not None else utt_vecs[i]) for i in range(len(utt_vecs))]
    else:
        final_vecs = utt_vecs
    final_vecs = [np.array(it) for it in final_vecs]
    return final_vecs, ptn_vecs


# def _select_ptn(patten_set):
#     # 1. pattern with many slots
#     # 2. longer pattern
#     cand = ""
#     slot_cnt = -1
#     idx = -1
#     for i, ptn in patten_set:
#         cnt = len(re.findall("{SLOT}", ptn))
#         if cnt > slot_cnt:
#             cand = ptn
#             slot_cnt = cnt
#             idx = i
#         elif cnt == slot_cnt and len(ptn) > len(cand):
#             cand = ptn
#             slot_cnt = cnt
#             idx = i
#     return idx, cand

def _select_utt(utt_set):
    # 1. select most frequent intent
    # 2. select highest scoring utterance
    import collections
    intent_cntr = collections.defaultdict(int)
    cur_int_cnt = 0
    cur_score = float('-inf')
    cur_item = None
    cur_idx = None

    # Intent Selection
    for i, item, score in utt_set:
        intent = item["intent"]
        int_cnt = intent_cntr[intent] + 1
        intent_cntr[intent] = int_cnt
        if int_cnt > cur_int_cnt or (int_cnt == cur_int_cnt and score > cur_score):
            # print(f"Utt update : {item['utt']} score {score}")
            cur_item = item
            cur_int_cnt = int_cnt
            cur_idx = i
            cur_score = score
    return cur_idx, cur_item

def l2_norm(x):
    return [float(item) for item in x / x.norm(p=2)]

def align_metric(x, y, alpha=2):
    x, y = np.array(x), np.array(y)
    return float((np.linalg.norm(x - y, ord=2) ** alpha).mean())

def uniform_metric(x, x_pos, t=2):
    x, x_pos = np.array(x), np.array(x_pos)
    return float(np.log(np.exp((np.linalg.norm(x - x_pos, ord=2) ** 2) * -t).mean()))

def uniform_metric2(x, x_pos, t=2):
    x, x_pos = np.array(x), np.array(x_pos)
    return float(np.exp((np.linalg.norm(x - x_pos, ord=2) ** 2) * -t).mean())


# Indices should match
def _search(query_vecs, key_vecs, value_items, thres_patterns=0.0, topk_patterns=0, topk_utterances=1):
    query_vecs = [np.array(res) for res in query_vecs]
    key_vecs = [np.array(res) for res in key_vecs]
    score_2d = _cos_sim(query_vecs, key_vecs)
    # print(f"LenX : {len(query_vecs)}, LenY : {len(key_vecs)}, Score2d shape : {score_2d.shape}")

    if topk_patterns == 0 and topk_utterances == 1:
        target_max_idx = np.argmax(score_2d, axis=1)
        scores = [score_2d[i][max_idx] for i, max_idx in enumerate(target_max_idx)]
        return [value_items[max_idx] for i, max_idx in enumerate(target_max_idx)], scores

    if topk_patterns > 0:
        topk_idx = np.argpartition(score_2d, -topk_patterns, axis=1)[:, -topk_patterns:]
        target_max_pairs = [[value_items[idx] for idx in top_list] for i, top_list in enumerate(topk_idx)]
        # Nested patterns
        # TODO : Score printing
        return target_max_pairs

    elif topk_utterances > 1:
        topk_idx = np.argpartition(score_2d, -topk_utterances, axis=1)[:, -topk_utterances:]
        target_max_pairs = [_select_utt([(idx, value_items[idx], score_2d[i][idx]) for idx in top_list]) for i, top_list in enumerate(topk_idx)]
    else:
        raise ValueError(f"Wrong topk : {topk_patterns}, {topk_utterances}")

    target_max_idx = [idx for idx, cand in target_max_pairs]
    target_max_item = [cand for idx, cand in target_max_pairs]


    scores = [score_2d[i][max_idx] for i, max_idx in enumerate(target_max_idx)]
    if topk_utterances > 1:
        return target_max_item, scores
    elif topk_patterns > 0:
        return [max_item if (topk_patterns == 0 or scores[i] > thres_patterns) else "N / A" for i, max_item in enumerate(target_max_item)], scores
    else:
        raise ValueError(f"Wrong topk : {topk_patterns}, {topk_utterances}")

def _txt_proc(txt):
    return re.sub(",", " ", txt)

def main(task, data_path, local_model_path, search_target, pattern_target, pattern_path_override, aux_ratio,
         use_pair_for_target, pattern_threshold, pattern_topk, utterance_topk, pattern_avg, model_type, normalize,
         proto, pooler, unialign):
    # print("Creating Spark")
    # spark = SparkBuilder.get_or_create(log_level="ERROR")
    # sc = spark.sparkContext
    # sc.addPyFile("/home/ec2-user/workspaces/semantic-similarity/src/AlexaDRSemSimBERT/src/alexa_dr_semsim_bert.zip")
    # print("Creating Spark done")

    import json
    print("Load data")
    train_utts = []
    dev_utts = []
    test_utts = []
    if task == "massive":
        with open(data_path) as in_file:
            for row in in_file:
                print(f"row : {row}")
                json_item = json.loads(row)
                item = {
                    # TODO : ID?
                    "utt": json_item["utt"],
                    "ptn": _pattern_proc(json_item["annot_utt"]),
                    "intent": json_item["intent"],
                    "scenario": json_item["scenario"]
                }
                if json_item["partition"] == "dev":
                    dev_utts.append(item)
                elif json_item["partition"] == "test":
                    test_utts.append(item)
                elif json_item["partition"] == "train":
                    train_utts.append(item)
    elif task == "snips" or task == "snips_full":
        SNIPS_LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                        "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
        for lbl in SNIPS_LABELS:
            # TODO : train_full ?
            for partition in ["validate", "train"] if task == "snips" else ["validate", "train_full"]:
                with open(os.path.join(data_path, lbl, f"{partition}_{lbl}.json" if partition != "train_full" else f"train_{lbl}_full.json"), encoding='latin-1') as in_file:
                    data = json.loads(in_file.read())
                    rows = data[lbl]
                    for row in rows:
                        txt = ""
                        ptn = ""
                        slot_cnt = 0
                        for el in row["data"]:
                            txt += el["text"]
                            if "entity" in el:
                                ptn += "{SLOT}"
                                slot_cnt += 1
                            else:
                                ptn += el["text"]
                        item = {
                            "utt": _txt_proc(txt),
                            "ptn": _pattern_proc(_txt_proc(ptn)),
                            "intent": lbl
                        }
                        if "train" in partition:
                            train_utts.append(item)
                        elif partition == "validate":
                            test_utts.append(item)
    elif task == "hwu":
        valid_lines = open("data/massive/hwu64_valid_raw").readlines()
        test_lines = open("data/massive/hwu64_test_raw").readlines()
        valid_lines = set([l[:-1] for l in valid_lines])
        test_lines = set([l[:-1] for l in test_lines])

        with open(data_path) as in_file:
            # https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv
            HWU_LABELS_RAW = "userid;answerid;scenario;intent;status;answer_annotation;notes;suggested_entities;answer_normalised;answer;question".split(';')
            HWU_LABELS = {lbl : i for i, lbl in enumerate(HWU_LABELS_RAW)}
            lbls = None
            for row in in_file:
                print(f"row : {row}")
                vals = row.split(';')
                if not lbls:
                    vals[-1] = vals[-1][:-1]
                    lbls = vals
                    print(lbls)
                    assert HWU_LABELS_RAW == lbls
                else:
                    txt = vals[HWU_LABELS["answer_normalised"]]
                    if len(txt) == 0:
                        continue
                    ptn = vals[HWU_LABELS["answer_annotation"]]
                    ptn = _pattern_proc(ptn)
                    item = {
                        "utt": _txt_proc(txt),
                        "ptn": _pattern_proc(_txt_proc(ptn)),
                        "intent": vals[HWU_LABELS["intent"]]
                    }
                    if txt in valid_lines:
                        dev_utts.append(item)
                    elif txt in test_lines:
                        test_utts.append(item)
                    else:
                        train_utts.append(item)
    elif task == "atis":
        for partition in ["train", "test"]:
            with open(os.path.join(data_path, f"atis_{partition}.json")) as in_file:
                data = json.loads(in_file.read())
                rows = data["rasa_nlu_data"]["common_examples"]
                for row in rows:
                    txt = row["text"]
                    ptn = txt
                    offset = 0
                    for en in row["entities"]:
                        ptn = "{SLOT}".join([ptn[:en["start"] + offset], ptn[en["end"] + offset:]])
                        offset += - (en["end"] - en["start"]) + 6
                    intent = row["intent"]
                    item = {
                        "utt": _txt_proc(txt),
                        "ptn": _txt_proc(ptn),
                        "intent": intent
                    }
                    if partition == "train":
                        train_utts.append(item)
                    elif partition == "test":
                        test_utts.append(item)
    elif task == "clinc" or task == "clinc_uttonly":
        print("No slots for CLINC150 evaluations!")
        from datasets import load_dataset
        clinc150 = load_dataset("clinc_oos", "plus")

        for partition in ["train", "test"]:
            if task == "clinc_uttonly":
                for i, row in enumerate(clinc150[partition]):
                    item = {
                        "utt": _txt_proc(row['text']),
                        "ptn": "",
                        "intent": row['intent']
                    }
                    if partition == "train":
                        train_utts.append(item)
                    elif partition == "test":
                        test_utts.append(item)
            else:
                # Uses output from extract_ptn_clinc.py
                with open(os.path.join(data_path, f"{partition}_clinc_out.csv")) as f:
                    rows = f.readlines()
                    for i, row in enumerate(clinc150[partition]):
                        split_row = rows[i].split(',')
                        item = {
                            "utt": _txt_proc(row['text']),
                            "ptn": _txt_proc(_pattern_proc(split_row[1])),
                            "intent": row['intent']
                        }
                        if item["utt"] != split_row[0]:
                            raise ValueError()
                        if partition == "train":
                            train_utts.append(item)
                        elif partition == "test":
                            test_utts.append(item)

    print(f"Train : {len(train_utts)}, Valid : {len(dev_utts)}, Test : {len(test_utts)}")

    if model_type is None:
        # TODO : Ratio
        model = SimCSE(local_model_path, pooler=pooler)
        aux_model = SimCSE(local_model_path, pooler=pooler)
    elif model_type == "DSE":
        model = DSE(local_model_path)
        aux_model = DSE(local_model_path)
    elif model_type == "LLM":
        model = LLM(local_model_path)
        # Should not be accessed
        aux_model = None
    else:
        raise ValueError(f"Wrong model type : {model_type}")

    # TODO : Spark support?
    print("Infer base representations to form ground truths in terms of intent")
    # What if there are too small valid representations? Then do the same with training

    if search_target == "train":
        target_utts = train_utts
    elif search_target == "dev":
        target_utts = dev_utts
    else:
        raise ValueError(f"Wrong search target : {search_target}")

    if pattern_target is not None and pattern_path_override is not None:
        raise ValueError("Only one pattern path can be provided.")

    print("Processing patterns....")
    if pattern_target is None:
        if pattern_path_override is not None:
            print("Pattern path override")
            target_patterns = []
            with open(pattern_path_override) as pattern_file:
                for row in pattern_file:
                    print(f"row : {row}")
                    target_patterns.append(_pattern_proc(row))
        else:
            target_patterns = None
    elif pattern_target == "train":
        target_patterns = [item["ptn"] for item in train_utts]
    elif pattern_target == "dev":
        target_patterns = [item["ptn"] for item in dev_utts]
    else:
        raise ValueError(f"Wrong pattern target : {pattern_target}")

    if target_patterns is not None:
        print(f"Previous pattern length : {len(target_patterns)}")
    # target_patterns = [ptn for ptn in target_patterns if "{SLOT}" in ptn]
    # print(f"New pattern length : {len(target_patterns)}")

    print("Encoding patterns for search {if available}")
    batch = 64
    # Encode patterns for search
    if target_patterns is not None:
        ptn_vecs = []
        for i in range(0, 1 + int(len(target_patterns) / batch)):
            items = target_patterns[i * batch: (1 + i) * batch]
            print(f"Encoding patt batch {i} : {items[0]}")
            res_vecs = model.encode(items, batch_size=batch).detach()
            ptn_vecs.extend(res_vecs)
    else:
        ptn_vecs = None

    print("Inferring utterances")
    target_vecs = []
    for i in range(0, 1 + int(len(target_utts) / batch)):
        items = target_utts[i * batch: (1 + i) * batch]
        print(f"Encoding batch {i}")
        query_utts = [it["utt"] for it in items]

        # TODO : Make this selectable?
        if ptn_vecs is None or use_pair_for_target:
            nested_ptns = [[it["ptn"]] for it in items]
        else:
            utt_vecs = model.encode(query_utts, batch_size=batch).detach()

            nested_ptns = _search(utt_vecs, ptn_vecs, target_patterns, pattern_threshold, pattern_topk)

        res_vecs, _ = _infer(model, aux_model, query_utts, aux_ratio, batch, nested_ptns=nested_ptns, norm=normalize)
        target_vecs.extend(res_vecs)

    if proto:
        import collections
        int_vecs = collections.defaultdict(list)
        int_utt_sample = {}
        # Average vectors with same labels
        for i, utt in enumerate(target_utts):
            intent = utt["intent"]
            int_vecs[intent].append(target_vecs[i])
            if intent not in int_utt_sample:
                int_utt_sample[intent] = utt

        print(len(int_vecs[intent][0]))
        int_cntrs = {}
        for intent in int_vecs:
            int_cntrs[intent] = np.mean(int_vecs[intent], axis=0)
        print(int_cntrs.keys())
        print(target_utts[0])
        # print(target_vecs[0])
        print(len(target_vecs[0]))
        target_vecs = []
        target_utts = []
        for intent, vec in int_cntrs.items():
            target_vecs.append(vec)
            target_utts.append(int_utt_sample[intent])
        print(target_utts[0])
        # print(target_vecs[0])
        print(len(target_vecs[0]))
        # Pattern pool should not be updated yet (target_patterns) - we should use same train data.
        # TODO : Patterns could also be updated as a form of average

    import time
    time.sleep(1)

    # Testing with rough implementation
    print("Starting test inference")
    test_utts = test_utts
    test_preds = []
    batch = 64

    # For uniformity / alignment
    import collections
    int_emb = collections.defaultdict(list)

    for i in range(0, 1 + int(len(test_utts) / batch)):
        items = test_utts[i * batch: (1 + i) * batch]
        print(f"Inferring batch {i}")
        query_utts = [it["utt"] for it in items]
        query_ints = [it["intent"] for it in items]
        ptn_scores = None
        if ptn_vecs is None or use_pair_for_target:
            nested_ptns = [[it["ptn"]] for it in items]
            display_ptn = nested_ptns[0]
        else:
            utt_vecs = model.encode(query_utts, batch_size=batch).detach()

            nested_ptns = _search(utt_vecs, ptn_vecs, target_patterns, pattern_threshold, pattern_topk)
            display_ptn = nested_ptns[0]

        res_vecs, ptn_vecs = _infer(model, aux_model, query_utts, aux_ratio, batch, nested_ptns=nested_ptns, norm=normalize)

        if unialign:
            for i, vec in enumerate(res_vecs):
                int_emb[query_ints[i]].append(vec)

        # TODO : FAISS
        print(f"Top {utterance_topk} comp for batch {i}")
        # print(target_utts)
        target_items, scores = _search(res_vecs, target_vecs, target_utts, topk_utterances=utterance_topk)
        # print(f"debug : {target_items}, {scores}")
        sim_utt = [item["utt"] for item in target_items]
        intents = [item["intent"] for item in target_items]
        print(
            f"utt : {query_utts[0]}, sim_utt : {sim_utt[0]}, sim_ptn : {display_ptn}, score : {scores[0]}, intent : {intents[0]}")
        test_preds.extend(intents)

    print("Test inference done")
    test_labels = [it["intent"] for it in test_utts]
    assert len(test_labels) == len(test_preds)
    intent_acc = sklm.accuracy_score(test_labels, test_preds)
    print(f"Intent Accuracy on test set for {task} : {intent_acc}")

    if unialign:
        print(f"Perfomring uniformity / alignment")
        print("Alignment")

        all_embs = []
        all_ints = []
        # TODO : Alignment per intent?
        align = 0
        cnt = 0
        for intent, embs in int_emb.items():
            all_embs.extend(embs)
            all_ints.extend([intent] * len(embs))
            for i, emb_i in enumerate(embs):
                for j in range(i + 1, len(embs)):
                    emb_j = embs[j]
                    cnt += 1
                    align += align_metric(emb_i, emb_j)

        print(len(all_embs))
        uniform = 0
        cnt = 0
        for i, emb_i in enumerate(all_embs):
            for j in range(i + 1, len(all_embs)):
                emb_j = all_embs[j]
                cnt += 1
                # uniform += uniform_metric(emb_i, emb_j)
                uniform += uniform_metric2(emb_i, emb_j)
        uniform /= cnt
        uniform = np.log(uniform)
        print(f"Alignment : {align / cnt}")
        print("Uniformity")
        print(f"Uniformity : {uniform}")

        # from sklearn.manifold import TSNE
        #
        # n_components = 2
        # tsne = TSNE(n_components)
        # tsne_result = tsne.fit_transform(all_embs)
        # y = all_ints
        # print(f"TSNE shape : {tsne_result.shape}")
        #
        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # plt.rcParams["figure.figsize"] = (10, 6)
        # # Testing
        # tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        # fig, ax = plt.subplots(1)
        # sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=20)
        # lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        #
        # plt.show()

if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)
