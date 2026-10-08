import re
from argparse import ArgumentParser
import collections
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=str, required=True)
    parser.add_argument("-p", "--extract-partition", type=str, required=True)
    parser.add_argument("-k", "--topk-slots", type=int, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-t", "--task", type=str, required=True)
    # TODO : Eval mode
    return parser

def _find_slots(annot):
    slot_vals = collections.defaultdict(set)
    for ent in re.findall("\[[^\[\]]+]", annot):
        kv = ent.strip("[] ").split(":")
        kv = [e.strip(" ") for e in kv]
        slot_vals[kv[0]].add(kv[1])
    return slot_vals

def _augment_slots(annot, entities):
    slots = []
    for ent in re.findall("\[[^\[\]]+]", annot):
        kv = ent.strip("[] ").split(" : ")
        slots.append(kv[0])

    cands = [annot]
    for slot in slots:
        repl = []
        for cand in cands:
            for val, cnt in entities[slot]:
                repl.append(re.sub("\[[^\[\]]+]", val, cand, count=1))
        cands = repl

    return cands

def _txt_proc(txt):
    return re.sub(",", " ", txt)

def _pattern_proc(annot):
    return re.sub("\[[^\[\]]+]", "{SLOT}", annot)

def main(data_path, extract_partition, topk_slots, output_path, task):
    print("Load data")
    import json

    entity_map = collections.defaultdict(list)
    slot_cnt = collections.defaultdict(int)
    if task == "massive":
        # Print Slot Info
        with open(data_path) as in_file:
            for row in in_file:
                json_item = json.loads(row)
                if json_item["partition"] == extract_partition:
                    ptn = json_item['annot_utt']
                    ents = _find_slots(ptn)
                    for slot, vals in ents.items():
                        for val in vals:
                            entity_map[slot].append(val)
                            slot_cnt[slot] += 1
                    print(f"{ents}, {ptn}")
    elif task == "snips":
        SNIPS_LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                        "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
        for lbl in SNIPS_LABELS:
            # TODO : train_full ?
            with open(os.path.join(data_path, lbl, f"{extract_partition}_{lbl}_full.json"), encoding='latin-1') as in_file:
                data = json.loads(in_file.read())
                rows = data[lbl]
                for row in rows:
                    for el in row["data"]:
                        if "entity" in el:
                            slot = el["entity"]
                            val = el["text"]
                            entity_map[slot].append(val)
                            slot_cnt[slot] += 1
                            print(f"{slot} : {val}")
    elif task == "hwu":
        valid_lines = open("./hwu64_valid_raw").readlines()
        test_lines = open("./hwu64_test_raw").readlines()
        valid_lines = set([l[:-1] for l in valid_lines])
        test_lines = set([l[:-1] for l in test_lines])

        with open(data_path) as in_file:
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
                    if extract_partition == "train" and ((txt in valid_lines) or (txt in test_lines)):
                        continue
                    if extract_partition == "valid" and txt not in valid_lines:
                        continue
                    if extract_partition == "test" and txt not in test_lines:
                        continue
                    ptn = vals[HWU_LABELS["answer_annotation"]]
                    ents = _find_slots(ptn)
                    for slot, vals in ents.items():
                        for val in vals:
                            entity_map[slot].append(val)
                            slot_cnt[slot] += 1
                    print(f"{ents}, {ptn}")
    elif task == "atis":
        with open(os.path.join(data_path, f"atis_{extract_partition}.json")) as in_file:
            data = json.loads(in_file.read())
            rows = data["rasa_nlu_data"]["common_examples"]
            for row in rows:
                for en in row["entities"]:
                    slot = en["entity"]
                    if '.' in slot:
                        slot = slot.split('.')[1]
                    val = en["value"]
                    entity_map[slot].append(val)
                    slot_cnt[slot] += 1
                    print(f"{slot} : {val}")


    print(f"Slots : {slot_cnt}")

    topk_entities = collections.defaultdict(list)
    for key, value in entity_map.items():
        topk_entities[key] = collections.Counter(value).most_common(topk_slots)

    counter = 0

    ptns = set()
    non_ptns = set()
    # Print permutations
    if task == "massive":
        with open(data_path) as in_file:
            with open(output_path, "w") as output_file:
                for row in in_file:
                    json_item = json.loads(row)
                    if json_item["partition"] == extract_partition:
                        utt = json_item['utt']
                        ptn = json_item['annot_utt']
                        cands = set(_augment_slots(ptn, topk_entities) + [utt])
                        for cand in cands:
                            non_ptns.add(ptn)
                            proced = _pattern_proc(ptn)
                            ptns.add(proced)
                            slots = proced.count("{SLOT}")
                            out_line = f"{cand},{proced},{slots}\n"
                            print(f"proc : {out_line}")
                            output_file.write(out_line)
                            counter += 1
    elif task == "snips":
        SNIPS_LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                        "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
        for lbl in SNIPS_LABELS:
            import json
            # TODO : train_full ?
            with open(os.path.join(data_path, lbl, f"{extract_partition}_{lbl}_full.json"), encoding='latin-1') as in_file:
                with open(output_path, "w") as out_file:
                    data = json.loads(in_file.read())
                    rows = data[lbl]
                    for row in rows:
                        utt = ""
                        ptn = ""
                        slots = 0
                        for el in row["data"]:
                            utt += el["text"]
                            if "entity" in el:
                                ptn += f"[{el['entity']} : {el['text']}]"
                                slots += 1
                            else:
                                ptn += el["text"]
                        cands = set(_augment_slots(ptn, topk_entities) + [utt])
                        for cand in cands:
                            non_ptns.add(ptn)
                            proced = _pattern_proc(ptn)
                            ptns.add(proced)
                            slots = proced.count("{SLOT}")
                            out_line = f"{_txt_proc(cand)},{_txt_proc(proced)},{slots}\n"
                            print(f"proc : {out_line}")
                            out_file.write(out_line)
                            counter += 1
    elif task == "hwu":
        with open(data_path) as in_file:
            with open(output_path, "w") as out_file:
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
                        if extract_partition == "train" and ((txt in valid_lines) or (txt in test_lines)):
                            continue
                        if extract_partition == "valid" and txt not in valid_lines:
                            continue
                        if extract_partition == "test" and txt not in test_lines:
                            continue
                        ptn = vals[HWU_LABELS["answer_annotation"]]
                        cands = set(_augment_slots(ptn, topk_entities) + [txt])
                        for cand in cands:
                            non_ptns.add(ptn)
                            proced = _pattern_proc(ptn)
                            ptns.add(proced)
                            slots = proced.count("{SLOT}")
                            out_line = f"{_txt_proc(cand)},{_txt_proc(proced)},{slots}\n"
                            print(f"proc : {out_line}")
                            out_file.write(out_line)
                            counter += 1
    elif task == "atis":
        with open(os.path.join(data_path, f"atis_{extract_partition}.json")) as in_file:
            with open(output_path, "w") as out_file:
                data = json.loads(in_file.read())
                rows = data["rasa_nlu_data"]["common_examples"]
                for row in rows:
                    txt = row["text"]
                    ptn = txt
                    offset = 0
                    for en in row["entities"]:
                        slot = en["entity"]
                        if '.' in slot:
                            slot = slot.split('.')[1]
                        val = en["value"]
                        comb = f"[{slot} : {val}]"
                        ptn = comb.join([ptn[:en["start"] + offset], ptn[en["end"] + offset:]])
                        offset += - (en["end"] - en["start"]) + len(comb)
                    cands = set(_augment_slots(ptn, topk_entities) + [txt])
                    for cand in cands:
                        non_ptns.add(ptn)
                        proced = _pattern_proc(ptn)
                        ptns.add(proced)
                        slots = proced.count("{SLOT}")
                        out_line = f"{_txt_proc(cand)},{_txt_proc(proced)},{slots}\n"
                        print(f"proc : {out_line}")
                        out_file.write(out_line)
                        counter += 1

    print(f"Write done : {counter}")
    print(f"Slot count : {len(entity_map.keys())}, Slot Value count : {len(set([item for sublist in entity_map.values() for item in sublist]))}, "
          f"template count : {len(non_ptns)}, proced template count : {len(ptns)}")


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)
