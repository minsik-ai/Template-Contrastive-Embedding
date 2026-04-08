from argparse import ArgumentParser
import re


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=str, required=True)
    parser.add_argument("-p", "--extract-partition", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-t", "--task-name", type=str, required=True)
    # TODO : Eval mode
    return parser

def _txt_proc(txt):
    return re.sub(",", " ", txt)

def _pattern_proc(annot):
    return re.sub("\[[^\[\]]+]", "{SLOT}", _txt_proc(annot))


def main(data_path, extract_partition, output_path, task_name):
    print(f"Load data, type : {task_name}")

    import os
    import json
    counter = 0
    if task_name == "massive":
        with open(data_path) as in_file:
            with open(output_path, "w") as output_file:
                for row in in_file:
                    print(f"row : {row}")
                    json_item = json.loads(row)
                    if json_item["partition"] == extract_partition:
                        ptn = _pattern_proc(json_item['annot_utt'])
                        slot_cnt = ptn.count("{SLOT}")
                        out_line = f"{json_item['utt']},{ptn},{slot_cnt}\n"
                        print(f"proc : {out_line}")
                        output_file.write(out_line)
                        counter += 1
    elif task_name == "massive_joint":

        folder_path = os.path.join(output_path, f"{extract_partition}")

        intent_path = os.path.join(folder_path, "intent_label.txt")
        seq_path = os.path.join(folder_path, 'seq.in')
        temp_path = os.path.join(folder_path, 'aux.in')
        label_path = os.path.join(folder_path, 'label')

        intents = set()

        with open(data_path) as in_file:
            with open(seq_path, "w") as seq_file:
                with open(temp_path, "w") as temp_file:
                    with open(label_path, "w") as label_file:
                        for row in in_file:
                            print(f"row : {row}")
                            json_item = json.loads(row)
                            if json_item["partition"] == extract_partition:
                                ptn = _pattern_proc(json_item['annot_utt'])
                                slot_cnt = ptn.count("{SLOT}")
                                utt = json_item['utt']
                                intent = json_item['intent']
                                intents.add(intent)

                                seq_file.write(f"{utt}\n")
                                temp_file.write(f"{ptn}\n")
                                label_file.write(f"{intent}\n")
                                counter += 1

        print(len(intents))
        with open(intent_path, "w") as intent_file:
            intents = list(intents)
            intents.sort()
            for intent in intents:
                intent_file.write(f"{intent}\n")

    elif task_name == "snips":
        SNIPS_LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                        "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
        for lbl in SNIPS_LABELS:
            import json
            # TODO : train_full ?
            with open(os.path.join(data_path, lbl, f"{extract_partition}_{lbl}.json"), encoding='latin-1') as in_file:
                with open(output_path, "w") as out_file:
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
                        out_line = f"{_txt_proc(txt)},{_txt_proc(ptn)},{slot_cnt}\n"
                        print(f"proc : {out_line}")
                        out_file.write(out_line)
                        counter += 1
    elif task_name == "hwu":
        valid_lines = open("./hwu64_valid_raw").readlines()
        test_lines = open("./hwu64_test_raw").readlines()
        valid_lines = set([l[:-1] for l in valid_lines])
        test_lines = set([l[:-1] for l in test_lines])

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
                        ptn = _pattern_proc(ptn)
                        slot_count = ptn.count("{SLOT}")
                        out_line = f"{_txt_proc(txt)},{ptn},{slot_count}\n"
                        print(f"proc : {out_line}")
                        out_file.write(out_line)
                        counter += 1
    elif task_name == "atis":
        with open(os.path.join(data_path, f"atis_{extract_partition}.json")) as in_file:
            with open(output_path, "w") as out_file:
                data = json.loads(in_file.read())
                rows = data["rasa_nlu_data"]["common_examples"]
                for row in rows:
                    txt = row["text"]
                    ptn = txt
                    offset = 0
                    for en in row["entities"]:
                        ptn = "{SLOT}".join([ptn[:en["start"] + offset], ptn[en["end"] + offset:]])
                        offset += - (en["end"] - en["start"]) + 6
                    # INTENT : row["intent"]
                    out_line = f"{_txt_proc(txt)},{_txt_proc(ptn)},{len(row['entities'])}\n"
                    print(f"proc : {out_line}")
                    out_file.write(out_line)
                    counter += 1
    else:
        raise ValueError(f"Wrong task name : {task_name}")

    print(f"{task_name}, write done : {counter}")


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)
