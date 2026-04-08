import spacy
import re
from argparse import ArgumentParser
from datasets import load_dataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-p", "--extract-partition", type=str, required=True)
    parser.add_argument("-k", "--topk-slots", type=int, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    # TODO : Eval mode
    return parser

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

def main(extract_partition, output_path, topk_slots):
    print("Starting")
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    clinc150 = load_dataset("clinc_oos", "plus")
    txts = []
    ptns = []
    lbls = []
    import collections
    ents = collections.defaultdict(list)
    non_ptns = set()
    ptnUnique = set()
    entCnt = collections.Counter()
    valCnt = collections.Counter()
    print("Processing")
    with open(output_path, "w") as f:
        for item in clinc150[extract_partition]:
            txt = item['text']
            lbl = item['intent']
            txts.append(txt)
            lbls.append(lbl)
            print(f"{txt}: {lbl}")

        for i, doc in enumerate(nlp.pipe(txts)):
            txt = doc.text
            print(f"Text : {txt} - {lbls[i]}")
            parts = []
            prevEnd = 0
            for ent in doc.ents:
                ents[ent.label_].append(ent.text)
                entCnt[ent.label_] += 1
                valCnt[ent.text] += 1
                parts.extend([v.text for v in doc[prevEnd:ent.start]])
                parts.append(f"[{ent.label_} : {ent.text}]")
                prevEnd = ent.end
                print(f"ent : {ent.text}, {ent.label_}")
            parts.extend([v.text for v in doc[prevEnd:]])
            ptn = ' '.join(parts)
            ptns.append(ptn)
            print(f"Ptn : {ptn}")

        if topk_slots > 0:
            import json
            topk_entities = collections.defaultdict(list)
            for key, value in ents.items():
                topk_entities[key] = collections.Counter(value).most_common(topk_slots)

            print(json.dumps(topk_entities, sort_keys=True, indent=4))

        cntr = 0
        for i, txt in enumerate(txts):
            ptn = ptns[i]
            if topk_slots > 0:
                cands = set(_augment_slots(ptn, topk_entities) + [txt])
            else:
                cands = set([txt])
            for cand in cands:
                non_ptns.add(ptn)
                proced = _pattern_proc(ptn)
                ptnUnique.add(ptn)
                slot_cnt = proced.count("{SLOT}")
                out_line = f"{_txt_proc(cand)},{_txt_proc(proced)},{slot_cnt}\n"
                print(f"proc : {out_line}")
                f.write(out_line)
                cntr += 1

    print(entCnt)
    print(ents)
    print(f"{cntr} lines written")
    print(f"Template 5 : {ptnUnique[0:5]}")
    print(f"Entities : {len(entCnt.keys())}, Values : {len(valCnt.keys())}, Raw Templates : {len(non_ptns)}, Templates : {len(ptnUnique)}")

if __name__ == "__main__":
    args = parse_args().parse_args()
    main(**args.__dict__)