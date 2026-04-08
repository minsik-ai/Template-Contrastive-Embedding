from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import transformers

from transformers import MODEL_FOR_MASKED_LM_MAPPING, BertForPreTraining

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import logging
import os
import sys
import pandas as pd

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForPreTraining,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.file_utils import cached_property

from tadse.models import BertForPatternCL, Qwen3ForPatternCL, RobertaForPatternCL
from tadse.trainers import CLTrainer

logger = logging.getLogger(__name__)
ENTITY_PAD_MARK = "[PAD]"


# def torch_required(func):
#     # Chose a different decorator name than in tests so it's clear they are not the same.
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if is_torch_available():
#             return func(*args, **kwargs)
#         else:
#             raise ImportError(f"Method `{func.__name__}` requires PyTorch.")
#
#     return wrapper

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    hidden_size: int = field(
        default=768,
        metadata={
            "help" : "hidden size (set to pooler output)"
        }
    )
    simcse_utt_temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    simcse_ptn_temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pairwise_temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, pad_left_no_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    apply_ptn_transform: bool = field(
        default=False,
        metadata={
            "help": "Apply Pattern-only transform."
        }
    )
    simcse_utt_loss_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Utterance loss ratio for SimCSE."
        }
    )
    simcse_ptn_loss_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Pattern loss ratio for SimCSE."
        }
    )
    pairwise_loss_ratio: float = field(
        default=0.01,
        metadata={
            "help": "Pairwise loss ratio."
        }
    )
    pairwise_neg_type: str = field(
        default="pattern",
        metadata={
            "help": "Negative type for pairwise loss. Values could be 'pattern', 'utterance' or 'balanced'."
        }
    )
    mlm_loss_ratio: float = field(
        default=0,
        metadata={
            "help": "MLM loss ratio for SimCSE."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    dataset_cache_path: Optional[str] = field(
        default="./data/", metadata={"help": "The path to dataset."}
    )

    slot_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to parquet slot file with column 'slot'."}
    )

    top_k_slots: Optional[int] = field(
        default=0,
        metadata={
            "help": "Top k slots to maintain."
        }
    )

    one_slot_token: bool = field(
        default=False,
        metadata={
            "help": "Add a {SLOT} token to Tokenizer."
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    resume_from_checkpoint: bool = field(
        default=False,
    )
    group_by_length: bool = field(
        default=False,
    )
    eval_transfer: bool = field(
        default=False,
    )
    distributed_state: Optional[Dict] = field(
        default=None,
    )

    @cached_property
    # @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()
        else:
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError(
                        "--deepspeed requires deepspeed: `pip install deepspeed`."
                    )
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


# data collator
@dataclass
class OurDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
            self,
            features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        special_keys = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "mlm_input_ids",
            "mlm_labels",
        ]
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]["input_ids"])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append(
                    {
                        k: feature[k][i] if k in special_keys else feature[k]
                        for k in feature
                    }
                )

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {
            k: batch[k].view(bs, num_sent, -1)
            if k in special_keys
            else batch[k].view(bs, num_sent, -1)[:, 0]
            for k in batch
        }

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def main():
    cls_args: Optional["OurTrainingArguments"] = None
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }
    if model_args.pooler_type == "pad_left_no_pooler":
        print("PAD LEFT NO POOLER")
        tokenizer_kwargs["padding_side"] = "left"

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if data_args.one_slot_token:
        test_output = tokenizer.encode("{SLOT} order")
        logger.info(f"Prev slot result : {test_output}")

        num_added = tokenizer.add_tokens(["{SLOT}"], special_tokens=True)
        # TEST
        test_output = tokenizer.encode("{SLOT} order")
        logger.info(f"Added : {num_added}, New slot result : {test_output}")
    elif data_args.slot_file:
        # Load slot list
        slot_df = pd.read_parquet(data_args.slot_file, columns=["slot"])
        slot_list = slot_df['slot'].values.tolist()

        topk_slots = data_args.top_k_slots
        raw_slots = slot_list[:topk_slots]
        replace_slots = slot_list[topk_slots:]

        ex_slot = "{SLOT} order"
        test_output = tokenizer.encode(ex_slot)
        logger.info(f"Prev slot result : {test_output}")

        ex_raw1 = f"{raw_slots[0]} order"
        test_output = tokenizer.encode(ex_raw1)
        logger.info(f"Prev slot result : {test_output}")

        ex_replace1 = f"{replace_slots[0]} order"
        test_output = tokenizer.encode(ex_replace1)
        logger.info(f"Prev slot result : {test_output}")

        num_added = tokenizer.add_tokens(raw_slots, special_tokens=True)
        logger.info(f"Added Slots : {num_added}")

        test_output = tokenizer.encode(ex_slot)
        logger.info(f"New slot result : {test_output}")

        test_output = tokenizer.encode(ex_raw1)
        logger.info(f"New slot result : {test_output}")

        test_output = tokenizer.encode(ex_replace1)
        logger.info(f"New slot result : {test_output}")

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.dataset_cache_path,
                                delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.dataset_cache_path)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # TODO : Consider loading pre-trained pattern embeddings?
    #  Probably possible since pattern size ~2000 for single slot.
    # load pretrained entity embeddings
    # embedding = Wikipedia2Vec.load(os.path.join(cwd, data_args.wikipedia2vec_path))
    # dim_size = embedding.syn0.shape[1]
    # OmegaConf.set_struct(model_args, True)
    # with open_dict(model_args):
    #     model_args.entity_emb_shape = (len(entity_vocab), dim_size)
    # entity_embeddings = np.random.uniform(
    #     low=-0.05, high=0.05, size=model_args.entity_emb_shape
    # )
    # entity_embeddings[0] = np.zeros(dim_size)
    # cnt = 0
    # if model_args.init_wiki2emb:
    #     for entity, index in tqdm(entity_vocab.items()):
    #         try:
    #             entity_embeddings[index] = embedding.get_entity_vector(entity)
    #             cnt += 1
    #         except KeyError:
    #             pass

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if data_args.one_slot_token:
        test_output = tokenizer.encode("{SLOT} order")
        logger.info(f"Prev slot result : {test_output}")

        num_added = tokenizer.add_tokens(["{SLOT}"], special_tokens=True)
        # TEST
        test_output = tokenizer.encode("{SLOT} order")
        logger.info(f"Added : {num_added}, New slot result : {test_output}")
    elif data_args.slot_file:
        # Load slot list
        slot_df = pd.read_parquet(data_args.slot_file, columns=["slot"])
        slot_list = slot_df['slot'].values.tolist()

        topk_slots = data_args.top_k_slots
        raw_slots = slot_list[:topk_slots]
        replace_slots = slot_list[topk_slots:]

        ex_slot = "{SLOT} order"
        test_output = tokenizer.encode(ex_slot)
        logger.info(f"Prev slot result : {test_output}")

        ex_raw1 = f"{raw_slots[0]} order"
        test_output = tokenizer.encode(ex_raw1)
        logger.info(f"Prev slot result : {test_output}")

        ex_replace1 = f"{replace_slots[0]} order"
        test_output = tokenizer.encode(ex_replace1)
        logger.info(f"Prev slot result : {test_output}")

        num_added = tokenizer.add_tokens(raw_slots, special_tokens=True)
        logger.info(f"Added Slots : {num_added}")

        test_output = tokenizer.encode(ex_slot)
        logger.info(f"New slot result : {test_output}")

        test_output = tokenizer.encode(ex_raw1)
        logger.info(f"New slot result : {test_output}")

        test_output = tokenizer.encode(ex_replace1)
        logger.info(f"New slot result : {test_output}")

    # TODO : Consider separate models for utterances and patterns
    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            model = RobertaForPatternCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )

        elif "Qwen3" in model_args.model_name_or_path:
            model = Qwen3ForPatternCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:

            model = BertForPatternCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        # else:
        #     # TODO : Prob DiffCSE support needed in some time....
        #     raise NotImplementedError

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    assert len(column_names) >= 2

    utt_cname = column_names[0]
    ptn_cname = column_names[1]

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[utt_cname])
        assert len(examples[utt_cname]) == len(examples[ptn_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[utt_cname][idx] is None:
                examples[utt_cname][idx] = " "
            if examples[ptn_cname][idx] is None:
                examples[ptn_cname][idx] = " "

        # Duplicating 2 times to enable dropout CL with itself.
        sentences = examples[utt_cname] + examples[utt_cname] + examples[ptn_cname] + examples[ptn_cname]

        # Slot replacer
        if data_args.slot_file:
            for slot in replace_slots:
                for idx in range(len(sentences)):
                    if slot in sentences[idx]:
                        sentences[idx] = sentences[idx].replace(slot, "{SLOT}")
                        print(f"Slot replaced : {slot}, sentence : {sentences[idx]}")
        elif data_args.one_slot_token:
            import re
            sentences = [re.sub('{[^}]*}', '{SLOT}', sent) for sent in sentences]
            # print(sentences)

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        for key in sent_features:
            # Pairwise - utterance x 2 + pattern x 2
            features[key] = [[sent_features[key][i], sent_features[key][i + total],
                              sent_features[key][i + 2 * total], sent_features[key][i + 3 * total]] \
                             for i in range(total)]

        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (
                    model_args.model_name_or_path is not None
                    and os.path.isdir(model_args.model_name_or_path)
            )
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # TODO : Consider correct eval setting ?
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
