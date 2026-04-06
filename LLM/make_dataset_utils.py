from transformers import AutoTokenizer
from datasets import load_dataset


def _resolve_max_length(tokenizer, max_length=None):
    if max_length is not None:
        return max_length
    model_max_length = getattr(tokenizer, "model_max_length", None)
    # Qwen-like tokenizers may expose very large context windows (e.g. 131072),
    # but this pipeline expects fixed-size CLM training samples similar to legacy setup.
    if model_max_length is None or model_max_length > 4096:
        return 544
    return model_max_length


def json_dfs_without_bracket(json, text_list: list):
    if isinstance(json, dict):
        json = dict(sorted(json.items()))
        for key, val in json.items():
            text_list.append(key)
            json_dfs_without_bracket(val, text_list)
    elif isinstance(json, (list, tuple)):
        for it in json:
            json_dfs_without_bracket(it, text_list)
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def json_dfs_with_bracket(json, text_list: list):
    if isinstance(json, dict):
        text_list.append("{")
        json = dict(sorted(json.items()))
        for idx, (key, val) in enumerate(json.items()):
            if idx != 0:
                text_list.append(",")
            text_list.append(key)
            json_dfs_with_bracket(val, text_list)
        text_list.append("}")
    elif isinstance(json, list):
        text_list.append("[")
        for idx, it in enumerate(json):
            if idx != 0:
                text_list.append(",")
            json_dfs_with_bracket(it, text_list)
        text_list.append("]")
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def json_to_token(json_lines):
    token_list = []
    for json_line in json_lines:
        text_list = []
        json_dfs_without_bracket(json_line["text"], text_list)
        json_line["text"] = " ".join(text_list)
        token_list.append(json_line)
    return token_list


def _load_raw_datasets(file, valid_percentage=5, validation_file=None):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        return load_dataset(
            extension,
            data_files=data_files,
            keep_in_memory=True
        )

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )
    if valid_percentage > 0:
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{valid_percentage}%]",
            keep_in_memory=True
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{valid_percentage}%:]",
            keep_in_memory=True
        )
    else:
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train",
            keep_in_memory=True
        )
    return raw_datasets


def make_dataset(
    file,
    dataset_path,
    tokenizer_path,
    for_clm_or_mlm,
    valid_percentage=5,
    max_length=None,
    validation_file=None,
):
    raw_datasets = _load_raw_datasets(
        file,
        valid_percentage=valid_percentage,
        validation_file=validation_file,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    effective_max_length = _resolve_max_length(tokenizer, max_length=max_length)

    column_names = list(raw_datasets["train"].features)
    if for_clm_or_mlm == "clm":
        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=effective_max_length,
            )
            output["labels"] = output["input_ids"].copy()
            output.pop("token_type_ids", None)
            return output
    elif for_clm_or_mlm == "mlm":
        column_names.remove("labels")
        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=effective_max_length,
            )
            return output
    else:
        assert(False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)


def make_dataset_test(file, dataset_path, tokenizer_path, for_clm_or_mlm, max_length=None):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    effective_max_length = _resolve_max_length(tokenizer, max_length=max_length)

    if for_clm_or_mlm == "clm":
        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=effective_max_length,
            )
            input_ids = output["input_ids"]
            for inp in input_ids:
                del inp[-1]
            attention_mask = output["attention_mask"]
            for mask in attention_mask:
                del mask[-1]
            output.pop("token_type_ids", None)
            return output
    elif for_clm_or_mlm == "mlm":
        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=effective_max_length,
            )
            return output
    else:
        assert(False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)