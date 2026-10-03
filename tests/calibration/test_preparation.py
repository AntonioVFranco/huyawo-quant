from __future__ import annotations

import copy
import json

import huyawo_quant.calibration as calibration_package
import pytest
from huyawo_quant.calibration import prepare_calibration_dataloader
from huyawo_quant.contracts import CalibrationManifest
from torch.utils.data import DataLoader

BASE_MANIFEST_PAYLOAD = {
    "dataset": {
        "config_name": None,
        "dataset_id": "HuggingFaceH4/ultrachat_200k",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
        "source": "huggingface",
        "split": "train_sft",
    },
    "max_sequence_length": 2048,
    "preprocessing": {
        "add_generation_prompt": False,
        "add_special_tokens": False,
        "render_mode": "chat_template",
        "source_column": "messages",
        "truncation": True,
    },
    "sample_count": 3,
    "sample_indices": [7, 2, 11],
    "schema_version": "1",
    "seed": 42,
    "shuffle": True,
    "token_budget": 4096,
    "tokenizer": {
        "requested_revision": "main",
        "resolved_revision": "fedcba9876543210",
        "source": "huggingface",
        "tokenizer_id": "Qwen/Qwen2.5-0.5B-Instruct",
    },
}


class FakeTokenizer:
    def __init__(self) -> None:
        self.tokenize_calls: list[dict[str, object]] = []
        self.chat_template_calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.tokenize_calls.append(
            {
                "text": text,
                **kwargs,
            }
        )

        add_special_tokens = kwargs["add_special_tokens"]
        truncation = kwargs["truncation"]
        max_length = kwargs["max_length"]

        assert isinstance(add_special_tokens, bool)
        assert isinstance(truncation, bool)
        assert isinstance(max_length, int)

        input_ids = [(ord(character) % 97) + 1 for character in text]

        if add_special_tokens:
            input_ids = [101, *input_ids, 102]

        if truncation:
            input_ids = input_ids[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> str:
        self.chat_template_calls.append(
            {
                "conversation": copy.deepcopy(conversation),
                **kwargs,
            }
        )

        parts = [f"{message['role']}:{message['content']}" for message in conversation]

        if kwargs["add_generation_prompt"]:
            parts.append("assistant:")

        return "|".join(parts)


def make_dataset() -> list[dict[str, object]]:
    return [
        {
            "text": f"row-{index}",
            "messages": [
                {
                    "role": "user",
                    "content": f"row-{index}",
                }
            ],
        }
        for index in range(12)
    ]


def make_manifest(
    *,
    source_column: str = "text",
    render_mode: str = "raw_text",
    add_generation_prompt: bool = False,
    add_special_tokens: bool = False,
    truncation: bool = True,
    sample_indices: tuple[int, ...] = (7, 2, 11),
    max_sequence_length: int = 64,
    token_budget: int | None = None,
    seed: int = 17,
    shuffle: bool = True,
) -> CalibrationManifest:
    payload = copy.deepcopy(BASE_MANIFEST_PAYLOAD)

    payload["preprocessing"] = {
        "source_column": source_column,
        "render_mode": render_mode,
        "add_generation_prompt": add_generation_prompt,
        "add_special_tokens": add_special_tokens,
        "truncation": truncation,
    }
    payload["sample_count"] = len(sample_indices)
    payload["sample_indices"] = list(sample_indices)
    payload["max_sequence_length"] = max_sequence_length
    payload["token_budget"] = token_budget
    payload["seed"] = seed
    payload["shuffle"] = shuffle

    return CalibrationManifest.model_validate_json(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )


def test_calibration_package_exports_preparation_entrypoint() -> None:
    assert calibration_package.__all__ == ["prepare_calibration_dataloader"]
    assert calibration_package.prepare_calibration_dataloader is prepare_calibration_dataloader


def test_prepare_calibration_dataloader_replays_exact_sample_indices_in_order() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest()

    loader = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 1
    assert loader.num_workers == 0
    assert len(loader.dataset) == 3
    assert [call["text"] for call in tokenizer.tokenize_calls] == [
        "row-7",
        "row-2",
        "row-11",
    ]
    assert len(list(loader)) == 3


def test_prepare_calibration_dataloader_raw_text_forwards_tokenization_contract() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(3,),
        add_special_tokens=True,
        truncation=True,
        max_sequence_length=5,
    )

    prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    assert tokenizer.tokenize_calls == [
        {
            "text": "row-3",
            "add_special_tokens": True,
            "truncation": True,
            "max_length": 5,
            "padding": False,
            "return_attention_mask": True,
            "return_tensors": None,
        }
    ]


def test_prepare_calibration_dataloader_chat_template_renders_before_tokenization() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        source_column="messages",
        render_mode="chat_template",
        add_generation_prompt=True,
        sample_indices=(4,),
    )

    prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    assert tokenizer.chat_template_calls == [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": "row-4",
                }
            ],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert tokenizer.tokenize_calls[0]["text"] == "user:row-4|assistant:"


def test_prepare_calibration_dataloader_truncates_when_enabled() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(10,),
        truncation=True,
        max_sequence_length=3,
    )

    loader = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    feature = loader.dataset[0]

    assert len(feature["input_ids"]) == 3
    assert len(feature["attention_mask"]) == 3


def test_prepare_calibration_dataloader_rejects_overlength_when_truncation_disabled() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(10,),
        truncation=False,
        max_sequence_length=3,
    )

    with pytest.raises(
        ValueError,
        match="exceeds max_sequence_length while truncation is disabled",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=make_dataset(),
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_accepts_token_budget_at_exact_cap() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(0,),
        token_budget=len("row-0"),
    )

    loader = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    assert len(loader.dataset) == 1


def test_prepare_calibration_dataloader_rejects_token_budget_overflow() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(0,),
        token_budget=len("row-0") - 1,
    )

    with pytest.raises(
        ValueError,
        match="prepared calibration tokens exceed token_budget",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=make_dataset(),
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_rejects_missing_source_column() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(0,),
    )
    dataset = [{"other": "value"}]

    with pytest.raises(
        ValueError,
        match="is missing source column",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=dataset,
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_rejects_non_string_raw_text() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(0,),
    )
    dataset = [{"text": 123}]

    with pytest.raises(
        ValueError,
        match="raw_text source.*must be a string",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=dataset,
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_rejects_invalid_chat_messages() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        source_column="messages",
        render_mode="chat_template",
        sample_indices=(0,),
    )
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": 123,
                }
            ]
        }
    ]

    with pytest.raises(
        ValueError,
        match="must contain string content",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=dataset,
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_rejects_out_of_range_sample_index() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(12,),
    )

    with pytest.raises(
        IndexError,
        match="outside dataset length",
    ):
        prepare_calibration_dataloader(
            manifest=manifest,
            dataset=make_dataset(),
            tokenizer=tokenizer,
        )


def test_prepare_calibration_dataloader_contains_only_token_features() -> None:
    tokenizer = FakeTokenizer()
    manifest = make_manifest(
        sample_indices=(1,),
    )

    loader = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=tokenizer,
    )

    feature = loader.dataset[0]

    assert set(feature) == {
        "input_ids",
        "attention_mask",
    }
    assert "text" not in feature
    assert "messages" not in feature


def test_prepare_calibration_dataloader_is_deterministic_for_same_inputs() -> None:
    manifest = make_manifest()

    first = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=FakeTokenizer(),
    )
    second = prepare_calibration_dataloader(
        manifest=manifest,
        dataset=make_dataset(),
        tokenizer=FakeTokenizer(),
    )

    assert tuple(first.dataset) == tuple(second.dataset)


def test_manifest_seed_and_shuffle_do_not_resample_replay() -> None:
    first_tokenizer = FakeTokenizer()
    second_tokenizer = FakeTokenizer()

    first = prepare_calibration_dataloader(
        manifest=make_manifest(
            seed=1,
            shuffle=True,
        ),
        dataset=make_dataset(),
        tokenizer=first_tokenizer,
    )

    second = prepare_calibration_dataloader(
        manifest=make_manifest(
            seed=999,
            shuffle=False,
        ),
        dataset=make_dataset(),
        tokenizer=second_tokenizer,
    )

    assert [call["text"] for call in first_tokenizer.tokenize_calls] == [
        "row-7",
        "row-2",
        "row-11",
    ]
    assert [call["text"] for call in second_tokenizer.tokenize_calls] == [
        "row-7",
        "row-2",
        "row-11",
    ]
    assert tuple(first.dataset) == tuple(second.dataset)
