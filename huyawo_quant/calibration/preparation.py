from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Protocol, cast

from huyawo_quant.contracts import CalibrationManifest


class _CalibrationDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


class _CalibrationTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> object: ...


def _load_runtime_components() -> tuple[
    Callable[..., object],
    Callable[..., object],
]:
    try:
        torch_data = import_module("torch.utils.data")
        transformers = import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "calibration preparation requires the baseline runtime dependencies"
        ) from error

    data_loader = getattr(torch_data, "DataLoader", None)
    collator = getattr(transformers, "default_data_collator", None)

    if not callable(data_loader):
        raise RuntimeError("torch.utils.data.DataLoader is unavailable")

    if not callable(collator):
        raise RuntimeError("transformers.default_data_collator is unavailable")

    return (
        cast(Callable[..., object], data_loader),
        cast(Callable[..., object], collator),
    )


def _normalize_token_sequence(
    value: object,
    *,
    field_name: str,
    source_index: int,
) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{field_name} for calibration row {source_index} must be a sequence of integers"
        )

    normalized: list[int] = []

    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"{field_name} for calibration row {source_index} must contain only integers"
            )

        normalized.append(item)

    if not normalized:
        raise ValueError(f"{field_name} for calibration row {source_index} must not be empty")

    return normalized


def _normalize_chat_messages(
    value: object,
    *,
    source_index: int,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"chat_template source for calibration row {source_index} "
            "must be a non-empty list of messages"
        )

    messages: list[dict[str, object]] = []

    for message_index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"chat message {message_index} for calibration row {source_index} must be a mapping"
            )

        normalized: dict[str, object] = {}

        for key, item in message.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"chat message {message_index} for calibration row "
                    f"{source_index} contains a non-string key"
                )

            normalized[key] = item

        role = normalized.get("role")
        content = normalized.get("content")

        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"chat message {message_index} for calibration row "
                f"{source_index} must contain a non-empty string role"
            )

        if not isinstance(content, str):
            raise ValueError(
                f"chat message {message_index} for calibration row "
                f"{source_index} must contain string content"
            )

        messages.append(normalized)

    return messages


def _render_source(
    *,
    manifest: CalibrationManifest,
    row: Mapping[str, object],
    tokenizer: _CalibrationTokenizer,
    source_index: int,
) -> str:
    source_column = manifest.preprocessing.source_column

    if source_column not in row:
        raise ValueError(
            f"calibration row {source_index} is missing source column {source_column!r}"
        )

    source_value = row[source_column]

    if manifest.preprocessing.render_mode == "raw_text":
        if not isinstance(source_value, str):
            raise ValueError(f"raw_text source for calibration row {source_index} must be a string")

        return source_value

    messages = _normalize_chat_messages(
        source_value,
        source_index=source_index,
    )

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=manifest.preprocessing.add_generation_prompt,
    )

    if not isinstance(rendered, str):
        raise ValueError(
            f"chat template for calibration row {source_index} must render to a string"
        )

    return rendered


def _tokenize_source(
    *,
    manifest: CalibrationManifest,
    rendered: str,
    tokenizer: _CalibrationTokenizer,
    source_index: int,
) -> dict[str, list[int]]:
    encoded = tokenizer(
        rendered,
        add_special_tokens=manifest.preprocessing.add_special_tokens,
        truncation=manifest.preprocessing.truncation,
        max_length=manifest.max_sequence_length,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,
    )

    if not isinstance(encoded, Mapping):
        raise ValueError(f"tokenizer output for calibration row {source_index} must be a mapping")

    input_ids = _normalize_token_sequence(
        encoded.get("input_ids"),
        field_name="input_ids",
        source_index=source_index,
    )

    attention_mask = _normalize_token_sequence(
        encoded.get("attention_mask"),
        field_name="attention_mask",
        source_index=source_index,
    )

    if len(input_ids) != len(attention_mask):
        raise ValueError(
            f"input_ids and attention_mask lengths differ for calibration row {source_index}"
        )

    if len(input_ids) > manifest.max_sequence_length:
        if manifest.preprocessing.truncation:
            raise ValueError(
                f"tokenizer failed to enforce max_sequence_length for "
                f"calibration row {source_index}"
            )

        raise ValueError(
            f"calibration row {source_index} exceeds max_sequence_length "
            "while truncation is disabled"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _prepare_features(
    *,
    manifest: CalibrationManifest,
    dataset: _CalibrationDataset,
    tokenizer: _CalibrationTokenizer,
) -> tuple[dict[str, list[int]], ...]:
    dataset_size = len(dataset)
    features: list[dict[str, list[int]]] = []
    total_token_count = 0

    for source_index in manifest.sample_indices:
        if source_index >= dataset_size:
            raise IndexError(
                f"calibration sample index {source_index} is outside dataset length {dataset_size}"
            )

        row = dataset[source_index]

        if not isinstance(row, Mapping):
            raise ValueError(f"calibration row {source_index} must be a mapping")

        rendered = _render_source(
            manifest=manifest,
            row=row,
            tokenizer=tokenizer,
            source_index=source_index,
        )

        feature = _tokenize_source(
            manifest=manifest,
            rendered=rendered,
            tokenizer=tokenizer,
            source_index=source_index,
        )

        total_token_count += len(feature["input_ids"])

        if manifest.token_budget is not None and total_token_count > manifest.token_budget:
            raise ValueError(
                f"prepared calibration tokens exceed token_budget {manifest.token_budget}"
            )

        features.append(feature)

    if len(features) != manifest.sample_count:
        raise RuntimeError("prepared calibration sample count diverged from CalibrationManifest")

    return tuple(features)


def prepare_calibration_dataloader(
    *,
    manifest: CalibrationManifest,
    dataset: _CalibrationDataset,
    tokenizer: _CalibrationTokenizer,
) -> object:
    """Replay one accepted CalibrationManifest into a sequential DataLoader."""

    features = _prepare_features(
        manifest=manifest,
        dataset=dataset,
        tokenizer=tokenizer,
    )

    data_loader_factory, collator = _load_runtime_components()

    return data_loader_factory(
        features,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )
