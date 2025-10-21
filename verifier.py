import logging
import re
from typing import Any, Iterable, Tuple, TypedDict, Literal

import verifiers as vf
from datasets import load_dataset, Dataset


FeatureMode = Literal["name", "id"]


class CapaRow(TypedDict):
    disasm: list[str]
    capa: list[str]


_system_prompt_prefix = "You are a reverse engineering expert. Analyze the provided disassembly code and identify which of the following features are present:\n\n"
CAPA_SYSTEM_PROMPT_NAME = f"""{_system_prompt_prefix}{{features}}\n\nReturn a comma-separated list of features. Put your final answer within \\boxed{{}}. For example, if the features are "Example A" and "Example B", you would respond with \\boxed{{Example A, Example B}}."""
CAPA_SYSTEM_PROMPT_ID = f"""{_system_prompt_prefix}{{features}}\n\nReturn a comma-separated list of feature IDs. Put your final answer within \\boxed{{}}. For example: \\boxed{{5, 22}}."""


_SPLIT_RE = re.compile(r"\s*(?:,|;)\s*")
_STRIP_PUNCT_RE = re.compile(r"^[\s{\[\(]+|[\s}\]\).]+$")
_ID_RE = re.compile(r"\d+")


def _normalize_feature(s: str) -> str:
    s = _STRIP_PUNCT_RE.sub("", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _format_feature_for_prompt(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def _parse_id_set(s: str) -> set[int]:
    return {int(tok.lstrip("0") or "0") for tok in _ID_RE.findall(s)}


def _parse_name_set(s: str) -> set[str]:
    txt = s.strip()
    tokens = [t for t in _SPLIT_RE.split(txt) if t.strip()]
    return {_normalize_feature(t) for t in tokens}


def _build_question(disasm_lines: Iterable[str]) -> str:
    return (
        "What features are present in the following function?\n\n"
        "```\n" + "\n".join(disasm_lines) + "\n```"
    )


def _extract_last_boxed_answer(text: str) -> str:
    def extract_boxed_answer(text: str, find_last: bool = False) -> str:
        def find_matching_brace(s: str, start: int) -> int:
            count = 1
            i = start
            while i < len(s) and count > 0:
                if s[i] == "{":
                    count += 1
                elif s[i] == "}":
                    count -= 1
                i += 1
            return i - 1 if count == 0 else -1

        # Find \boxed{
        if find_last:
            boxed_start = text.rfind("\\boxed{")
        else:
            boxed_start = text.find("\\boxed{")
        if boxed_start == -1:
            return text
        # Find the content between the braces
        content_start = boxed_start + 7  # len('\\boxed{')
        closing_brace = find_matching_brace(text, content_start)

        if closing_brace == -1:
            return text

        return text[content_start:closing_brace]

    # https://github.com/willccbb/verifiers/pull/310
    # return vf.extract_boxed_answer(text, find_last=True)

    return extract_boxed_answer(text, find_last=True)


class CapaRubric(vf.Rubric):
    def __init__(
        self,
        feature_mode: FeatureMode,
        funcs: list[vf.RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: vf.Parser | None = None,
    ):
        parser = parser or vf.ThinkParser(extract_fn=_extract_last_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.feature_mode = feature_mode
        self.add_reward_func(self.correct_features_reward_func)

    def _to_gold_pred_sets(self, gold_str: str, raw_pred_str: str) -> tuple[set, set]:
        if self.feature_mode == "id":
            gold = _parse_id_set(gold_str)
            pred = _parse_id_set(raw_pred_str)
        else:
            gold = _parse_name_set(gold_str)
            pred = _parse_name_set(raw_pred_str)
        return gold, pred

    def correct_features_reward_func(
        self, parser: vf.Parser, completion: vf.Messages, answer: str, **kwargs
    ) -> float:
        try:
            raw = (parser.parse_answer(completion) or "").strip()
            gold, pred = self._to_gold_pred_sets(answer, raw)

            if not gold and not pred:
                logging.debug("Both gold and pred empty > reward=1.0")
                return 1.0
            if not pred:
                reward = 0.0
                logging.debug(
                    f"No prediction. raw_response={raw} gold={answer} > reward={reward:.4f}"
                )
                return reward

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)

            denom = 2 * tp + fp + fn
            reward = (2 * tp / denom) if denom > 0 else 0.0

            logging.debug(
                f"\n\nOriginal: {completion}\nResponse: {raw}\nAnswer: {answer}\nTP: {tp} FP: {fp} FN: {fn}\nReward (F1): {reward:.4f}\n"
            )
            return reward
        except BaseException as e:
            logging.warning("Error in reward function: %s", e)
            return 0.0


def _feature_vocab_from_raw(dataset: Iterable[CapaRow]) -> list[str]:
    seen: set[str] = set()
    for row in dataset:
        for f in row.get("capa", []) or []:
            seen.add(_normalize_feature(f))
    return sorted(seen)


def _get_column_names(ds: Any) -> list[str]:
    cols = getattr(ds, "column_names", None)
    if isinstance(cols, list):
        return cols
    feats = getattr(ds, "features", None)
    if getattr(feats, "keys", None):
        try:
            return list(feats.keys())
        except Exception:
            return []
    return []


def preprocess_capa(
    x: CapaRow,
    feature_mode: FeatureMode,
    feature_to_id: dict[str, int] | None = None,
) -> CapaRow:
    canonical = sorted({_normalize_feature(f) for f in x["capa"]})
    prompt = _build_question(x["disasm"])

    if feature_mode == "id":
        ids = sorted({feature_to_id[f] for f in canonical})
        answer = ", ".join(str(i) for i in ids)
    else:
        answer = ", ".join(canonical)

    return {"question": prompt, "answer": answer}


def _filter_capa_entry(
    x: CapaRow, min_inst: int, max_inst: int, min_feat: int, max_feat: int
) -> bool:
    lines = x.get("disasm")
    features = x.get("capa")
    if not isinstance(lines, list) or not isinstance(features, list):
        return False
    if not (min_inst <= len(lines) <= max_inst):
        return False
    if not (min_feat <= len(features) <= max_feat):
        return False
    if not all(isinstance(s, str) and s for s in lines):
        return False
    if not all(isinstance(f, str) and f.strip() for f in features):
        return False
    return True


def load_capa_dataset(
    feature_mode: FeatureMode,
    split: str = "train",
    n: int = 10_000,
    seed: int = 1337,
    min_disasm_inst: int = 10,
    max_disasm_inst: int = 250,
    min_capa_features: int = 1,
    max_capa_features: int = 4,
) -> Tuple[Dataset, list[str], dict[str, int]]:
    # streaming=True because EMBER2024-capa is huge
    raw = load_dataset("joyce8/EMBER2024-capa", split=split, streaming=True)

    raw = raw.filter(
        _filter_capa_entry,
        fn_kwargs={
            "min_inst": min_disasm_inst,
            "max_inst": max_disasm_inst,
            "min_feat": min_capa_features,
            "max_feat": max_capa_features,
        },
    )

    raw = raw.shuffle(seed=seed, buffer_size=10_000).take(n)

    # convert to a regular dataset after filtering
    raw = Dataset.from_list(list(raw))

    feature_vocab = _feature_vocab_from_raw(raw)
    feature_to_id = {feat: i for i, feat in enumerate(feature_vocab, start=1)}

    processed = raw.map(
        preprocess_capa,
        fn_kwargs={"feature_mode": feature_mode, "feature_to_id": feature_to_id},
        remove_columns=[
            c for c in _get_column_names(raw) if c not in ("capa", "disasm")
        ],
        num_proc=10,
    )
    processed = processed.remove_columns(
        [c for c in _get_column_names(processed) if c not in ("question", "answer")]
    )

    return processed, feature_vocab, feature_to_id


def _build_system_prompt(
    feature_mode: FeatureMode,
    feature_vocab: list[str],
    feature_to_id: dict[str, int],
) -> str:
    if feature_mode == "id":
        lines = [
            f"{fid}. {_format_feature_for_prompt(feat)}"
            for feat, fid in sorted(feature_to_id.items(), key=lambda kv: kv[1])
        ]
        feature_list = "\n".join(lines) if lines else "(none)"
        return CAPA_SYSTEM_PROMPT_ID.replace("{features}", feature_list).strip()
    else:
        pretty = [_format_feature_for_prompt(f) for f in feature_vocab]
        feature_list = "- " + "\n- ".join(pretty) if pretty else "(none)"
        return CAPA_SYSTEM_PROMPT_NAME.replace("{features}", feature_list).strip()


def load_environment(
    use_think: bool = True,
    feature_mode: FeatureMode = "id",
) -> vf.Environment:
    dataset, feature_vocab, feature_to_id = load_capa_dataset(
        feature_mode=feature_mode, n=8000, seed=1337
    )

    updated_system_prompt = _build_system_prompt(
        feature_mode=feature_mode,
        feature_vocab=feature_vocab,
        feature_to_id=feature_to_id,
    )
    logging.debug(f"System prompt:\n{updated_system_prompt}\n")

    if use_think:
        parser = vf.ThinkParser(extract_fn=_extract_last_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=_extract_last_boxed_answer)

    rubric = CapaRubric(parser=parser, feature_mode=feature_mode)

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=updated_system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
