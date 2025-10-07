"""
Implementation of data loading and processing functions for math datasets.

"""

import re
from typing import Iterable
from datasets import Dataset, Value, Features, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .grader import grade_answer

# tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"
def set_tokenizer_name(name):
    global tokenizer_name
    tokenizer_name = name

SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$.\n\nWhere [answer] is just the final number or expression that solves the problem."

_CONCLUSION_PREFIX = re.compile(
    r"Therefore,\s*the\s*final\s*answer\s*is:\s*(?:\$\s*)?\\boxed\{",
    flags=re.IGNORECASE,
)
_CONCLUSION_SUFFIX = re.compile(
    r"\s*\$?\s*\.\s*",
    flags=re.IGNORECASE,
)
_STEP_HEADER_PATTERN = re.compile(
    r"(?:(?:^)|(?:\n\n))\s*(?:[#*]+\s*)?Step[^0-9\r\n]*?(\d+)\b",
    flags=re.IGNORECASE,
)


def _system_chat_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Compose a chat-style prompt that always uses the math system instructions."""

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _extract_balanced_braces(text: str, start_idx: int) -> tuple[str, int | None]:
    """Return the content between balanced braces starting at ``start_idx``."""

    depth = 1
    cursor = start_idx
    buffer: list[str] = []

    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(buffer), cursor + 1
        buffer.append(char)
        cursor += 1

    return "".join(buffer), None


def _locate_conclusion(text: str) -> tuple[str, int, int] | None:
    """Locate the instruct-style conclusion and extract the boxed answer."""

    matches = list(_CONCLUSION_PREFIX.finditer(text))
    if not matches:
        return None

    prefix_match = matches[-1]
    answer_start = prefix_match.end()
    extracted, closing_idx = _extract_balanced_braces(text, answer_start)
    if closing_idx is None:
        return None

    suffix_match = _CONCLUSION_SUFFIX.match(text[closing_idx:])
    if not suffix_match:
        return None

    answer = extracted.strip()
    conclusion_end = closing_idx + suffix_match.end()
    return answer, prefix_match.start(), conclusion_end


def _last_boxed_span(text: str) -> tuple[int, int] | None:
    """Return the start/end indices of the last ``\boxed{...}`` block."""

    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    _, closing_idx = _extract_balanced_braces(text, start + len(marker))

    if closing_idx is None:
        return None

    return start, closing_idx


def _step_matches_before(text: str, limit: int) -> list[re.Match]:
    """Return step header matches that appear before ``limit``."""

    if limit <= 0:
        return []
    preamble = text[:limit]
    matches = list(_STEP_HEADER_PATTERN.finditer(preamble))
    return [m for m in matches if _requires_blank_line(preamble, m.start())]


def _requires_blank_line(text: str, pos: int) -> bool:
    """Ensure each step header is preceded by a blank line."""

    if pos <= 0:
        return True
    first_nl = text.rfind("\n", 0, pos)

    if first_nl == -1:
        return False
    if text[first_nl + 1:pos].strip():
        return False

    second_nl = text.rfind("\n", 0, first_nl)

    if second_nl == -1:
        return False
    return text[second_nl + 1:first_nl].strip() == ""


def _step_sequence(numbers: list[int]) -> bool:
    """Check that step numbers increase sequentially from 1."""

    if not numbers:
        return False
    expected = list(range(1, len(numbers) + 1))
    return numbers == expected


def _step_segments_have_content(text: str, matches: list[re.Match], limit: int) -> bool:
    """Ensure each step header is followed by non-empty prose before ``limit``."""

    if not matches:
        return False

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else limit

        if start >= end:
            return False
        if text[start:end].strip() == "":
            return False

    return True


def extract_last_integer(text: str) -> str:
    """Extract the last integer from the text for numeric fallback grading."""

    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]
    return ""


def extract_last_boxed_answer(text: str) -> str:
    """Return the content of the last ``\boxed{...}`` expression in ``text``."""

    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return ""

    content, closing_idx = _extract_balanced_braces(text, idx + len(marker))
    if closing_idx is None:
        return ""
    return content.strip()


def extract_instruct_answer(text: str) -> str:
    """Extract the boxed answer according to the instruct-style conclusion."""

    conclusion = _locate_conclusion(text)
    if conclusion:
        return conclusion[0]
    return ""


def answer_correct(text: str, answer: str) -> bool:
    """Determine whether ``text`` contains a mathematically correct answer."""

    candidates: Iterable[str] = (
        extract_instruct_answer(text),
        extract_last_boxed_answer(text),
        extract_last_integer(text),
        text,
    )

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    return any(grade_answer(candidate, answer) for candidate in unique)


def format_score(text: str) -> float:
    """Return the format reward (max 0.05) for well-structured completions."""

    boxed_span = _last_boxed_span(text)

    if boxed_span is None:
        return 0.0

    boxed_start, boxed_end = boxed_span
    matches = _step_matches_before(text, boxed_start)

    if not matches:
        return 0.0

    numbers = [int(match.group(1)) for match in matches]

    if not _step_sequence(numbers) or len(numbers) > 3:
        return 0.0
    if not _step_segments_have_content(text, matches, boxed_start):
        return 0.0

    conclusion = _locate_conclusion(text)
    span_end = conclusion[2] if conclusion is not None else boxed_end
    extra_len = len(text[span_end:].strip())
    alpha = 5e-4
    bonus = max(0.0, 1.0 - alpha * extra_len)
    return 0.05 * bonus


def format_correct(text: str) -> bool:
    """Boolean proxy mirroring ``format_score`` for downstream consumers."""

    return format_score(text) > 0.0


def filter_function(example, tokenizer: AutoTokenizer) -> bool:
    # Filter out examples that are too long
    prompt = example['prompt']
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')
    return tokenized_prompt['input_ids'].shape[1] <= 512


def get_math8k_questions(split: str = "train") -> Dataset:
    """Load Math8K prompts using the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("parquet", data_files=f"datasets/math8k/{split}.parquet")["train"]
    data = data.map(
        lambda x: {
            "prompt": _system_chat_prompt(tokenizer, x["question"]),
            "answer": x["gt_answer"],
        }
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def extract_gsm8k_answer(text: str) -> str | None:
    # Extracts the answer from the GSM8K dataset
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")


def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load GSM8K prompts using the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": _system_chat_prompt(tokenizer, x["question"]),
            "answer": extract_gsm8k_answer(x["answer"]),
        }
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def extract_math_answer(text: str) -> str:
    # Extracts the answer from the Math dataset
    if "\\boxed{" not in text:
        return None
    answer = text.split("\\boxed{")[-1]
    answer = answer.split("}")[0]
    return answer.strip()


def get_math_questions(split: str = "train") -> Dataset:
    """Load the MATH dataset prompts with the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    datasets = [load_dataset("EleutherAI/hendrycks_math", s, split=split) for s in subsets]
    data = concatenate_datasets(datasets)
    data = data.map(
        lambda x: {
            "prompt": _system_chat_prompt(tokenizer, x["problem"]),
            "answer": extract_math_answer(x["solution"]),
        }
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer}).shuffle(seed=42)
    return data


def get_math500_questions(split: str = "test") -> Dataset:
    """Load the Math500 dataset prompts with the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("HuggingFaceH4/MATH-500")[split]
    data = data.map(
        lambda x: {
            "prompt": _system_chat_prompt(tokenizer, x["problem"]),
            "answer": x["answer"],
        }
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def get_amc23_questions(split: str = "test") -> Dataset:
    """Load the AMC23 dataset prompts with the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("zwhe99/amc23")[split]
    data = data.map(
        lambda x: {
            "prompt": _system_chat_prompt(tokenizer, x["question"]),
            "answer": str(int(x["answer"])),
        }
    )
    data = data.cast_column("answer", Value("string")).filter(
        filter_function, fn_kwargs={"tokenizer": tokenizer}
    )
    return data


def get_questions(name: str, split: str = "train") -> Dataset:
    """Load a dataset by name using the unified system instructions."""

    if name == "math8k":
        return get_math8k_questions(split)
    if name == "gsm8k":
        return get_gsm8k_questions(split)
    if name == "math500":
        return get_math500_questions(split)
    if name == "amc23":
        return get_amc23_questions(split)
    if name == "math":
        return get_math_questions(split)
    raise ValueError(f"Unknown dataset name: {name}")

# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    # Correctness reward, 1.0 for correct, 0.0 for incorrect
    responses = [completion for completion in completions]
    return [1.0 if answer_correct(r, a) else 0.0 for r, a in zip(responses, answer)]


def format_reward_func(completions, answer, **kwargs) -> list[float]:
    # Format reward, max 0.05 when the chain is well structured
    responses = [completion for completion in completions]
    return [0.0 if answer_correct(r, a) else format_score(r) for r, a in zip(responses, answer)]


def length_penalty_func(completion_mask, max_completion_length, **kwargs) -> list[float]:
    # Length penalty, 0.5 for completion length >= max completion length
    completion_lengths = completion_mask.sum(dim=1).tolist()
    return [-0.5 if l >= max_completion_length else 0.0 for l in completion_lengths]


def instruct_structure_score(text: str) -> float:
    """Reward boxed answers and concise step-by-step structure (max 0.1)."""

    boxed_span = _last_boxed_span(text)

    if boxed_span is None:
        return 0.0

    boxed_start, _ = boxed_span
    score = 0.05
    matches = _step_matches_before(text, boxed_start)

    if not matches:
        return score

    numbers = [int(match.group(1)) for match in matches]

    if not _step_sequence(numbers) or len(numbers) > 3:
        return score
    if not _step_segments_have_content(text, matches, boxed_start):
        return score

    step_bonus = len(numbers) / 3.0
    return score + 0.05 * step_bonus


def xmlcount_reward_func(completions, answer, **kwargs) -> list[float]:
    """Legacy name retained: rewards instruct-style structure when incorrect."""

    contents = [completion for completion in completions]
    return [0.0 if answer_correct(r, a) else instruct_structure_score(r) for r, a in zip(contents, answer)]
