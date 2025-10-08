"""
Implementation of data loading and processing functions for math datasets.

"""

import re
from typing import Iterable, NamedTuple
from datasets import Dataset, Value, Features, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .grader import grade_answer

# tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"
def set_tokenizer_name(name):
    global tokenizer_name
    tokenizer_name = name

# SYSTEM_PROMPT = "Solve the following math problem carefully and present your reasoning step by step.\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nAlways insert a blank line (two newline characters) before each Step header.\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$.\n\nWhere [answer] is just the final number or expression that solves the problem."
SYSTEM_PROMPT = \
"""You are a math problem-solving assistant. Your task is to solve the given math problem by providing a clear, step-by-step solution.

You must adhere strictly to the following formatting rules:

1.  **Blank Line Pre-Header:** ALWAYS place a single blank line (`\\n\\n`) immediately before each `## Step` header.
    - **Correct:**
      ...some text.\\n\\n## Step 2
    - **Incorrect:**
      ...some text.\\n## Step 2
      
2.  **Final Answer:** ALWAYS end your entire response with the phrase `Therefore, the final answer is: $\\boxed{answer}`. This must be the absolute last text in your output.

**Example of the required format:**
---
## Step 1
[Explanation and calculations for the first step.]

## Step 2
[Explanation and calculations for the second step.]

... and so on for any additional steps.

Therefore, the final answer is: $\\boxed{The Final Answer}`
---

Now, solve the following problem, strictly adhering to all formatting rules:
"""

_CONCLUSION_PREFIX = re.compile(
    r"Therefore,\s*the\s*final\s*answer\s*is:\s*(?:\$\s*)?\\boxed\{",
    flags=re.IGNORECASE,
)
_CONCLUSION_SUFFIX = re.compile(
    r"\s*\$?\s*\.\s*",
    flags=re.IGNORECASE,
)
_STEP_HEADER_PATTERN = re.compile(
    r"(?:(?:^)|(?:\n\s*\n))\s*(?:[#*]+\s*)?Step[^0-9\r\n]*?(\d+)\b",
    flags=re.IGNORECASE,
)

_STEP_VARIANT_PATTERN = re.compile(
    r"^\s*(?:[#*]+\s*)?(?:step\s*(\d+)|(\d+)[.)\-:])\s*:?(.*)$",
    flags=re.IGNORECASE,
)


class StepInfo(NamedTuple):
    start: int
    end: int
    number: int
    has_blank: bool


def _has_double_newline_before(text: str, pos: int) -> bool:
    if pos <= 0:
        return False
    return bool(re.search(r"\n\s*\n\s*$", text[:pos]))


def _extract_step_infos(text: str, limit: int | None = None) -> list[StepInfo]:
    segment = text if limit is None else text[:limit]
    infos: list[StepInfo] = []

    for match in _STEP_VARIANT_PATTERN.finditer(segment):
        num_str = match.group(1) or match.group(2)
        if not num_str:
            continue
        start = match.start()
        end = match.end()
        has_blank = _has_double_newline_before(segment, start)
        infos.append(StepInfo(start, end, int(num_str), has_blank))

    return infos


def _system_chat_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Compose a chat-style prompt that always uses the math system instructions."""

    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    trimmed = chat_prompt.rstrip()
    return f"{trimmed}\n## Step 1:"


def _text_prompt(question: str) -> str:
    """Return a plain-text prompt containing the system guidance followed by the problem."""

    return (
        f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:\n## Step 1:"
    )


def _build_prompt(tokenizer: AutoTokenizer, question: str, style: str) -> str:
    """Construct the prompt according to ``style`` while reusing the unified system prompt."""

    normalized = style.lower()
    if normalized == "base":
        return _text_prompt(question)
    if normalized == "instruct":
        return _system_chat_prompt(tokenizer, question)
    raise ValueError(f"Unknown style: {style}")


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


def _step_sequence(numbers: list[int], start: int = 1) -> bool:
    """Check that step numbers increase sequentially starting from ``start``."""

    if not numbers:
        return False
    expected = list(range(start, start + len(numbers)))
    return numbers == expected


def _step_segments_have_content(
    text: str,
    infos: list[StepInfo],
    limit: int,
    *,
    require_blank: bool = True,
) -> bool:
    """Ensure each step header is followed by meaningful prose before ``limit``."""

    if not infos:
        return False

    final_limit = len(text) if limit is None else limit

    for idx, info in enumerate(infos):
        if require_blank and not info.has_blank:
            return False

        start = info.end
        end = infos[idx + 1].start if idx + 1 < len(infos) else final_limit

        if start >= end:
            return False

        segment = text[start:end].strip()

        if segment == "":
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
    """Return the format reward (0.1) when the response follows the required structure."""

    boxed_span = _last_boxed_span(text)

    if boxed_span is None:
        return 0.0

    boxed_start, _ = boxed_span
    infos = _extract_step_infos(text, boxed_start)
    blank_infos = [info for info in infos if info.has_blank]

    if not blank_infos:
        return 0.0

    numbers = [info.number for info in blank_infos]

    if not _step_sequence(numbers, start=1):
        return 0.0
    if not _step_segments_have_content(text, blank_infos, boxed_start, require_blank=True):
        return 0.0

    return 0.1


def format_correct(text: str) -> bool:
    """Boolean proxy mirroring ``format_score`` for downstream consumers."""

    return format_score(text) > 0.0


def filter_function(example, tokenizer: AutoTokenizer) -> bool:
    # Filter out examples that are too long
    prompt = example['prompt']
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')
    return tokenized_prompt['input_ids'].shape[1] <= 512


def get_math8k_questions(split: str = "train", style: str = "base") -> Dataset:
    """Load Math8K prompts using the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("parquet", data_files=f"datasets/math8k/{split}.parquet")["train"]
    data = data.map(
        lambda x: {
            "prompt": _build_prompt(tokenizer, x["question"], style),
            "answer": x["gt_answer"],
        },
        load_from_cache_file=False,
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def extract_gsm8k_answer(text: str) -> str | None:
    # Extracts the answer from the GSM8K dataset
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")


def get_gsm8k_questions(split: str = "train", style: str = "base") -> Dataset:
    """Load GSM8K prompts using the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": _build_prompt(tokenizer, x["question"], style),
            "answer": extract_gsm8k_answer(x["answer"]),
        },
        load_from_cache_file=False,
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


def get_math_questions(split: str = "train", style: str = "base") -> Dataset:
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
            "prompt": _build_prompt(tokenizer, x["problem"], style),
            "answer": extract_math_answer(x["solution"]),
        },
        load_from_cache_file=False,
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer}).shuffle(seed=42)
    return data


def get_math500_questions(split: str = "test", style: str = "base") -> Dataset:
    """Load the Math500 dataset prompts with the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("HuggingFaceH4/MATH-500")[split]
    data = data.map(
        lambda x: {
            "prompt": _build_prompt(tokenizer, x["problem"], style),
            "answer": x["answer"],
        },
        load_from_cache_file=False,
    )
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def get_amc23_questions(split: str = "test", style: str = "base") -> Dataset:
    """Load the AMC23 dataset prompts with the unified system instructions."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("zwhe99/amc23")[split]
    data = data.map(
        lambda x: {
            "prompt": _build_prompt(tokenizer, x["question"], style),
            "answer": str(int(x["answer"])),
        },
        load_from_cache_file=False,
    )
    data = data.cast_column("answer", Value("string")).filter(
        filter_function, fn_kwargs={"tokenizer": tokenizer}
    )
    return data


def get_questions(name: str, split: str = "train", style: str = "base") -> Dataset:
    """Load a dataset by name using the unified system instructions."""

    if name == "math8k":
        return get_math8k_questions(split, style)
    if name == "gsm8k":
        return get_gsm8k_questions(split, style)
    if name == "math500":
        return get_math500_questions(split, style)
    if name == "amc23":
        return get_amc23_questions(split, style)
    if name == "math":
        return get_math_questions(split, style)
    raise ValueError(f"Unknown dataset name: {name}")

# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    # Correctness reward, 1.0 for correct, 0.0 for incorrect
    responses = [completion for completion in completions]
    return [1.0 if answer_correct(r, a) else 0.0 for r, a in zip(responses, answer)]


def format_reward_func(completions, answer, **kwargs) -> list[float]:
    # Format reward, 0.1 when the chain follows the full structure
    responses = [completion for completion in completions]
    return [0.0 if answer_correct(r, a) else format_score(r) for r, a in zip(responses, answer)]


def length_penalty_func(completion_mask, max_completion_length, **kwargs) -> list[float]:
    # Length penalty, 0.5 for completion length >= max completion length
    completion_lengths = completion_mask.sum(dim=1).tolist()
    return [-0.5 if l >= max_completion_length else 0.0 for l in completion_lengths]


def instruct_structure_score(text: str) -> float:
    """Reward structured steps (0.0125 each, up to three) plus boxed answer bonus."""

    boxed_span = _last_boxed_span(text)
    limit = boxed_span[0] if boxed_span is not None else len(text)
    infos = _extract_step_infos(text, limit)

    reward = 0.0

    if infos:
        numbers = [info.number for info in infos]

        if _step_sequence(numbers, start=1) and _step_segments_have_content(text, infos, limit, require_blank=False):
            for info in infos[:3]:
                reward += 0.025 if info.has_blank else 0.0125

    box_count = text.count("\\boxed{")

    if box_count >= 1:
        reward += 0.0125
        if box_count > 1:
            reward -= 0.00625 * (box_count - 1)

    return reward


def xmlcount_reward_func(completions, answer, **kwargs) -> list[float]:
    """Legacy name retained: rewards instruct-style structure when incorrect."""

    contents = [completion for completion in completions]
    return [0.0 if answer_correct(r, a) else instruct_structure_score(r) for r, a in zip(contents, answer)]
