import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional

def encode_batch(tokenizer: Any, texts: List[str]) -> List[List[int]]:
    tok = getattr(tokenizer, "tokenizer", None) or tokenizer
    enc = tok(texts, add_special_tokens=False, padding=False, return_attention_mask=False)  # type: ignore[misc]
    ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
    return ids

class StepGeneration:
    def __init__(
        self,
        max_steps: int,
        *,
        step_token: Optional[str] = None,
        tokens_per_step: Optional[int] = None,
        stop_token: Optional[str] = None,
        include_stop_str_in_output: bool = False,
    ) -> None:
        # exactly one of step_token or tokens_per_step
        if (step_token is None) == (tokens_per_step is None):
            raise ValueError("Provide exactly one of step_token or tokens_per_step")
        if tokens_per_step is not None and tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be a positive integer")

        self.max_steps = int(max_steps)
        self.step_token = step_token
        self.tokens_per_step = tokens_per_step
        self.stop_token = stop_token
        self.include_stop_str_in_output = include_stop_str_in_output


# --- Confidence-based SMC with vLLM ---

@dataclass(slots=True)
class SMCParticle:
    is_stopped: bool
    win_vals: List[float]
    win_sum: float
    total_new: int
    ctx_chunks: List[str]


class SMCVLLM:
    """Self-confidence SMC in vLLM.
    - Input prompts_text length is G*N (each unique prompt repeated N times). We collapse to G groups.
    - Step segmentation with step_token; resample at each step boundary.
    - Confidence from top-k logprobs per generated token.
    - Resample only from alive particles (finished excluded) and replenish back to N.
    - Terminate on stop_token string in step text or when total_new_tokens >= max_new_tokens.
    - Return N sequences per group for training; optional return_all for eval.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tokenizer: Any,
        num_particles: int,
        pad_token_id: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        sg: StepGeneration,
        smc_topk: int,
        window_size: int,
        confidence_eta: float,
        max_new_tokens: int,
        scoring: str = "entropy",
        return_all: bool = False,
    ) -> None:
        self.llm = llm
        self.tok = tokenizer
        self.N = int(num_particles)
        self.pad = int(pad_token_id)
        self.temp = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.rep = float(repetition_penalty)
        self.sg = sg
        self.smc_topk = int(smc_topk)
        self.window_size = int(window_size)
        self.confidence_eta = float(confidence_eta)
        self.max_new_tokens = int(max_new_tokens)
        self.scoring = scoring
        self.return_all = bool(return_all)
        # Ensure vLLM engine allows at least smc_topk logprobs (no heavy error handling)
        engine = getattr(self.llm, "llm_engine", None)
        mc = getattr(getattr(engine, "vllm_config", None), "model_config", None) or getattr(engine, "model_config", None)
        if mc is not None and self.smc_topk > 0:
            curr = getattr(mc, "max_logprobs", 0) #TODO: for cumulative_logprob we don't need this
            if curr != -1 and curr < self.smc_topk:
                mc.max_logprobs = int(self.smc_topk)

        # Prebuild sampling params
        stop_list = [self.sg.step_token] if self.sg.step_token is not None else None
        self._sp = self._sampling_params(stop=stop_list)

    def _sampling_params(self, *, stop: Optional[List[str]]) -> Any:
        from vllm import SamplingParams
        kwargs = dict(
            n=1,
            temperature=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.rep,
            stop=stop,
            detokenize=True,
            logprobs=self.smc_topk if self.smc_topk > 0 else None,
            max_tokens = self.max_new_tokens
        )
        return SamplingParams(**kwargs)

    def _entropy_conf_from_logprobs(self, logprob_entry) -> float:
        """Compute exp(-H) over top-k candidates for one generated token."""
        vals: List[float] = []
        if logprob_entry is None:
            return 0.0
        if isinstance(logprob_entry, dict):
            vals = [float(getattr(v, "logprob", v)) for v in logprob_entry.values() if getattr(v, "logprob", v) is not None]  # type: ignore[arg-type]
            vals.sort(reverse=True)
        else:
            vals = [float(getattr(x, "logprob")) for x in logprob_entry if getattr(x, "logprob", None) is not None]
            vals.sort(reverse=True)
        if len(vals) == 0:
            return 0.0
        if self.smc_topk > 0:
            vals = vals[: self.smc_topk]
        # softmax over logprobs (equivalent to normalize exp(lp))
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        if s <= 0.0:
            return 0.0
        probs = [e / s for e in exps]
        # entropy H = -sum p log p
        H = 0.0
        for p in probs:
            if p > 0:
                H -= p * math.log(p + 1e-20)
        return math.exp(-H)

    def generate(
        self,
        prompts_text: List[str],
        original_prompts: List[Any],
    ) -> List[List[int]]:
        total = len(prompts_text)
        if total % self.N != 0:
            raise ValueError(f"SMC expects G*N items (N={self.N}). Got total={total}.")
        G = total // self.N
        headers = [prompts_text[g * self.N] for g in range(G)]

        # Particles and finished pools per group
        particles: List[List[SMCParticle]] = []
        for g in range(G):
            group = []
            for _ in range(self.N):
                group.append(
                    SMCParticle(
                        is_stopped=False,
                        win_vals=[],
                        win_sum=0.0,
                        total_new=0,
                        ctx_chunks=[headers[g]],
                    )
                )
            particles.append(group)

        # Step loop
        for _step in range(self.sg.max_steps):
            # Build contexts for alive particles (string prompts)
            batch_inputs: List[str] = []
            idx_map: List[tuple[int, int]] = []
            for g in range(G):
                for j in range(self.N):
                    p = particles[g][j]
                    if p.is_stopped:
                        continue
                    # Build prompt from chunks; append step token only for this call
                    prompt_text = "".join(p.ctx_chunks)
                    if isinstance(self.sg.step_token, str) and self.sg.step_token and not prompt_text.endswith(self.sg.step_token):
                        prompt_text = prompt_text + self.sg.step_token
                    batch_inputs.append(prompt_text)
                    idx_map.append((g, j))

            if not batch_inputs:
                break

            outs = self.llm.generate(batch_inputs, sampling_params=self._sp, use_tqdm=False)

            # Update particles with outputs
            for k, out in enumerate(outs):
                g, j = idx_map[k]
                p = particles[g][j]
                if not out.outputs:
                    p.is_stopped = True
                    continue
                o = out.outputs[0]
                step_ids = getattr(o, "token_ids", None) or [] #TODO: 비효율적?
                p.total_new += len(step_ids)
                # Update context text
                step_text = getattr(o, "text", "")
                if step_text:
                    p.ctx_chunks.append(step_text)

                # Update confidence per token using top-k logprobs for each generated position
                if self.smc_topk != 0:  # if 0, skip logprob processing
                    lp_list = getattr(o, "logprobs", None) or []
                    for lp_entry in lp_list:
                        conf = self._entropy_conf_from_logprobs(lp_entry) if self.scoring == "entropy" else 0.0
                        # window maintenance
                        p.win_vals.append(conf)
                        p.win_sum += conf
                        if len(p.win_vals) > self.window_size:
                            p.win_sum -= p.win_vals.pop(0)

                # Stop conditions: EOS via vLLM (stop_reason is None), or stop_token substring, or max_new_tokens
                finished = (getattr(o, "stop_reason", None) is None)
                if isinstance(self.sg.stop_token, str) and self.sg.stop_token:
                    if self.sg.stop_token in step_text:
                        finished = True
                if self.max_new_tokens and p.total_new >= self.max_new_tokens: #TODO: vllm 자체가 max_token 만족하도록 해야함.
                    finished = True
                if finished:
                    p.is_stopped = True

            # Resample per group among alive only, replenish to N #TODO: replenish 하는건 좀..
            for g in range(G):
                alive_indices = [j for j in range(self.N) if not particles[g][j].is_stopped]
                if len(alive_indices) == 0:
                    continue
                # weights from windowed confidence average ^ eta
                ws: List[float] = []
                for j in alive_indices:
                    denom = max(1, len(particles[g][j].win_vals))
                    avg = max(particles[g][j].win_sum / denom, 0.0)
                    ws.append(avg ** self.confidence_eta)
                s = sum(ws)
                if s <= 0:
                    probs = [1.0 / len(alive_indices) for _ in alive_indices]
                else:
                    probs = [w / s for w in ws]
                # sample N parents among alive
                parents = random.choices(alive_indices, weights=probs, k=self.N)
                new_group: List[SMCParticle] = []
                for j_parent in parents:
                    src = particles[g][j_parent]
                    # Shallow copy is fine; but we need independent win_vals list
                    new_group.append(
                        SMCParticle(
                            is_stopped=src.is_stopped,
                            win_vals=list(src.win_vals),
                            win_sum=float(src.win_sum),
                            total_new=int(src.total_new),
                            ctx_chunks=list(src.ctx_chunks),
                        )
                    )
                particles[g] = new_group
            # Exit if all groups done
            if all(p.is_stopped for grp in particles for p in grp):
                break

        # Build results as completion token IDs encoded by processing_class
        completions_text: List[str] = []
        for g in range(G):
            for j in range(self.N):
                completions_text.append("".join(particles[g][j].ctx_chunks[1:]))
        return encode_batch(self.tok, completions_text)
