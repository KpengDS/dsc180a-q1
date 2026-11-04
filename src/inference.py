"""Shared Qwen runtime for smoke tests and sensitivity experiments."""
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import time

from .ag_news import LABELS, SYSTEM, TEMPLATE, build_prompt, parse_label
from .base_protocol import POLICY, build_base_prompt, completion_line, parse_base_label

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "fancyzhx/ag_news"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"


def setup_environment():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(ROOT / "data/huggingface"))


def load_data(revision):
    setup_environment()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(DATASET, revision=revision).sha
    return load_dataset(DATASET, revision=sha, cache_dir=str(ROOT / "data/datasets")), sha


class QwenClassifier:
    def __init__(self, seed, device="auto", revision=MODEL_REVISION, model_id=MODEL, prompt_mode="chat"):
        if prompt_mode not in ("chat", "completion"):
            raise ValueError("prompt_mode must be chat or completion")
        self.model_id, self.prompt_mode = model_id, prompt_mode
        setup_environment()
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        mps = torch.backends.mps.is_available()
        self.device = ("mps" if mps else "cpu") if device == "auto" else device
        if self.device == "mps" and not mps:
            raise RuntimeError("MPS unavailable; check outside the sandbox or use --device cpu")
        self.dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.revision = HfApi().model_info(self.model_id, revision=revision).sha
        cache = str(ROOT / "models")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, cache_dir=cache)
        print(f"Loading {self.model_id}@{self.revision} on {self.device} ({self.dtype})", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=cache, torch_dtype=self.dtype,
            attn_implementation="eager", use_safetensors=True,
        ).to(self.device).eval()
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=12,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

    def predict(self, demonstrations, article, max_chars=1000):
        import torch
        if self.prompt_mode == "completion":
            rendered = build_base_prompt(demonstrations, article, max_chars=max_chars)
        else:
            prompt = build_prompt(demonstrations, article, max_chars=max_chars)
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        if inputs.input_ids.shape[1] + self.generation.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds model context; reduce --max-chars or shots")
        generation_kwargs = {}
        if self.prompt_mode == "completion":
            from transformers import StoppingCriteria, StoppingCriteriaList
            tokenizer, prompt_length = self.tokenizer, inputs.input_ids.shape[1]

            class StopAtNewline(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
                    return "\n" in text

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopAtNewline()])
        if self.device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=self.generation, **generation_kwargs)
        raw = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if self.device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        extra = {}
        parse_input = raw
        if self.prompt_mode == "completion":
            new_ids = outputs[0, inputs.input_ids.shape[1]:].tolist()
            eos = self.generation.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            reason = "newline" if "\n" in raw else "eos" if new_ids and new_ids[-1] in eos_ids else "token_cap"
            parse_input = completion_line(raw)
            extra = dict(completion_text=parse_input, termination_reason=reason, generated_token_ids=new_ids)
        error = None
        try:
            if self.prompt_mode == "completion":
                if extra["termination_reason"] == "token_cap":
                    raise ValueError("Base completion reached token cap without newline or EOS")
                predicted = parse_base_label(parse_input)
            else:
                predicted = parse_label(raw)
        except ValueError as exc:
            predicted, error = None, str(exc)
        return dict(predicted_label=predicted,
                    predicted_name=LABELS[predicted] if predicted is not None else None,
                    raw_output=raw, parse_error=error, rendered_prompt=rendered,
                    input_tokens=inputs.input_ids.shape[1], seconds=elapsed, **extra)

    def metadata(self):
        return dict(model=self.model_id, model_revision=self.revision, labels=LABELS,
                    prompt_template=TEMPLATE, system_prompt=SYSTEM if self.prompt_mode == "chat" else None,
                    prompt_mode=self.prompt_mode, parser_policy=POLICY if self.prompt_mode == "completion" else "existing Instruct parser",
                    chat_template=self.tokenizer.chat_template if self.prompt_mode == "chat" else None, generation=self.generation.to_dict(),
                    device=self.device, dtype=str(self.dtype), python=platform.python_version(),
                    platform=platform.platform(), deterministic_algorithms=True,
                    versions={p: importlib.metadata.version(p) for p in
                              ("torch", "transformers", "datasets", "huggingface-hub")})
