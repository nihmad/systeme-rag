
from __future__ import annotations

from .config import CONFIG

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la documentation technique du "
    "développement web. Réponds en français, de façon concise et exacte."
)


class Generator:
    def __init__(self, model_name: str | None = None, load_in_4bit: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or CONFIG.model.generator_model
        load_in_4bit = CONFIG.model.load_in_4bit if load_in_4bit is None else load_in_4bit

        print(f"[generator] Chargement du LLM : {self.model_name} (4bit={load_in_4bit})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        kwargs: dict = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _generate(self, messages: list[dict]) -> str:
        import torch

        chat_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=CONFIG.model.max_new_tokens,
                do_sample=CONFIG.model.temperature > 0,
                temperature=max(CONFIG.model.temperature, 1e-4),
                top_p=0.9,
                repetition_penalty=1.05,
                # 32000 = token de fin de tour de CroissantLLMChat
                eos_token_id=([self.tokenizer.eos_token_id, 32000]
                              if "croissant" in self.model_name.lower()
                              else self.tokenizer.eos_token_id),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = tokens[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    # deux modes : avec contexte (RAG) ou sans (closed-book)
    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        ctx = "\n\n".join(f"[Extrait {i+1}] {c}" for i, c in enumerate(contexts))
        user = (
            f"Contexte issu de la documentation :\n{ctx}\n\n"
            f"Question : {question}\n\n"
            "En t'appuyant uniquement sur le contexte ci-dessus, réponds à la "
            "question. Si le contexte ne contient pas la réponse, dis-le."
        )
        return self._generate([
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user},
        ])

    def answer_closed_book(self, question: str) -> str:
        return self._generate([
            {"role": "user", "content": SYSTEM_PROMPT + "\n\nQuestion : " + question},
        ])
