import re
import warnings
import soundfile as sf
import torch
import transformers

from pathlib import Path

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)

HF_MODELS = {
    "Qwen3-Omni-30B-A3B-Instruct": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
    },
    "Phi-4-multimodal-instruct": {
        "id": "microsoft/Phi-4-multimodal-instruct",
        "revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
    },
    "Kimi-Audio-7B-Instruct": {
        "id": "moonshotai/Kimi-Audio-7B-Instruct",
        "revision": "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b",
    },
}

# ---------------------------------------------------------------------
# Optional model classes: availability depends on transformers version
# ---------------------------------------------------------------------

try:
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
except Exception:
    Qwen3OmniMoeForConditionalGeneration = None
    Qwen3OmniMoeProcessor = None

QWEN3_OMNI_UTILS_IMPORT_ERROR = None

try:
    from qwen_omni_utils import (
        process_mm_info as qwen3_omni_process_mm_info
    )
except Exception as e:
    qwen3_omni_process_mm_info = None
    QWEN3_OMNI_UTILS_IMPORT_ERROR = e

# Kimi-Audio runs in full precision and must not require bitsandbytes merely
# to import this module.  Keep quantization optional for the models that use it.
BNB_CONFIG_ERROR = None
try:
    bnb_config_4bits = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    bnb_config_8bits = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
except Exception as exc:
    bnb_config_4bits = None
    bnb_config_8bits = None
    BNB_CONFIG_ERROR = exc

def _model_device(model):
    
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _move_batch(batch, model):

    dev = _model_device(model)

    if hasattr(batch, "to"):
        batch = batch.to(dev)
    else:
        batch = {
            k: (v.to(dev) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    return batch

def _clean_structured_answer(text):
    
    if not text:
        return text

    # Remove obvious role continuations if early stopping did not catch them.
    text = re.split(
        r"\n\s*(?:Human|User|Assistant|System)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    answer_match = re.search(
        r"Answer\s*:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    reasoning_match = re.search(
        r"Reasoning\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not answer_match:
        return text.strip()

    answer = answer_match.group(1).strip()

    if not reasoning_match:
        return f"Answer: {answer}"

    reasoning = reasoning_match.group(1).strip()

    # Keep only the first complete reasoning sentence. The experimental prompt
    # explicitly asks for a brief explanation.
    sentence = re.match(
        r"(.+?[.!?。！？])(?:\s|$)",
        reasoning,
        flags=re.DOTALL,
    )
    if sentence:
        reasoning = sentence.group(1).strip()

    return (
        f"Answer: {answer}\n"
        f"Reasoning: {reasoning}"
    )



class LALMModel:

    _model_cache = {}

    def __init__(
        self,
        model_name,
        quantization_config=None,
        do_sample=False,
        device_map="auto",
    ):
        if model_name not in HF_MODELS:
            raise ValueError(
                f"Unsupported LALM model: {model_name}\n"
                f"Available models: {sorted(HF_MODELS)}"
            )

        self.model_name = model_name
        self.model_id = HF_MODELS[model_name]["id"]
        self.model_revision = HF_MODELS[model_name]["revision"]

        self.do_sample = do_sample
        self.device_map = device_map

        if quantization_config == "4bits":
            if bnb_config_4bits is None:
                raise ImportError(
                    "4-bit quantization requires bitsandbytes. "
                    "Install it in this environment or use --quantization_config none."
                ) from BNB_CONFIG_ERROR
            self.quantization_config = bnb_config_4bits
        elif quantization_config == "8bits":
            if bnb_config_8bits is None:
                raise ImportError(
                    "8-bit quantization requires bitsandbytes. "
                    "Install it in this environment or use --quantization_config none."
                ) from BNB_CONFIG_ERROR
            self.quantization_config = bnb_config_8bits
        elif quantization_config in (None, "none", "None"):
            self.quantization_config = None
        else:
            raise ValueError(
                "quantization_config must be None, '4bits', or '8bits'."
            )

        self.prompt = ""
        self.audio_path = None
        self.messages = None
        self.response = ""
        self.token_probabilities = None

        self.family = None

        self.tokenizer = None

        cache_key = (
            self.model_name,
            quantization_config,
            self.device_map,
        )

        if cache_key in LALMModel._model_cache:
            cached = LALMModel._model_cache[cache_key]
            self.model = cached["model"]
            self.processor = cached["processor"]
            self.tokenizer = cached.get("tokenizer")
            self.family = cached["family"]
        else:
            self._load_model_and_processor()
            LALMModel._model_cache[cache_key] = {
                "model": self.model,
                "processor": self.processor,
                "tokenizer": self.tokenizer,
                "family": self.family,
            }

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _base_load_kwargs(self):
        kwargs = {
            "device_map": self.device_map,
        }

        if torch.cuda.is_available():
            # Transformers 5 renamed this keyword. Keep the Kimi and Qwen
            # environments compatible across their supported releases.
            dtype_kwarg = (
                "dtype"
                if int(transformers.__version__.split(".", 1)[0]) >= 5
                else "torch_dtype"
            )
            kwargs[dtype_kwarg] = torch.bfloat16

        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        return kwargs

    def _load_model_and_processor(self):

        # -------------------------------------------------------------
        # Microsoft Phi-4 multimodal
        # -------------------------------------------------------------
        if self.model_name == "Phi-4-multimodal-instruct":
            # Phi-4's remote code is compatible with the Transformers 4.x
            # loading path used in its official release, but not the meta
            # tensor initialization introduced by Transformers 5.
            version_parts = transformers.__version__.split(".")
            version_tuple = tuple(
                int(part) if part.isdigit() else 0
                for part in version_parts[:2]
            )
            if not ((4, 48) <= version_tuple < (5, 0)):
                raise ImportError(
                    "Phi-4-multimodal-instruct must run in the 'phi_kimi' environment "
                    "with transformers==4.48.2. It is incompatible with the "
                    f"installed version ({transformers.__version__})."
                )

            # Phi-4 uses its own audio placeholder and processor API.  Its
            # official loader is AutoModelForCausalLM with remote code.
            kwargs = self._base_load_kwargs()
            kwargs["trust_remote_code"] = True
            kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"

            # Phi defaults to FlashAttention.  Keep that fast path when it is
            # installed, but make the model runnable in the shared environment
            # without imposing flash-attn as a dependency.
            try:
                import flash_attn  # noqa: F401
                kwargs["_attn_implementation"] = "flash_attention_2"
            except Exception:
                kwargs["_attn_implementation"] = "eager"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                **kwargs,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                trust_remote_code=True,
            )
            self.family = "phi4_multimodal"
            return

        # -------------------------------------------------------------
        # Moonshot AI Kimi-Audio
        # -------------------------------------------------------------
        if self.model_name == "Kimi-Audio-7B-Instruct":

            try:
                from kimia_infer.api.kimia import KimiAudio
            except ImportError as exc:
                raise ImportError(
                    "Kimi-Audio requires its official inference package. "
                    "Install it in the server environment with:\n"
                    "  python -m pip install git+https://github.com/MoonshotAI/Kimi-Audio.git"
                ) from exc

            if self.quantization_config is not None:
                warnings.warn(
                    "Kimi-Audio uses its own official loader; "
                    "quantization_config is ignored. Run this model with "
                    "--quantization_config none."
                )

            # The bias benchmark only needs audio -> text.  Avoid loading the
            # speech detokenizer, which Kimi uses only for audio generation.
            self.model = KimiAudio(
                model_path=self.model_id,
                load_detokenizer=False,
            )
            self.processor = None
            self.family = "kimi_audio"
            return

        if self.model_name == "Qwen3-Omni-30B-A3B-Instruct":
            if (
                Qwen3OmniMoeForConditionalGeneration is None
                or Qwen3OmniMoeProcessor is None
            ):
                raise ImportError(
                    "Qwen3-Omni requires a recent transformers version with "
                    "Qwen3OmniMoeForConditionalGeneration and "
                    "Qwen3OmniMoeProcessor. Qwen recommends transformers >= 5.2.0."
                )

            if qwen3_omni_process_mm_info is None:
                raise ImportError(
                    "Could not import qwen_omni_utils.process_mm_info.\n"
                    "qwen-omni-utils may be installed, but an internal "
                    "dependency or version may be incompatible.\n\n"
                    f"Underlying error:\n"
                    f"  {type(QWEN3_OMNI_UTILS_IMPORT_ERROR).__name__}: "
                    f"{QWEN3_OMNI_UTILS_IMPORT_ERROR}\n\n"
                    "Check with:\n"
                    "  python -c \"from qwen_omni_utils import process_mm_info; "
                    "print('OK')\""
                )

            kwargs = {
                "device_map": self.device_map,
                "dtype": "auto",
            }

            # FlashAttention 2 is recommended by Qwen for lower VRAM usage,
            # but keep a safe fallback when flash_attn is not installed.
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass

            if self.quantization_config is not None:
                kwargs["quantization_config"] = self.quantization_config

            self.model = (
                Qwen3OmniMoeForConditionalGeneration
                .from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    **kwargs,
                )
                .eval()
            )

            # We only need audio -> textual answer for the bias benchmark.
            # Qwen documents that disabling Talker saves about 10 GB VRAM.
            if hasattr(self.model, "disable_talker"):
                self.model.disable_talker()

            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_id,
                revision=self.model_revision,
            )

            self.family = "qwen3_omni"
            return

        raise ValueError(f"No loader was found for {self.model_name}")

    # -----------------------------------------------------------------
    # Prompt / audio setup
    # -----------------------------------------------------------------
    def _set_qwen3_omni_messages(self):
        if self.audio_path is None:
            raise ValueError("Qwen3-Omni requires an audio file.")

        audio_content = {"type": "audio", "audio": self.audio_path}
        text_content = {
            "type": "text",
            "text": self.prompt,
        }
        content = [audio_content, text_content]
        self.messages = [{"role": "user", "content": content}]

    def _set_phi4_audio_messages(self):
        audio_markers = "<|audio_1|>"
        user_content = f"{audio_markers}{self.prompt}"
        self.messages = (
            f"<|user|>{user_content}"
            "<|end|><|assistant|>"
        )

    def _set_kimi_audio_messages(self):
        if self.audio_path is None:
            raise ValueError("Kimi-Audio requires an audio file.")

        audio_message = {
            "role": "user",
            "message_type": "audio",
            "content": self.audio_path,
        }
        text_message = {
            "role": "user",
            "message_type": "text",
            "content": self.prompt,
        }
        self.messages = [audio_message, text_message]

    def set_prompt(
        self,
        prompt,
        audio_path=None,
    ):
        
        self.prompt = prompt
        self.audio_path = audio_path
        self.messages = None
        self.response = ""

        if audio_path is None:
            raise ValueError("audio_path is required.")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        # Prepare message structures for the supported chat-template families.
        if self.family == "phi4_multimodal":
            # Phi-4 does not use a generic chat template for speech: the
            # numbered <|audio_N|> placeholders are part of its documented
            # prompt format.
            self._set_phi4_audio_messages()

        elif self.family == "kimi_audio":
            self._set_kimi_audio_messages()

        elif self.family == "qwen3_omni":
            self._set_qwen3_omni_messages()

    # -----------------------------------------------------------------
    # Common generation helpers
    # -----------------------------------------------------------------

    def _capture_token_probabilities(self, output):

        sequences = getattr(output, "sequences", None)
        logits = getattr(output, "logits", None)
        if sequences is None or logits is None:
            raise RuntimeError(
                "The model did not return logits; "
                "return_dict_in_generate=True y output_logits=True."
            )

        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)

        self.token_probabilities = []
        if not logits:
            return sequences

        generated = sequences[0, -len(logits):]
        if len(generated) != len(logits):
            raise RuntimeError(
                "Generated tokens do not match the logits."
            )

        for token_id, step_logits in zip(generated.tolist(), logits):
            log_probability = torch.log_softmax(
                step_logits[0].float(), dim=-1
            )[token_id]
            self.token_probabilities.append(
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ),
                    "probability": float(log_probability.exp().item()),
                }
            )
        return sequences

    def get_token_probabilities(self):

        return self.token_probabilities

    def _generation_kwargs(self, max_new_tokens):

        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "return_dict_in_generate": True,
            "output_logits": True,
        }

        return kwargs

    # -----------------------------------------------------------------
    # Family-specific inference
    # -----------------------------------------------------------------

    def _run_phi4_multimodal(self, max_new_tokens):
        # Do not resample, normalize, trim, or otherwise alter the WAV.
        # Phi's processor receives both the original waveform and its
        # original sampling rate, exactly as in Microsoft's example.
        audio, sampling_rate = sf.read(self.audio_path)
        if audio.size == 0:
            raise ValueError(
                f"Phi-4 audio file is empty: {self.audio_path}"
            )

        inputs = self.processor(
            text=self.messages,
            audios=[(audio, sampling_rate)],
            return_tensors="pt",
        )

        # Phi's reference call moves tensors to CUDA without casting the raw
        # audio.  Preserve that behaviour instead of reusing the BF16 audio
        # casting used by other model families.
        inputs = _move_batch(inputs, self.model)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens),
            )

        outputs = self._capture_token_probabilities(outputs)
        self.response = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _run_kimi_audio(self, max_new_tokens):
        if self.messages is None:
            raise RuntimeError(
                "No messages are defined for Kimi-Audio. "
                "Call set_prompt() first."
            )

        # The benchmark consumes only text.  The official API returns
        # (waveform_or_none, text) for output_type="text".
        _, text = self.model.generate(
            self.messages,
            # Kimi samples its audio branch even for output_type="text", then
            # immediately discards that token.  Greedy audio sampling avoids
            # a needless multinomial call on this unused branch.
            audio_temperature=0.0,
            text_temperature=0.0,
            output_type="text",
            max_new_tokens=max_new_tokens,
        )
        self.response = "" if text is None else str(text).strip()


    def _run_qwen3_omni(self, max_new_tokens):

        if self.messages is None:
            raise RuntimeError(
                "No messages are defined for Qwen3-Omni. "
                "Call set_prompt() first."
            )

        use_audio_in_video = False

        prompt_text = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos = qwen3_omni_process_mm_info(
            self.messages,
            use_audio_in_video=use_audio_in_video,
        )

        processor_kwargs = {
            "text": prompt_text,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_audio_in_video,
        }

        if audios is not None:
            processor_kwargs["audio"] = audios

        if images is not None:
            processor_kwargs["images"] = images

        if videos is not None:
            processor_kwargs["videos"] = videos

        inputs = self.processor(
            **processor_kwargs
        )

        # Move all tensors to the Thinker/model input device. Cast only floating
        # multimodal tensors, never input_ids/attention masks.
        dev = _model_device(self.model)
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue

            value = value.to(dev)

            if torch.is_floating_point(value):
                value = value.to(model_dtype)

            inputs[key] = value

        if "input_ids" not in inputs:
            raise RuntimeError(
                "Qwen3-Omni processor did not return input_ids."
            )

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "thinker_output_logits": True,
            "thinker_max_new_tokens": max_new_tokens,
            "thinker_do_sample": self.do_sample,
            "use_audio_in_video": use_audio_in_video,
        }

        if not self.do_sample:
            # Explicit deterministic decoding for reproducible bias evaluation.
            generate_kwargs["thinker_temperature"] = 0

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        # With return_audio=False, current Qwen3-Omni returns either:
        #   (text_output, None)
        # or directly text_output depending on transformers release.
        if isinstance(output, tuple):
            text_output = output[0]
        else:
            text_output = output

        self._capture_token_probabilities(text_output)

        # When thinker_return_dict_in_generate=True, text output has .sequences.
        if hasattr(text_output, "sequences"):
            sequences = text_output.sequences
        else:
            sequences = text_output

        if not torch.is_tensor(sequences):
            raise RuntimeError(
                "Formato de salida inesperado de Qwen3-Omni: "
                f"{type(sequences)}"
            )

        if (
            sequences.ndim == 2
            and sequences.shape[1] >= input_len
            and torch.equal(
                sequences[:, :input_len],
                inputs["input_ids"],
            )
        ):
            generated_ids = sequences[:, input_len:]
        else:
            generated_ids = sequences

        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        self.response = _clean_structured_answer(
            decoded
        )

    # -----------------------------------------------------------------
    # Public inference entry point
    # -----------------------------------------------------------------

    def run_inference(self, max_new_tokens):
        
        self.token_probabilities = None
        if not self.prompt:
            raise RuntimeError(
                "No prompt is defined. Call set_prompt() first."
            )

        if self.audio_path is None:
            raise RuntimeError(
                "No audio is defined. Call set_prompt(..., audio_path=...) first."
            )

        if self.family == "kimi_audio":
            return self._run_kimi_audio(max_new_tokens)

        if self.family == "qwen3_omni":
            return self._run_qwen3_omni(max_new_tokens)

        if self.family == "phi4_multimodal":
            return self._run_phi4_multimodal(max_new_tokens)

        raise RuntimeError(
            f"Unrecognized LALM configuration: family={self.family}"
        )

    # -----------------------------------------------------------------
    # Output cleanup
    # -----------------------------------------------------------------

    @staticmethod
    def _remove_thinking_content(text):

        if not isinstance(text, str) or not text:
            return text

        # Remove complete <think>...</think> blocks.
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Also remove an unterminated <think> block, if a model ever emits one.
        cleaned = re.sub(
            r"<think>.*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    def get_response(self):
        return self._remove_thinking_content(self.response)
