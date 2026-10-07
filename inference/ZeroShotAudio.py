from ALMBase import LALMModel

import time

GENERAL_LALM_MODELS = {
    "Qwen3-Omni-30B-A3B-Instruct",
    "Kimi-Audio-7B-Instruct",
    "Phi-4-multimodal-instruct",
}

class ZeroShotAudio:
    def __init__(
        self,
        model_name,
        quantization_config = "none",
    ):
        if model_name not in GENERAL_LALM_MODELS:
            raise ValueError(
                f"Unsupported model for this audio-question task: "
                f"{model_name}\n\n"
                "Supported models:\n"
                + "\n".join(sorted(GENERAL_LALM_MODELS))
            )

        if quantization_config not in {
            "4bits",
            "8bits",
            "none",
        }:
            raise ValueError(
                "quantization_config must be "
                "4bits, 8bits, or none"
            )

        quant = (
            None
            if quantization_config == "none"
            else quantization_config
        )

        self.model_name = model_name

        self.lalm = LALMModel(
            model_name=model_name,
            quantization_config=quant,
            do_sample=False,
        )

    def ask(self, prompt, audio_path, max_new_tokens):

        total_start = time.time()

        self.lalm.set_prompt(
            prompt=prompt,
            audio_path=audio_path,
        )

        inference_start = time.time()

        self.lalm.run_inference(
            max_new_tokens=max_new_tokens
        )

        inference_time = time.time() - inference_start

        response = self.lalm.get_response()

        response = (
            ""
            if response is None
            else str(response).strip()
        )

        total_time = time.time() - total_start

        return {
            "response": response,
            "token_probabilities": self.lalm.get_token_probabilities(),
            "inference_time_sec": inference_time,
            "total_time_sec": total_time,
        }
