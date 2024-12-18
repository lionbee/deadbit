import torch

from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel

LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
LLM_CONTEXT = "You are a script writer"
WEB_COMIC_SCRIPT_PROMPT = """
Write 4 short descriptions for the panels of a 4 panel funny comic.
It is about a software developer named deadbit.
The comic always has a punchline in the last panel.

Only give me the output for the 4 panels and format it as follows:
Panel 1: [description]
Panel 2: [description]
Panel 3: [description]
Panel 4: [description]
"""


def write_script():
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    messages = [
        {"role": "system", "content": LLM_CONTEXT},
        {"role": "user", "content": WEB_COMIC_SCRIPT_PROMPT},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def create_image_prompt(response: str) -> str:
    image_context = "4 panel comic styled as 8bit game."
    image_prompt = f"{image_context} {response}"
    return image_prompt


def create_image_pipeline() -> StableDiffusion3Pipeline:
    model_id = DIFFUSION_MODEL_ID

    nf4_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    t5_nf4 = T5EncoderModel.from_pretrained(
        "diffusers/t5-nf4", torch_dtype=torch.bfloat16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16,
    )

    pipeline.enable_model_cpu_offload()
    return pipeline


def create_image(
    pipeline: StableDiffusion3Pipeline, image_prompt: str, image_path: str
) -> None:
    image = pipeline(
        prompt=image_prompt,
        prompt_3=image_prompt,
        num_inference_steps=20,
        guidance_scale=10,
    ).images[0]
    image.save(image_path)


if __name__ == "__main__":
    script = write_script()
    image_prompt = create_image_prompt(script)
    image_pipeline = create_image_pipeline()
    create_image(image_pipeline, image_prompt, f"webcomic.png")
    with open(f"webcomic.txt", "w") as f:
        f.write(script)
