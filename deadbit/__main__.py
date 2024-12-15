import torch

from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from outlines import models, generate
from pydantic import BaseModel
from transformers import T5EncoderModel

LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
WEB_COMIC_SCRIPT_PROMPT = """
Write a description of less than 70 words or less for a funny picture with 4 panels.
It is about a software developer, named deadbit, and the daily troubles he faces.
"""


class WebComic(BaseModel):
    title: str
    content: str


def create_web_comic_script(prompt: str) -> WebComic:
    model = models.transformers(LLM_MODEL_ID)
    generator = generate.json(model, WebComic)
    result = generator(prompt)
    return result


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
    script = create_web_comic_script(WEB_COMIC_SCRIPT_PROMPT)
    print(script)
    image_pipeline = create_image_pipeline()
    image_prompt = create_image_prompt(script.content)
    create_image(image_pipeline, image_prompt, f"webcomic-{script.title}.png")
    with open(f"webcomic-{script.title}.txt", "w") as f:
        f.write(script.content)
