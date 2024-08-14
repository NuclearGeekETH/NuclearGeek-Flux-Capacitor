# First run these commands if using Windows and installing manually:
#
# pip install git+https://github.com/huggingface/diffusers.git
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install python-dotenv transformers accelerate sentencepiece protobuf optimum-quanto gradio

import os
import torch
import gradio as gr
from diffusers import FluxTransformer2DModel, FluxPipeline
from huggingface_hub import login
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from dotenv import load_dotenv

load_dotenv()

hk_token = os.getenv('HF_TOKEN')

login(token=hk_token)

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=dtype)
print("Running transformer quantize DEV")
quantize(transformer, weights=qfloat8)
print("Running transformer freeze DEV")
freeze(transformer)

text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
print("Running text_encoder quantize DEV")
quantize(text_encoder_2, weights=qfloat8)
print("Running text_encoder freeze DEV")
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.enable_model_cpu_offload()

# Generate Dev Image
def gen_image_dev(prompt, steps, height, width, seed, guidance_scale):
    print("Generating...")
    image = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=int(guidance_scale),
        output_type="pil",
        num_inference_steps=int(steps),
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(int(seed))
    ).images[0]
    print("Saving...")
    return image
    # image.save(f"{prompt}.png")

# Create Gradio webapp
with gr.Blocks(theme=gr.themes.Soft(), title="NuclearGeek's Flux Capacitor") as demo:
    gr.Markdown(f"<h1 style='text-align: center; display:block'>{'NuclearGeek&apos;s Flux Capacitor'}</h1>")
    
    # Dev Tab
    with gr.Tab("FLUX.1-dev"):
        with gr.Row():

            steps_slider = gr.Slider(
                0,100,
                label = "Steps",
                value = 20,
                render = False
            )

            height_slider = gr.Slider(
                0,2048,
                label = "Height",
                value = 1024,
                render = False
            )

            width_slider = gr.Slider(
                0,2048,
                label = "Width",
                value = 1024,
                render = False
            )

            seed_slider = gr.Slider(
                0,99999999,
                label = "Seed",
                value = 0,
                render = False
            )

            guidance_slider = gr.Slider(
                0,20,
                label = "Guidance Scale",
                value = 3.5,
                render = False
            )

        chat = gr.Interface(
            fn = gen_image_dev,
            inputs = [gr.Text(label="Input Prompt"), steps_slider, height_slider, width_slider, seed_slider, guidance_slider], 
            outputs=[gr.Image(type="numpy", label="Output Image")]
        )

if __name__ == "__main__":

    demo.queue()
    # # Toggle this on if you want to share your app, change the username and password
    # demo.launch(server_port=7862, share=True, auth=("nuke", "password"))

    # Toggle this on if you want to only run local
    demo.launch()