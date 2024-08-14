# First run these commands if using Windows and installing manually:
#
# pip install git+https://github.com/huggingface/diffusers.git
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install python-dotenv transformers accelerate sentencepiece protobuf optimum-quanto gradio

import os
import torch
import gradio as gr
from diffusers import FluxPipeline, AutoPipelineForImage2Image
from diffusers.utils import load_image
from huggingface_hub import login
from optimum.quanto import freeze, qfloat8, qint4, quantize
from dotenv import load_dotenv

load_dotenv()

hk_token = os.getenv('HF_TOKEN')

login(token=hk_token)

pipe_dev = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe_schnell = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe_img2img = AutoPipelineForImage2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, use_safetensors=True)

# Memory-efficient Diffusion Transformers with Quanto and Diffusers
# https://huggingface.co/blog/quanto-diffusers

# Dev Versions
print("Running transformer quantize DEV")
# Toggle whichever quantize method will work better for your system:
quantize(pipe_dev.transformer, weights=qfloat8)
# quantize(pipe.transformer, weights=qint4, exclude="proj_out")
print("Running transformer freeze DEV")
freeze(pipe_dev.transformer)
print("Running text_encoder quantize DEV")
quantize(pipe_dev.text_encoder, weights=qfloat8)
# quantize(pipe.text_encoder, weights=qint4, exclude="proj_out")
print("Running text_encoder freeze DEV")
freeze(pipe_dev.text_encoder)

# # Dev Img2Img Versions
# print("Running transformer quantize DEV")
# # Toggle whichever quantize method will work better for your system:
# quantize(pipe_img2img.transformer, weights=qfloat8)
# # quantize(pipe.transformer, weights=qint4, exclude="proj_out")
# print("Running transformer freeze DEV")
# freeze(pipe_img2img.transformer)
# print("Running text_encoder quantize DEV")
# quantize(pipe_img2img.text_encoder, weights=qfloat8)
# # quantize(pipe.text_encoder, weights=qint4, exclude="proj_out")
# print("Running text_encoder freeze DEV")
# freeze(pipe_img2img.text_encoder)

# # Schnell Versions
# print("Running transformer quantize SCHNELL")
# # Toggle whichever quantize method will work better for your system:
# quantize(pipe_schnell.transformer, weights=qfloat8)
# # quantize(pipe.transformer, weights=qint4, exclude="proj_out")
# print("Running transformer freeze SCHNELL")
# freeze(pipe_schnell.transformer)
# print("Running text_encoder quantize SCHNELL")
# quantize(pipe_schnell.text_encoder, weights=qfloat8)
# # quantize(pipe.text_encoder, weights=qint4, exclude="proj_out")
# print("Running text_encoder freeze SCHNELL")
# freeze(pipe_schnell.text_encoder)

# save some VRAM by offloading the model to CPU, disable this if you have enough gpu power
pipe_dev.enable_model_cpu_offload() 
# pipe_schnell.enable_model_cpu_offload() 
# pipe_img2img.enable_model_cpu_offload() 


# Generate Dev Image
def gen_image_dev(prompt, steps, height, width, seed, guidance_scale):
    print("Generating...")
    image = pipe_dev(
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

# # Generate Schnell Image
# def gen_image_schnell(prompt, steps, height, width, seed, guidance_scale):
#     print("Generating...")
#     image = pipe_schnell(
#         prompt,
#         height=int(height),
#         width=int(width),
#         guidance_scale=int(guidance_scale),
#         output_type="pil",
#         num_inference_steps=int(steps),
#         max_sequence_length=256,
#         generator=torch.Generator("cuda").manual_seed(int(seed))
#     ).images[0]
#     print("Saving...")
#     return image

# # Generate Dev Image
# def gen_image_to_image_dev(prompt, init_image, steps, height, width, seed, guidance_scale):
#     print("Generating...")
#     init_image = load_image(init_image)
#     image = pipe_img2img(
#         prompt,
#         image=init_image,
#         height=int(height),
#         width=int(width),
#         guidance_scale=int(guidance_scale),
#         output_type="pil",
#         num_inference_steps=int(steps),
#         max_sequence_length=512,
#         generator=torch.Generator("cuda").manual_seed(int(seed))
#     ).images[0]
#     print("Saving...")
#     return image

# Create Gradio webapp
with gr.Blocks(theme=gr.themes.Soft(), title="NuclearGeek's Flux Capacitor") as demo:
    gr.Markdown(f"<h1 style='text-align: center; display:block'>{'NuclearGeek&apos;s Flux Capacitor'}</h1>")
    
    # Dev Tab
    with gr.Tab("FLUX.1-dev"):
        with gr.Row():

            steps_slider = gr.Slider(
                0,100,
                label = "Steps",
                value = 50,
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
                value = 7.5,
                render = False
            )

        chat = gr.Interface(
            fn = gen_image_dev,
            inputs = [gr.Text(label="Input Prompt"), steps_slider, height_slider, width_slider, seed_slider, guidance_slider], 
            outputs=[gr.Image(type="numpy", label="Output Image")]
        )

    # Schnell Tab
    # with gr.Tab("FLUX.1-schnell"):
    #     with gr.Row():

    #         steps_slider = gr.Slider(
    #             0,100,
    #             label = "Steps",
    #             value = 4,
    #             render = False
    #         )

    #         height_slider = gr.Slider(
    #             0,2048,
    #             label = "Height",
    #             value = 1024,
    #             render = False
    #         )

    #         width_slider = gr.Slider(
    #             0,2048,
    #             label = "Width",
    #             value = 1024,
    #             render = False
    #         )

    #         seed_slider = gr.Slider(
    #             0,99999999,
    #             label = "Seed",
    #             value = 0,
    #             render = False
    #         )

    #         guidance_slider = gr.Slider(
    #             0,20,
    #             label = "Guidance Scale",
    #             value = 0,
    #             render = False
    #         )

    #     chat = gr.Interface(
    #         fn = gen_image_schnell,
    #         inputs = [gr.Text(label="Input Prompt"), steps_slider, height_slider, width_slider, seed_slider, guidance_slider], 
    #         outputs=[gr.Image(type="numpy", label="Output Image")]
    #     )

    # with gr.Tab("Image-to-Image"):
    #     with gr.Row():

    #         image = gr.Image(
    #             label = "Image Input",
    #             type = "filepath",
    #             render = False,
    #             height = "512",
    #             width = "512",
    #         )

    #         steps_slider = gr.Slider(
    #             0,100,
    #             label = "Steps",
    #             value = 50,
    #             render = False
    #         )

    #         height_slider = gr.Slider(
    #             0,2048,
    #             label = "Height",
    #             value = 1024,
    #             render = False
    #         )

    #         width_slider = gr.Slider(
    #             0,2048,
    #             label = "Width",
    #             value = 1024,
    #             render = False
    #         )

    #         seed_slider = gr.Slider(
    #             0,99999999,
    #             label = "Seed",
    #             value = 0,
    #             render = False
    #         )

    #         guidance_slider = gr.Slider(
    #             0,20,
    #             label = "Guidance Scale",
    #             value = 3.5,
    #             render = False
    #         )

    #     chat = gr.Interface(
    #         fn = gen_image_to_image_dev,
    #         inputs = [gr.Text(label="Input Prompt"), image, steps_slider, height_slider, width_slider, seed_slider, guidance_slider], 
    #         outputs=[gr.Image(type="numpy", label="Output Image")]
    #     )

if __name__ == "__main__":

    demo.queue()
    # # Toggle this on if you want to share your app, change the username and password
    # demo.launch(server_port=7862, share=True, auth=("nuke", "password"))

    # Toggle this on if you want to only run local
    demo.launch()