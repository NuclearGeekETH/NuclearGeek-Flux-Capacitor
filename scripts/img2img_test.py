import torch
from diffusers.pipelines.flux import FluxImg2ImgPipeline
from diffusers.utils import load_image
from optimum.quanto import freeze, qfloat8, qint4, quantize

pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() 

# Schnell Versions
print("Running transformer quantize SCHNELL")
# Toggle whichever quantize method will work better for your system:
quantize(pipe.transformer, weights=qfloat8)
# quantize(pipe.transformer, weights=qint4, exclude="proj_out")
print("Running transformer freeze SCHNELL")
freeze(pipe.transformer)
print("Running text_encoder quantize SCHNELL")
quantize(pipe.text_encoder, weights=qfloat8)
# quantize(pipe.text_encoder, weights=qint4, exclude="proj_out")
print("Running text_encoder freeze SCHNELL")
freeze(pipe.text_encoder)

# pipe.to("cuda")
url = ("qr.png")
init_image = load_image(url)
prompt = "A cat holding a sign that says hello world"
# Depending on the variant being used, the pipeline call will slightly vary.
# Refer to the pipeline documentation for more details.
image = pipe(prompt, image=init_image, num_inference_steps=4, guidance_scale=7).images[0]
image.save("flux.png")