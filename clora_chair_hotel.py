import os
import sys
import torch
import psutil
from diffusers import StableDiffusionPipeline
from pipeline_clora import CloraPipeline
from datetime import datetime

def log_info(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")

def log_error(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [ERROR] {message}")

def find_token_index(tokenizer, text, target_word):
    """Find the token index for a target word in the tokenized text"""
    tokens = tokenizer.tokenize(text)
    log_info(f"Tokens for '{text}': {tokens}")
    
    # Try different variations of the target word
    candidates = [target_word, f" {target_word}", target_word.lower(), f" {target_word.lower()}"]
    
    for candidate in candidates:
        if candidate in tokens:
            idx = tokens.index(candidate)
            log_info(f"Found token '{candidate}' at index {idx}")
            return idx
    
    # If not found, try partial matching
    for i, token in enumerate(tokens):
        if target_word.lower() in token.lower():
            log_info(f"Found partial match '{token}' for '{target_word}' at index {i}")
            return i
    
    log_error(f"Could not find token for '{target_word}' in tokens: {tokens}")
    # Return a reasonable default (usually the last meaningful token)
    return len(tokens) - 2 if len(tokens) > 2 else 1

def main():
    log_info("Starting CLoRA Chair + Hotel Room composition")

    # Check system memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    log_info(f"System memory: {memory_gb:.1f}GB total, {available_gb:.1f}GB available")

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log_info(f"Using device: {device}, dtype: {dtype}")

    if device == "cpu":
        log_info("WARNING: CUDA not available. This will be very slow on CPU.")
        if available_gb < 8:
            log_error(f"Insufficient memory for CPU execution. Need at least 8GB, have {available_gb:.1f}GB")
            log_error("Consider running on a machine with more RAM or with GPU support.")
            return False
    
    # Check for LoRA files
    chair_path = "models/chair/Tantra__Chair.safetensors"
    hotel_path = "models/hotelroom/lovehotel_SD15_V7.safetensors"
    
    if not os.path.exists(chair_path):
        log_error(f"Chair LoRA not found: {chair_path}")
        return False
    
    if not os.path.exists(hotel_path):
        log_error(f"Hotel LoRA not found: {hotel_path}")
        return False
    
    log_info("Found both LoRA files")
    
    try:
        # Load base Stable Diffusion 1.5 pipeline
        log_info("Loading Stable Diffusion 1.5 base model...")
        base = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        log_info("Base model loaded successfully")
        
        # Create CLoRA pipeline
        log_info("Creating CLoRA pipeline...")
        pipe = CloraPipeline(
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            unet=base.unet,
            scheduler=base.scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False
        ).to(device)
        log_info("CLoRA pipeline created successfully")
        
        # Load LoRAs
        log_info("Loading chair LoRA...")
        pipe.load_lora_weights(chair_path, adapter_name="chair")
        log_info("Chair LoRA loaded")
        
        log_info("Loading hotel LoRA...")
        pipe.load_lora_weights(hotel_path, adapter_name="hotel")
        log_info("Hotel LoRA loaded")
        
        # Define prompts
        prompts = [
            "a proportional chair in a well-lit hotel room, proper scale, realistic proportions",
            "a bright elegant hotel room interior with natural lighting, well-illuminated space"
        ]
        neg_prompts = [
            "oversized chair, too large, too big, disproportionate, dark, dim lighting, shadows, underexposed",
            "dark room, dim lighting, poor illumination, shadows, underexposed, gloomy"
        ]
        
        log_info(f"Prompts: {prompts}")
        
        # Find token indices
        log_info("Finding token indices...")
        chair_idx = find_token_index(pipe.tokenizer, prompts[0], "chair")
        hotel_idx = find_token_index(pipe.tokenizer, prompts[1], "hotel")
        room_idx = find_token_index(pipe.tokenizer, prompts[1], "room")
        
        # Use both hotel and room indices for the hotel concept
        hotel_indices = [hotel_idx, room_idx] if hotel_idx != room_idx else [hotel_idx]
        
        log_info(f"Chair token index: {chair_idx}")
        log_info(f"Hotel token indices: {hotel_indices}")
        
        # Construct indices for CLoRA
        important_token_indices = [
            [[chair_idx], []],  # class 0: "chair" from prompt 0
            [[], hotel_indices],  # class 1: "hotel/room" from prompt 1
        ]
        
        mask_indices = [
            [chair_idx],  # for "chair" LoRA
            hotel_indices,  # for "hotel" LoRA
        ]
        
        log_info("Token indices configured")
        
        # Generate image
        log_info("Starting image generation with CLoRA...")
        log_info("This may take several minutes...")

        try:
            images, attn_maps, masks = pipe(
                prompt_list=prompts,
                negative_prompt_list=neg_prompts,
                lora_list=["chair", "hotel"],
                important_token_indices=important_token_indices,
                mask_indices=mask_indices,
                latent_update=True,
                step_size=20,
                max_iter_to_alter=25,
                apply_mask_after=8,
                mask_threshold_alpha=0.35,
                mask_erode=False,
                mask_dilate=True,
                mask_opening=False,
                mask_closing=True,
                num_inference_steps=50,
                guidance_scale=7.5,
                use_text_encoder_lora=True,
                height=512,
                width=512,
                generator=torch.Generator(device).manual_seed(42),
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log_error("Out of memory error. Try running on a machine with more RAM or GPU.")
                log_error("For CPU execution, you need at least 16GB RAM.")
                return False
            else:
                raise e
        except Exception as e:
            log_error(f"Error during image generation: {str(e)}")
            raise e
        
        # Save the result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"chair_in_hotel_room_{timestamp}.png"
        images[0].save(output_filename)
        
        log_info(f"SUCCESS: Image saved as {output_filename}")
        log_info("CLoRA composition completed successfully!")
        
        return True
        
    except Exception as e:
        log_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
