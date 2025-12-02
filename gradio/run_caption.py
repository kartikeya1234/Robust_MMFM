"""
Script to generate captions for images using the VLM model.
This script runs in the RobustMMFMEnv conda environment.
"""

import argparse
import sys
import os
import warnings


warnings.filterwarnings('ignore')


# Add the parent directory to the path to import vlm_eval modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_caption(image_path, epsilon, sparsity, attack_algo, num_iters, model_name="open_flamingo", num_shots=0, targeted=False):
    """
    Generate caption for a single image.

    Args:
        image_path: Path to the image file
        model_name: Name of the model to use
        num_shots: Number of shots for few-shot learning

    Returns:
        str: Generated caption
    """
    try:
        # Import required modules
        from PIL import Image
        import torch
        import numpy as np
        import tempfile
        from open_flamingo.eval.models.of_eval_model_adv import EvalModelAdv
        from open_flamingo.eval.coco_metric import postprocess_captioning_generation
        from vlm_eval.attacks.apgd import APGD
        from vlm_eval.attacks.saif import SAIF  

        # Model arguments
        model_args = {
            "lm_path": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
            "lm_tokenizer_path": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "checkpoint_path": "/home/kc/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-4B-vitl-rpj3b/snapshots/df8d3f7e75bcf891ce2fbf5253a12f524692d9c2/checkpoint.pt",
            "cross_attn_every_n_layers": "2",
            "precision": "float16",
        }

        eval_model = EvalModelAdv(model_args, adversarial=True)
        eval_model.set_device(0 if torch.cuda.is_available() else -1)

        image = Image.open(image_path).convert("RGB")
        image = eval_model._prepare_images([[image]])

        prompt = eval_model.get_caption_prompt()
        
        # Generate original caption
        orig_caption = eval_model.get_outputs(
            batch_images=image,
            batch_text=[prompt],  # Note: wrapped in list
            min_generation_length=0,
            max_generation_length=20,
            num_beams=3,
            length_penalty=-2.0,
        )
        
        #orig_caption = [postprocess_captioning_generation(out).replace('"', "") for out in orig_caption
        #]

        
        
        # For adversarial attack, create the adversarial text prompt
        targeted = False  # or True if you want targeted attack
        target_str = "a dog"  # your target if targeted=True
        adv_caption = orig_caption[0] if not targeted else target_str
        prompt_adv = eval_model.get_caption_prompt(adv_caption)

        # ⭐ THIS IS THE CRITICAL MISSING STEP ⭐
        eval_model.set_inputs(
            batch_text=[prompt_adv],  # Use adversarial prompt
            past_key_values=None,
            to_device=True,
        )

        # Now run the attack
        if attack_algo == "APGD":
            attack = APGD(
                eval_model if not targeted else lambda x: -eval_model(x),
                norm="linf",
                eps=epsilon/255.0,
                mask_out=None,
                initial_stepsize=1.0,
            )

            adv_image = attack.perturb(
                image.to(eval_model.device, dtype=eval_model.cast_dtype),
                iterations=num_iters,
                pert_init=None,
                verbose=False,
            )
            
        elif attack_algo == "SAIF":
            attack = SAIF(
                    model=eval_model,
                    targeted=targeted,
                    img_range=(0,1),
                    steps=num_iters,
                    mask_out=None,
                    eps=epsilon/255.0,
                    k=sparsity,
                    ver=False
                )

            adv_image, _ = attack(
                    x=image.to(eval_model.device, dtype=eval_model.cast_dtype),
                )
        else:
            raise ValueError(f"Unsupported attack algorithm: {attack_algo}")

        adv_image = adv_image.detach().cpu()

        # Generate adversarial caption
        adv_caption_output = eval_model.get_outputs(
            batch_images=adv_image,
            batch_text=[prompt],  # Use clean prompt for generation
            min_generation_length=0,
            max_generation_length=20,
            num_beams=3,
            length_penalty=-2.0,
        )
        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in adv_caption_output
        ]

        # At the end, instead of:
# print(orig_caption[0])
# print(new_predictions[0])

# Do this - strip the list and get just the string:
        #print(orig_caption)

        orig_img_np = image.view(1,3,224,224).squeeze(0).cpu().permute(1, 2, 0).numpy()
        adv_img_np = adv_image.view(1,3,224,224).squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Calculate perturbation (difference between adversarial and original)
        perturbation = adv_img_np - orig_img_np
        # Magnify by 10x for visualization
        perturbation_magnified = perturbation * 10

        # Normalize to [0, 255] for display
        orig_img_np = ((orig_img_np - orig_img_np.min()) / (orig_img_np.max() - orig_img_np.min()) * 255).astype(np.uint8)
        adv_img_np = ((adv_img_np - adv_img_np.min()) / (adv_img_np.max() - adv_img_np.min()) * 255).astype(np.uint8)

        # Normalize perturbation to [0, 255] for visualization
        pert_img_np = ((perturbation_magnified - perturbation_magnified.min()) /
                      (perturbation_magnified.max() - perturbation_magnified.min()) * 255).astype(np.uint8)

        # ✅ Save images to temporary files
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            orig_img_path = f.name
            Image.fromarray(orig_img_np).save(orig_img_path)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            adv_img_path = f.name
            Image.fromarray(adv_img_np).save(adv_img_path)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            pert_img_path = f.name
            Image.fromarray(pert_img_np).save(pert_img_path)

        results = {
            "original_caption": orig_caption[0],
            "adversarial_caption": new_predictions[0],
            "original_image_path": orig_img_path,  # Return file paths
            "adversarial_image_path": adv_img_path,
            "perturbation_image_path": pert_img_path
        }

        return results

    except Exception as e:
        import traceback
        error_msg = f"Error in caption generation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        # Return dict with error information
        return {
            "original_caption": f"Error: {str(e)}",
            "adversarial_caption": "",
            "original_image_path": None,
            "adversarial_image_path": None,
            "perturbation_image_path": None
        }

def main():
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="open_flamingo", help="Model to use")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Epsilon for adversarial attack")
    parser.add_argument("--sparsity", type=int, default=0, help="Sparsity for SAIF attack")
    parser.add_argument("--attack_algo", type=str, default="APGD", help="Adversarial attack algorithm (APGD or SAIF)")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations for adversarial attack")

    args = parser.parse_args()

    # Generate caption
    caption = generate_caption(args.image_path, args.epsilon, args.sparsity, args.attack_algo, args.num_iters, args.model, args.shots)

    if caption:
        print(caption)
        sys.exit(0)
    else:
        print("Failed to generate caption", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
