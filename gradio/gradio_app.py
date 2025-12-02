import gradio as gr
import subprocess
import os
import tempfile
import json

def generate_caption(image, epsilon, sparsity, attack_algo, num_iters):
    """
    Generate caption for the uploaded image using the model in RobustMMFMEnv.

    Args:
        image: The uploaded image from Gradio

    Returns:
        tuple: (original_caption, adversarial_caption, original_image, adversarial_image, perturbation_image)
    """
    if image is None:
        return "Please upload an image first.", "", None, None, None

    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp_file:
            tmp_image_path = tmp_file.name
            # Save the image
            from PIL import Image
            import numpy as np

            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
                img.save(tmp_image_path)
            else:
                image.save(tmp_image_path)

        # Prepare the command to run in RobustMMFMEnv
        # This is a placeholder - you'll need to create the actual script
        conda_env = "RobustMMFMEnv"
        script_path = os.path.join(os.path.dirname(__file__), "run_caption.py")

        # Run the caption generation script in the RobustMMFMEnv conda environment
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", script_path,
            "--image_path", tmp_image_path,
            "--epsilon", str(epsilon),
            "--num_iters", str(num_iters),
            "--sparsity", str(sparsity),
            "--attack_algo", attack_algo
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 seconds timeout
        )
        
        # Clean up temporary file
        os.unlink(tmp_image_path)

        if result.returncode == 0:
            # Parse the output
            output = result.stdout.strip()
            #return output if output else "No caption generated."
        
            try:
                # Parse the dictionary output
                import ast
                result_dict = ast.literal_eval(output)

                original = result_dict.get('original_caption', '').strip()
                adversarial = result_dict.get('adversarial_caption', '').strip()

                orig_img_path = result_dict.get('original_image_path')
                adv_img_path = result_dict.get('adversarial_image_path')
                pert_img_path = result_dict.get('perturbation_image_path')

                orig_image = None
                adv_image = None
                pert_image = None

                if orig_img_path and os.path.exists(orig_img_path):
                    orig_image = np.array(Image.open(orig_img_path))
                    try:
                        os.unlink(orig_img_path)
                    except:
                        pass

                if adv_img_path and os.path.exists(adv_img_path):
                    adv_image = np.array(Image.open(adv_img_path))
                    try:
                        os.unlink(adv_img_path)
                    except:
                        pass

                if pert_img_path and os.path.exists(pert_img_path):
                    pert_image = np.array(Image.open(pert_img_path))
                    try:
                        os.unlink(pert_img_path)
                    except:
                        pass

                return original, adversarial, orig_image, adv_image, pert_image  # Return 5 values

            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse output: {e}", flush=True)
                # If parsing fails, try to return raw output
                return f"Parse error: {str(e)}", "", None, None, None
        else:
            error_msg = result.stderr.strip()
            return f"Error generating caption: {error_msg}", "", None, None, None

    except subprocess.TimeoutExpired:
        return "Error: Caption generation timed out (>60s)", "", None, None, None
    except Exception as e:
        return f"Error: {str(e)}", "", None, None, None

# Create the Gradio interface
with gr.Blocks(title="Image Captioning") as demo:
    gr.Markdown("# Evaluating Robustness of Multimodal Models Against Adversarial Perturbations")
    gr.Markdown("Upload an image to generate the adversarial image and caption using the APGD/SAIF algorithm.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="numpy"
            )

            attack_algo = gr.Dropdown(
                choices=["APGD", "SAIF"],
                value="APGD",
                label="Adversarial Attack Algorithm",
                interactive=True
            )

            epsilon = gr.Slider(
                minimum=1, maximum=255, value=8, step=1, interactive=True,
                label="Epsilon (max perturbation, 0-255 scale)"
            )
            sparsity = gr.Slider(
                minimum=0, maximum=10000, value=0, step=100, interactive=True,
                label="Sparsity (L1 norm of the perturbation, for SAIF only)"
            )
            num_iters = gr.Slider(
                minimum=1, maximum=100, value=8, step=1, interactive=True,
                label="Number of Iterations"
            )

    with gr.Row():
        with gr.Column():
            generate_btn = gr.Button("Generate Captions", variant="primary")
            
    with gr.Row():
        with gr.Column():
            orig_image_output = gr.Image(label="Original Image")
            orig_caption_output = gr.Textbox(
                label="Generated Original Caption",
                lines=5,
                placeholder="Caption will appear here..."
            )
        with gr.Column():
            pert_image_output = gr.Image(label="Perturbation (10x magnified)")
        with gr.Column():
            adv_image_output = gr.Image(label="Adversarial Image")
            adv_caption_output = gr.Textbox(
                label="Generated Adversarial Caption",
                lines=5,
                placeholder="Caption will appear here..."
            )

    # Set up the button click event
    generate_btn.click(
        fn=generate_caption,
        inputs=[image_input, epsilon, sparsity, attack_algo, num_iters],
        outputs=[orig_caption_output, adv_caption_output, orig_image_output, adv_image_output, pert_image_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )
