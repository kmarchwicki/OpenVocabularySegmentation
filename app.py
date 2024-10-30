from typing import Optional, Tuple
from PIL import Image
import gradio as gr

from utils.models_manager import ModelsManager
from utils.utils_modes import (
    IMAGE_INFERENCE_MODES,
    IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
    VIDEO_INFERENCE_MODES,
)

MARKDOWN = """
# Florence2 + SAM2

Two-stage inference pipeline. 
1. Florence2 - open-vocabulary object detection. 
2. SAM2 - object segmentation on the image.
"""


class UIHandler:
    def __init__(self):
        self.models_manager = ModelsManager()

    def on_mode_dropdown_change(text):
        return [
            gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
        ]

    def process_image_wrapper(
        self, image_input: str, text_input: str
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        if not image_input:
            gr.Info("Please upload an image.")
            return None, None

        if not text_input:
            gr.Info("Please enter a text prompt.")
            return None, None

        return self.models_manager.process_image(image_input, text_input)

    def process_video_wrapper(
        self, video_input: str, text_input: str, progress=gr.Progress(track_tqdm=True)
    ) -> Optional[str]:
        if not video_input:
            gr.Info("Please upload a video.")
            return None

        if not text_input:
            gr.Info("Please enter a text prompt.")
            return None

        try:
            return self.models_manager.process_video(video_input, text_input)
        except ValueError as err:
            gr.Info(err)

        return None


def main() -> None:
    ui_handler = UIHandler()

    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Tab("Image"):
            image_processing_mode_dropdown_component = gr.Dropdown(
                choices=IMAGE_INFERENCE_MODES,
                value=IMAGE_INFERENCE_MODES[0],
                label="Mode",
                info="Select a mode to use.",
                interactive=True,
            )
            with gr.Row():
                with gr.Column():
                    image_processing_image_input_component = gr.Image(
                        type="pil", label="Upload image"
                    )
                    image_processing_text_input_component = gr.Textbox(
                        label="Text prompt",
                        placeholder="Enter comma separated text prompts",
                    )
                    image_processing_submit_button_component = gr.Button(
                        value="Submit", variant="primary"
                    )
                with gr.Column():
                    image_processing_image_output_component = gr.Image(
                        type="pil", label="Image output"
                    )
                    image_processing_text_output_component = gr.Textbox(
                        label="Caption output", visible=False
                    )

        with gr.Tab("Video"):
            video_processing_mode_dropdown_component = gr.Dropdown(
                choices=VIDEO_INFERENCE_MODES,
                value=VIDEO_INFERENCE_MODES[0],
                label="Mode",
                info="Select a mode to use.",
                interactive=True,
            )
            with gr.Row():
                with gr.Column():
                    video_processing_video_input_component = gr.Video(
                        label="Upload video"
                    )
                    video_processing_text_input_component = gr.Textbox(
                        label="Text prompt",
                        placeholder="Enter comma separated text prompts",
                    )
                    video_processing_submit_button_component = gr.Button(
                        value="Submit", variant="primary"
                    )
                with gr.Column():
                    video_processing_video_output_component = gr.Video(
                        label="Video output"
                    )

        # handle button clicks and text input for the image
        image_processing_submit_button_component.click(
            fn=ui_handler.process_image_wrapper,
            inputs=[
                image_processing_image_input_component,
                image_processing_text_input_component,
            ],
            outputs=[
                image_processing_image_output_component,
                image_processing_text_output_component,
            ],
        )
        image_processing_text_input_component.submit(
            fn=ui_handler.process_image_wrapper,
            inputs=[
                image_processing_image_input_component,
                image_processing_text_input_component,
            ],
            outputs=[
                image_processing_image_output_component,
                image_processing_text_output_component,
            ],
        )
        image_processing_mode_dropdown_component.change(
            ui_handler.on_mode_dropdown_change,
            inputs=[image_processing_mode_dropdown_component],
            outputs=[
                image_processing_text_input_component,
                image_processing_text_output_component,
            ],
        )

        # handle button clicks and text input for the image
        video_processing_submit_button_component.click(
            fn=ui_handler.process_video_wrapper,
            inputs=[
                video_processing_video_input_component,
                video_processing_text_input_component,
            ],
            outputs=video_processing_video_output_component,
        )
        video_processing_text_input_component.submit(
            fn=ui_handler.process_video_wrapper,
            inputs=[
                video_processing_video_input_component,
                video_processing_text_input_component,
            ],
            outputs=video_processing_video_output_component,
        )
        video_processing_mode_dropdown_component.change(
            ui_handler.on_mode_dropdown_change,
            inputs=[video_processing_mode_dropdown_component],
            outputs=[video_processing_text_input_component],
        )

    demo.launch(debug=False, show_error=True)


if __name__ == "__main__":
    main()
