import os
from typing import Optional, Tuple
import cv2
import numpy as np
from utils.utils_annotate import BOX_ANNOTATOR, MASK_ANNOTATOR, annotate_image
from utils.utils_device import get_device_string, get_device
from utils.utils_florence import (
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
    load_florence_model,
    load_florence_processor,
    run_florence_inference,
)
from utils.utils_sam import (
    load_sam_image_model,
    load_sam_video_model,
    run_sam_inference,
)
import spaces
import torch
import supervision as sv

from PIL import Image
from tqdm import tqdm

from utils.utils_video import create_directory, delete_directory, generate_unique_name


class ModelsManager:
    def __init__(self, video_target_directory="tmp", video_scale_factor=0.5):
        self.device_str = get_device_string()
        self.device = get_device(self.device_str)
        self.florence_model = load_florence_model(device=self.device)
        self.florence_processor = load_florence_processor()
        self.sam_image_model = load_sam_image_model(device=self.device)
        self.sam_video_model = load_sam_video_model(device=self.device)
        self.video_target_directory = video_target_directory
        self.video_scale_factor = video_scale_factor

        self.video_name = generate_unique_name()
        create_directory(directory_path=self.video_target_directory)

    @spaces.GPU
    @torch.inference_mode()
    def process_image(
        self, image_input, text_input
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        with torch.autocast(device_type=self.device_str, dtype=torch.bfloat16):
            texts = [prompt.strip() for prompt in text_input.split(",")]
            detections_list = []
            for text in texts:
                _, result = run_florence_inference(
                    model=self.florence_model,
                    processor=self.florence_processor,
                    device=self.device,
                    image=image_input,
                    task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                    text=text,
                )
                detections = sv.Detections.from_lmm(
                    lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image_input.size
                )
                detections = run_sam_inference(
                    self.sam_image_model, image_input, detections
                )
                detections_list.append(detections)

            detections = sv.Detections.merge(detections_list)
            detections = run_sam_inference(
                self.sam_image_model, image_input, detections
            )
            return annotate_image(image_input, detections), None
        return image_input, None

    @spaces.GPU(duration=500)
    @torch.inference_mode()
    def process_video(self, video_input, text_input) -> Optional[str]:
        with torch.autocast(device_type=self.device_str, dtype=torch.bfloat16):
            frame_generator = sv.get_video_frames_generator(video_input)
            frame = next(frame_generator)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            texts = [prompt.strip() for prompt in text_input.split(",")]
            detections_list = []
            for text in texts:
                _, result = run_florence_inference(
                    model=self.florence_model,
                    processor=self.florence_processor,
                    device=self.device,
                    image=frame,
                    task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                    text=text,
                )
                detections = sv.Detections.from_lmm(
                    lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=frame.size
                )
                detections = run_sam_inference(self.sam_image_model, frame, detections)
                detections_list.append(detections)

            detections = sv.Detections.merge(detections_list)
            detections = run_sam_inference(self.sam_image_model, frame, detections)

            if len(detections.mask) == 0:
                raise ValueError(
                    f"No objects of class {text_input} found in the first frame of the video. "
                    "Please use different starting frame or try a different text prompt."
                )

            frame_directory_path = os.path.join(
                self.video_target_directory, self.video_name
            )
            frames_sink = sv.ImageSink(
                target_dir_path=frame_directory_path, image_name_pattern="{:05d}.jpeg"
            )

            video_info = sv.VideoInfo.from_video_path(video_input)
            video_info.width = int(video_info.width * self.video_scale_factor)
            video_info.height = int(video_info.height * self.video_scale_factor)

            frames_generator = sv.get_video_frames_generator(video_input)

            # first progress bar
            with frames_sink:
                for frame in tqdm(
                    frames_generator,
                    total=video_info.total_frames,
                    desc="splitting video into frames",
                ):
                    frame = sv.scale_image(frame, self.video_scale_factor)
                    frames_sink.save_image(frame)

            inference_state = self.sam_video_model.init_state(
                video_path=frame_directory_path,
            )

            for mask_index, mask in enumerate(detections.mask):
                _, object_ids, mask_logits = self.sam_video_model.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=mask_index,
                    mask=mask,
                )

            video_path = os.path.join(
                self.video_target_directory, f"{self.video_name}.mp4"
            )
            frames_generator = sv.get_video_frames_generator(video_input)
            masks_generator = self.sam_video_model.propagate_in_video(inference_state)

            # second progress bar
            try:
                with sv.VideoSink(video_path, video_info=video_info) as sink:
                    for frame, (_, tracker_ids, mask_logits) in zip(
                        frames_generator, masks_generator
                    ):
                        frame = sv.scale_image(frame, self.video_scale_factor)
                        masks = (mask_logits > 0.0).cpu().numpy().astype(bool)

                        if len(masks.shape) == 4:
                            masks = np.squeeze(masks, axis=1)

                        detections = sv.Detections(
                            xyxy=sv.mask_to_xyxy(masks=masks),
                            mask=masks,
                            class_id=np.array(tracker_ids),
                        )
                        annotated_frame = frame.copy()
                        annotated_frame = MASK_ANNOTATOR.annotate(
                            scene=annotated_frame, detections=detections
                        )
                        annotated_frame = BOX_ANNOTATOR.annotate(
                            scene=annotated_frame, detections=detections
                        )
                        sink.write_frame(annotated_frame)
            except (TypeError, ValueError, RuntimeError) as e:
                print(e)

            delete_directory(frame_directory_path)

            return video_path
