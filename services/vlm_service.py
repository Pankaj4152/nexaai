from pathlib import Path
from typing import Optional, List
import io

from nexaai import VLM
from nexaai.common import (
    GenerationConfig,
    ModelConfig,
    MultiModalMessage,
    MultiModalMessageContent
)

from logger import get_logger
from config import config


logger = get_logger(__name__)

class VLMService:
    def __init__(self):
        self.model: Optional[VLM] = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info("Loading VLM model...")

            m_cfg = ModelConfig(n_gpu_layers=config.gpu_layers)

            self.vlm = VLM.from_(
                name_or_path=config.vlm_model_path,
                mmproj_path=config.mmproj_path,
                m_cfg = m_cfg,
                plugin_id = config.plugin_id
            )

            logger.info(f"VLM model loaded successfully. | GPU Layers: {config.gpu_layers}")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise RuntimeError(f"VLM model loading failed. {e}")
        

    # State Reset
    def _reset_state(self):
        """
        Ensures the model has no leftover state between calls
        (important for consistent descriptions)
        """
        try:
            if hasattr(self.vlm, "reset"):
                self.vlm.reset()

            if hasattr(self.vlm, "_model"):
                inner = self.vlm._model
                if hasattr(inner, "reset_cache"):
                    inner.reset_cache()

            logger.debug("VLM model state reset successfully.")
        except Exception as e:
            logger.warning(f"Failed to reset VLM model state: {e}")


    def generate_description(self, image_path: str) -> Optional[str]:
        if not self.vlm:
            logger.error("VLM model is not initialized.")
            return None

        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        self._reset_state()

        prompt = (
            "Describe this image in detail. Include:"
            " objects, people, background, colors, actions,"
            " and overall context. Be descriptive and precise."
        )


        conversation = [
            MultiModalMessage(
                role="user",
                content=[
                    MultiModalMessageContent(type="text", text=prompt),
                    MultiModalMessageContent(type="image", path=image_path),
                ],
            )
        ]

        try:
            # Format prompt
            formatted_prompt = self.vlm.apply_chat_template(conversation)

            # Streaming generation
            buffer = io.StringIO()
            logger.debug(f"Generating description for: {image_path}")


            for token in self.vlm.generate_stream(
                formatted_prompt,
                g_cfg=GenerationConfig(
                    max_tokens=config.max_tokens,
                    image_paths=[image_path]
                )
            ):
                buffer.write(token)

            description = buffer.getvalue().strip()

            if not description:
                logger.warning(f"No description generated for: {image_path}")
                return None

            logger.info(f"Description generated for {image_path}")

            return description

        except Exception as e:
            logger.error(f"Failed to generate description for {image_path}: {e}")
            return None