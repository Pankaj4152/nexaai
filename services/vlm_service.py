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
        logger.debug("Initializing VLMService...")
        self.model: Optional[VLM] = None
        self._load_model()
        logger.debug("VLMService initialized.")

    def _load_model(self):
        logger.info("Loading VLM model...")
        
        try:
            m_cfg = ModelConfig(n_gpu_layers=config.gpu_layers)
            logger.debug(f"ModelConfig: {m_cfg}")

            logger.debug(f"Paths -> Model: {config.vlm_model_path}, mmproj: {config.mmproj_path}")
            
            self.vlm = VLM.from_(
                name_or_path=config.vlm_model_path,
                mmproj_path=config.mmproj_path,
                m_cfg = m_cfg,
                plugin_id = config.plugin_id
            )
            logger.debug("VLM loaded successfully.")

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
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

            logger.debug("VLM model state reset.")
        except Exception as e:
            logger.warning(f"State reset failed, continuing anyway: {e}")


    def generate_description(self, image_path: str) -> Optional[str]:
        if not self.vlm:
            logger.error("VLM not initialized.")
            return None

        if not Path(image_path).exists():
            logger.warning(f"Image missing: {image_path}")
            return None
        
        logger.info(f"Processing image: {image_path}")
        logger.debug("Resetting VLM state...")
        self._reset_state()

        prompt = (
            "Describe this image in detail. Include:"
            " objects, people, background, colors, actions,"
            " and overall context. Be descriptive and precise."
        )

        # Build Conversation
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
            logger.debug(f"Generating tokens for: {image_path}")


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
                logger.warning(f"No description generated: {image_path}")
                return None

            logger.info(f"Description generated: {image_path}")
            logger.debug(f"Output length: {len(description)} chars")

            return description

        except Exception as e:
            logger.error(f"Generation failed for {image_path}: {e}")
            logger.debug("Traceback:", exc_info=True)
            return None