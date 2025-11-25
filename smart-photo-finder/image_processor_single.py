import os
import io
import time
# import torch
from dotenv import load_dotenv
load_dotenv()

from nexaai.vlm import VLM
from nexaai.embedder import Embedder
from nexaai.common import (
    GenerationConfig,
    ModelConfig,
    MultiModalMessage,
    MultiModalMessageContent
)

class ImageProcessor:
    """
    Processes images using NexaAI:
    1. VLM describes what's in the image (image → text)
    2. Embedder converts description to vector (text → numbers)
    3. Stores vectors locally for fast semantic search
    """
    def __init__(self):
        print("🚀 Initializing NexaAI models...")

        # ===== LOAD VLM (Vision Language Model) =====
        print("\n📸 Loading Vision Language Model (VLM)...")
        
        model_path = "NexaAI/Qwen3-VL-4B-Instruct-GGUF"
        
        

        # Check for GPU availability
        # gpu_available = torch.cuda.is_available()
        # print(f"   GPU Available: {gpu_available}")
        m_cfg = ModelConfig(n_gpu_layers=0)
        
        # if gpu_available:
        #     m_cfg.n_gpu_layers = -1  # Use some GPU layers if available
        #     print("   ✅ GPU detected! Using hybrid CPU+GPU mode.")
        # else:
        #     print("   ⚠️ No GPU detected. Using CPU-only mode.")
        #     m_cfg.n_gpu_layers = 0      # Force CPU-only

        # For this version, we will use CPU-only mode

        self.vlm = VLM.from_(
            name_or_path=f"{model_path}/Qwen3-VL-4B-Instruct.Q4_0.gguf",
            mmproj_path=f"{model_path}/mmproj.F32.gguf",
            m_cfg=m_cfg,
            plugin_id="nexaml"
        )
        print("   ✅ VLM loaded successfully on CPU!")

        # Skip embedder for now - we'll add it later
        print("\n🔢 Skipping embedder for v1 (using text-based search)")
        self.embedder = None
        self.embedding_dim = 0
        
        print("\n✅ All models loaded successfully!\n")
    
    def scan_images(self, image_folder):
        """Scans the given folder for images."""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        images = []
        
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    images.append(os.path.join(root, file))
        
        print(f"📂 Found {len(images)} images in {image_folder}")
        return images
    
    def process_image(self, image_path):
        """Generate description for a single image using VLM."""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
            
            print(f"\n🖼️  Processing: {os.path.basename(image_path)}")
            
            prompt = (
                "Describe this image in detail. Include: "
                "main subjects, setting, colors, activities, and mood. "
                "Be specific and descriptive."
            )

            # ✅ FIXED: Use 'path' instead of 'image' for the image field
            conversation = [
                MultiModalMessage(
                    role="user",
                    content=[
                        MultiModalMessageContent(
                            type="text",
                            text=prompt
                        ),
                        MultiModalMessageContent(
                            type="image",
                            path=image_path  # ← CHANGED: Use 'path' not 'image'
                        )
                    ]
                )
            ]

            formatted_prompt = self.vlm.apply_chat_template(conversation)

            response_buffer = io.StringIO()
            
            print("   🤖 Generating description (CPU mode - may take 20-30 seconds)...")
            for token in self.vlm.generate_stream(
                formatted_prompt,
                g_cfg=GenerationConfig(
                    max_tokens=200,
                    image_paths=[image_path]
                )
            ):
                response_buffer.write(token)
                print(token, end="", flush=True)
            
            description = response_buffer.getvalue().strip()
            print("\n   ✅ Description generated!")
            
            return description
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_embedding(self, text):
        """Convert text description to embedding vector."""
        # For v1, just return the text itself
        return text

            
if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print("SMART PHOTO FINDER - Image Processor Test")
    print("=" * 60)
    
    try:
        processor = ImageProcessor()
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    image_path = r"D:\Pankaj\Nexa AI\nexaai\smart-photo-finder\test_images\photographer.jpeg"
    folder_path = "test_images"
    if os.path.exists(image_path):
        description = processor.process_image(image_path)
        
        if description:
            print(f"\n" + "=" * 60)
            print(f"📝 DESCRIPTION:")
            print("=" * 60)
            print(description)
            print("=" * 60)
            
            print("\n✅ Storing description as text (no embedding needed for v1)")
            print(f"   Description length: {len(description)} characters")
            
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! Image description is working!")
            print("=" * 60)
            print("\n✨ Next steps:")
            print("   1. Process multiple images")
            print("   2. Store descriptions in SQLite")
            print("   3. Build simple text search")
            print("   4. Create Gradio UI")
            print("   5. Add embeddings in v2 for better search")
    else:
        print(f"❌ Test image not found: {image_path}")

    print(f"\n⏱️  Total time: {time.time() - start_time:.2f} seconds")
