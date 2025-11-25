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
    
    load_time = time.time()
    print(f"\n⏱️  Initialization time: {load_time - start_time:.2f} seconds")
    
    folder_path = "test_images_batch"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        exit(1)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    
    print(f"\n📂 Found {len(image_files)} images to process")
    print("=" * 60)
    
    # Store results
    results = []
    
    # Process each image
    for idx, file in enumerate(image_files, 1):
        img_time_start = time.time()
        print(f"\n[{idx}/{len(image_files)}] Processing: {file}")
        
        image_path = os.path.join(folder_path, file)
        
        description = processor.process_image(image_path)
        
        img_time_end = time.time()
        
        if description:
            results.append({
                'filename': file,
                'description': description,
                'time': img_time_end - img_time_start
            })
            print(f"   ✅ Completed in {img_time_end - img_time_start:.2f}s")
        else:
            print(f"   ❌ Failed to process")
        
        print("-" * 60)
    
    # Display all results
    print("\n" + "=" * 60)
    print("📊 PROCESSING COMPLETE - ALL DESCRIPTIONS")
    print("=" * 60)
    
    for idx, result in enumerate(results, 1):
        print(f"\n[{idx}] {result['filename']}")
        print(f"Time: {result['time']:.2f}s")
        print(f"Description: {result['description']}")
        print("-" * 60)
    
    # Summary
    total_processing_time = time.time() - load_time
    total_time = time.time() - start_time
    avg_time = total_processing_time / len(results) if results else 0
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {len(results)}/{len(image_files)} images")
    print(f"⏱️  Initialization time: {load_time - start_time:.2f}s")
    print(f"⏱️  Processing time: {total_processing_time:.2f}s")
    print(f"⏱️  Average per image: {avg_time:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print("=" * 60)
