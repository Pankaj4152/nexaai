import json
import os
import io
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from nexaai.vlm import VLM
from nexaai.common import (
    GenerationConfig,
    ModelConfig,
    MultiModalMessage,
    MultiModalMessageContent
)

class ImageProcessor:
    """
    Image Processor using NexaAI VLM and sentence-transformers embeddings.
    - VLM: Generates detailed image descriptions (Nexa SDK)
    - Embedder: Converts descriptions to vectors for semantic search
    """
    def __init__(self):
        print("🚀 Initializing AI models...")
        print("=" * 70)

        # ===== LOAD VLM (Nexa SDK) =====
        print("\n📸 Loading Vision Language Model (VLM)...")
        model_path = "NexaAI/Qwen3-VL-4B-Instruct-GGUF"
        
        m_cfg = ModelConfig(n_gpu_layers=0)  # CPU-only mode

        self.vlm = VLM.from_(
            name_or_path=f"{model_path}/Qwen3-VL-4B-Instruct.Q4_0.gguf",
            mmproj_path=f"{model_path}/mmproj.F32.gguf",
            m_cfg=m_cfg,
            plugin_id="nexaml"
        )
        print("   ✅ VLM loaded successfully!")
        print("   📦 Model: Qwen3-VL-4B (Nexa SDK)")
        print("   🖥️  Device: CPU")

        # ===== LOAD EMBEDDER (sentence-transformers) =====
        print("\n📚 Loading Text Embedder...")
        
        self.embedder = None
        self.embedding_dim = 0

        try:
            from sentence_transformers import SentenceTransformer
            
            # all-MiniLM-L6-v2: Fast, efficient, good quality
            embedder_model = 'all-MiniLM-L6-v2'
            print(f"   Loading: {embedder_model}")
            
            self.embedder = SentenceTransformer(embedder_model)
            self.embedding_dim = 384  # This model outputs 384-dim vectors
            
            print(f"   ✅ Embedder loaded successfully!")
            print(f"   📊 Embedding dimension: {self.embedding_dim}")
            print(f"   🖥️  Device: CPU")
            
        except ImportError:
            print("   ❌ sentence-transformers not installed!")
            print("   💡 Install with: pip install sentence-transformers")
            print("   ⚠️  Continuing in text-only mode")
            self.embedder = None
            
        except Exception as e:
            print(f"   ⚠️  Embedder failed: {str(e)[:100]}")
            print("   💡 Continuing in text-only mode")
            self.embedder = None

        print("\n" + "=" * 70)
        print("✅ Initialization complete!")
        print("=" * 70)
        print(f"   VLM: {'✓' if self.vlm else '✗'} Qwen3-VL-4B")
        print(f"   Embedder: {'✓' if self.embedder else '✗'} all-MiniLM-L6-v2")
        print(f"   Mode: {'Full (VLM + Embeddings)' if self.embedder else 'Text-only (VLM only)'}")
        print("=" * 70 + "\n")


    def _reset_vlm_state(self):
        """
        Reset VLM internal state between image processing calls.
        This prevents context/memory leakage between images.
        """
        try:
            # Some models have an explicit reset method
            if hasattr(self.vlm, 'reset'):
                self.vlm.reset()
            
            # Clear any cached tensors/states
            if hasattr(self.vlm, '_model'):
                if hasattr(self.vlm._model, 'reset_cache'):
                    self.vlm._model.reset_cache()
            
        except Exception as e:
            # Silent fail - not all models support this
            pass

            
    def scan_images(self, image_folder):
        """
        Recursively scan folder for image files.
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            List of image file paths
        """
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        images = []
        
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    images.append(os.path.join(root, file))
        
        print(f"📂 Found {len(images)} images in {image_folder}")
        return images
    
    def process_image(self, image_path):
        """
        Process single image: generate description + embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with path, filename, description, embedding
        """
        self._reset_vlm_state()
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
            
            print(f"\n🖼️  Processing: {os.path.basename(image_path)}")
            
            # Step 1: Generate description using VLM
            description = self._generate_description(image_path)
            
            if not description:
                print("   ❌ Failed to generate description")
                return None
            
            # Step 2: Generate embedding from description
            embedding = None
            if self.embedder is not None:
                embedding = self._generate_embedding(description)
            else:
                print("   ℹ️  Skipping embedding (text-only mode)")
            
            # Step 3: Build result
            result = {
                "path": image_path,
                "filename": os.path.basename(image_path),
                "description": description,
                "embedding": embedding.tolist() if embedding is not None else None
            }
            
            # Show preview
            preview = description[:80] + ("..." if len(description) > 80 else "")
            print(f"   📝 {preview}")
            print(f"   ✅ Complete!")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_description(self, image_path):
        """
        Internal: Generate image description using VLM.
        """
        prompt = (
            "Describe this image in detail. Include: "
            "objects, people, setting, colors, activities, and mood. "
            "Be specific and descriptive."
        )

        conversation = [
            MultiModalMessage(
                role="user",
                content=[
                    MultiModalMessageContent(type="text", text=prompt),
                    MultiModalMessageContent(type="image", path=image_path)
                ]
            )
        ]

        formatted_prompt = self.vlm.apply_chat_template(conversation)
        response_buffer = io.StringIO()
        
        print("   🤖 Generating description...", end=" ", flush=True)
        
        token_count=0
        for token in self.vlm.generate_stream(
            formatted_prompt,
            g_cfg=GenerationConfig(
                max_tokens=200,
                image_paths=[image_path]
            )
        ):
            response_buffer.write(token)
            token_count += 1
        
        if( token_count == 0 ):
            print("✗ (no response)")
            return None
        description = response_buffer.getvalue().strip()
        print("✓")
        
        
        return description if description else None
    
    def _generate_embedding(self, text):
        """
        Internal: Generate embedding vector from text.
        """
        if self.embedder is None:
            return None
        
        try:
            print("   📊 Generating embedding...", end=" ", flush=True)
            
            # Encode text to vector
            embedding = self.embedder.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            print(f"✓ ({len(embedding)}D)")
            return embedding
            
        except Exception as e:
            print(f"✗ ({str(e)[:40]})")
            return None

    def process_and_save(self, image_paths, output_file="image_database.json"):
        """
        Process multiple images and save to JSON database.
        
        Args:
            image_paths: List of image file paths
            output_file: Output JSON file path
            
        Returns:
            List of processed results
        """
        results = []
        total = len(image_paths)
        
        print(f"\n🚀 Processing {total} images...")
        print("=" * 70)
        
        start_time = time.time()
        
        for idx, img_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{total}]", end=" ")
            
            result = self.process_image(img_path)
            
            if result:
                results.append(result)
            
            # Progress estimate
            if idx > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (total - idx) * avg_time
                print(f"   ⏱️  {elapsed:.1f}s elapsed | ~{remaining:.1f}s remaining")
            
            print("-" * 70)
        
        # Save to JSON
        print(f"\n💾 Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        total_time = time.time() - start_time
        
        print(f"\n✅ Processing complete!")
        print("=" * 70)
        print(f"   📊 Successful: {len(results)}/{total} images")
        print(f"   📁 Database: {output_file}")
        print(f"   💾 File size: {file_size:.1f} KB")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   ⚡ Avg time: {total_time/total:.1f}s per image")
        print("=" * 70)
        
        return results


# ===== EXAMPLE USAGE =====
# if __name__ == "__main__":
#     print("\n" + "🎨" * 35)
#     print("       SMART PHOTO FINDER - Image Processor")
#     print("🎨" * 35 + "\n")
    
#     start_time = time.time()
    
#     # Initialize processor
#     try:
#         processor = ImageProcessor()
#     except Exception as e:
#         print(f"\n❌ Initialization failed: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)
    
#     # ===== TEST 1: Single Image =====
#     print("\n" + "=" * 70)
#     print("TEST 1: Single Image Processing")
#     print("=" * 70)
    
#     test_image = "test_images/photographer.jpeg"
    
#     if os.path.exists(test_image):
#         result = processor.process_image(test_image)
        
#         if result:
#             print(f"\n📊 RESULT:")
#             print(f"   File: {result['filename']}")
#             print(f"   Description: {result['description'][:150]}...")
            
#             if result['embedding']:
#                 emb = result['embedding']
#                 print(f"   Embedding: {len(emb)} dimensions")
#                 print(f"   Sample values: [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, ...]")
#                 print(f"   Vector norm: {np.linalg.norm(emb):.3f}")
#             else:
#                 print(f"   Embedding: None (text-only mode)")
            
#             print(f"\n✅ Single image test PASSED!")
#         else:
#             print(f"❌ Single image test FAILED!")
#     else:
#         print(f"⚠️  Test image not found: {test_image}")
    
#     # ===== TEST 2: Batch Processing =====
#     print("\n" + "=" * 70)
#     print("TEST 2: Batch Processing")
#     print("=" * 70)
    
#     folder = "test_images"
    
#     if os.path.exists(folder):
#         images = processor.scan_images(folder)
        
#         if images:
#             # Process first 3 images (or all if less than 3)
#             test_batch = images[:min(3, len(images))]
            
#             results = processor.process_and_save(
#                 test_batch,
#                 output_file="test_database.json"
#             )
            
#             # Show sample from database
#             if results:
#                 print(f"\n📋 Sample from database:")
#                 sample = results[0]
#                 print(f"   Filename: {sample['filename']}")
#                 print(f"   Description length: {len(sample['description'])} chars")
#                 print(f"   Has embedding: {'Yes' if sample['embedding'] else 'No'}")
                
#                 print(f"\n✅ Batch processing test PASSED!")
#             else:
#                 print(f"❌ No images processed successfully")
#         else:
#             print(f"⚠️  No images found in {folder}")
#     else:
#         print(f"⚠️  Test folder not found: {folder}")
    
#     # ===== SUMMARY =====
#     total_time = time.time() - start_time
    
#     print("\n" + "=" * 70)
#     print("🎉 ALL TESTS COMPLETE!")
#     print("=" * 70)
#     print(f"   Total execution time: {total_time:.2f}s")
#     print(f"   Database file: test_database.json")
#     print(f"\n   Next steps:")
#     print(f"   1. Implement search functionality")
#     print(f"   2. Build Gradio UI")
#     print(f"   3. Test with larger dataset")
#     print("=" * 70 + "\n")



if __name__ == "__main__":
    processor = ImageProcessor()
    # images = ["test_images/photographer.jpeg", "test_images/parrot.jpeg"]
    images = [os.path.join("test_images_batch", img) for img in os.listdir("test_images_batch") if img.endswith(('.jpg', '.jpeg', '.png'))]
    results = processor.process_and_save(images, output_file="image_database.json")
    # print(results)