from image_processor import ImageProcessor
import time


def main():
    print("🖼️  Smart Photo Finder - Powered by NexaAI")
    print("=" * 50)
    
    # Initialize processor
    processor = ImageProcessor()

    # Get folder path from user
    folder = input("Enter path to your photos folder: ").strip()

    # Scan images
    # start_time = time.time()
    images = processor.scan_images(folder)
    # print(f"⏱️  Scanning completed in {time.time() - start_time} seconds.")

    if not images:
        print("No images found! Add some photos to test_images/")
        return
    
    print(f"\n✅ Found {len(images)} images!")
    print("\nNext: Generating embeddings...")

if __name__ == "__main__":
    main()