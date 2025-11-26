from services.vlm_service import VLMService
from services.embedder_service import EmbedderService
from services.image_processor_service import ImageProcessorService

from search.search_engine import SearchEngine
from utils.file_utils import scan_image_folder
from utils.json_db import save_database, load_database

from config import config


def process_images_flow():
    folder = input("Enter image folder path: ").strip()
    image_paths = scan_image_folder(folder)

    if not image_paths:
        print("No valid images found.")
        return

    vlm = VLMService()
    embedder = EmbedderService()
    processor = ImageProcessorService(vlm, embedder)

    results = processor.process_images(image_paths)

    if not results:
        print("No images processed.")
        return

    if save_database(results):
        print(f"Database saved to: {config.db_path}")
    else:
        print("Failed to save database.")


def search_flow():
    db = load_database()
    if not db:
        print("Database is empty. Process images first.")
        return

    embedder = EmbedderService()
    engine = SearchEngine(embedder)

    print("Type 'exit' to stop.")
    while True:
        query = input("Search: ").strip()
        if query.lower() == "exit":
            break

        results = engine.search(query)
        if not results:
            print("No results.")
            continue

        for r in results:
            print(f"[{r['score']:.4f}] {r['path']}")
        print()


def main():
    while True:
        print("\n1. Process images")
        print("2. Search images")
        print("3. Exit")

        choice = input("Choice: ").strip()

        if choice == "1":
            process_images_flow()
        elif choice == "2":
            search_flow()
        elif choice == "3":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
