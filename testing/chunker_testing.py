import os
import shutil
import json

from RAG.chunker import load_corpus_from_directory, TRACKING_FILE


TEST_DIR = "test_data"

#writing test cases to check if the chunking and tracking system works as expected. We will create a test directory with some files, run the chunking process, modify one file, and check if only the modified file is re-chunked.
def setup_test_dir():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Create initial files
    with open(os.path.join(TEST_DIR, "file1.txt"), "w") as f:
        f.write("BM25 is a ranking algorithm used in search engines.")

    with open(os.path.join(TEST_DIR, "file2.txt"), "w") as f:
        f.write("Transformers are widely used in NLP tasks.")


def modify_file():
    # Modify file1
    with open(os.path.join(TEST_DIR, "file1.txt"), "w") as f:
        f.write("BM25 is a keyword-based retrieval algorithm.")


def run_test():
    print("\n--- STEP 1: Initial Processing ---")
    corpus1 = load_corpus_from_directory(TEST_DIR)
    print(f"Chunks created: {len(corpus1)}")

    print("\nTracking file after step 1:")
    with open(TRACKING_FILE) as f:
        print(json.dumps(json.load(f), indent=2))

    print("\n--- STEP 2: No Changes (Should Skip) ---")
    corpus2 = load_corpus_from_directory(TEST_DIR)
    print(f"Chunks created: {len(corpus2)} (should be 0)")

    print("\n--- STEP 3: Modify One File ---")
    modify_file()

    corpus3 = load_corpus_from_directory(TEST_DIR)
    print(f"Chunks created: {len(corpus3)} (should be > 0 but only for file1)")

    print("\nTracking file after modification:")
    with open(TRACKING_FILE) as f:
        print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    setup_test_dir()
    run_test()