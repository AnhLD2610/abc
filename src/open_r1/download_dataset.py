from datasets import load_dataset

def download_default_subset():
    # Load only the default subset
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")
    print(f"Downloaded default subset with {len(dataset['train'])} examples")
    return dataset

if __name__ == "__main__":
    dataset = download_default_subset()
    # Print first example to verify
    print("\nExample from the dataset:")
    print(dataset['train'][0])
