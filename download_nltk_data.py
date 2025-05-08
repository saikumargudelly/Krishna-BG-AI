import nltk

def download_nltk_data():
    try:
        # Download punkt tokenizer
        nltk.download('punkt')
        print("Successfully downloaded NLTK punkt data")
        
        # Download punkt_tab tokenizer
        nltk.download('punkt_tab')
        print("Successfully downloaded NLTK punkt_tab data")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    download_nltk_data() 