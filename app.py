from src.data_loader import load_all_documents



if __name__ == "__main__":
    docs=load_all_documents("data")

    print(docs)