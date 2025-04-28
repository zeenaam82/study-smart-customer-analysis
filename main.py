from src.data.loader import load_data

if __name__ == "__main__":
    test_df = load_data("data/test.ft.txt")

    print(test_df.head())