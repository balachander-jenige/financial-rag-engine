from src.indexing.qdrant import QdrantStore


def main():
    store = QdrantStore()

    store.create_collection()

    info = store.client.get_collection(
        store.collection_name
    )

    print(info)


if __name__ == "__main__":
    main()