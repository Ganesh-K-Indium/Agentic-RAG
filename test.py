from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# List all collections
collections = client.get_collections()
print("collections=", [c.name for c in collections.collections])

# Count vectors in a collection
text_count = client.count(collection_name="10K_vector_db").count
image_count = client.count(collection_name="multimodel_vector_db").count

print(f"Text vectors: {text_count}, Image vectors: {image_count}")
