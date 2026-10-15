import pymongo
from classifier import classify_document # Import function from the first file

# Establish connection with MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["supply_chain_context"]
collection = db["contextual_docs"]

# Fetch all documents to apply classification
for doc in collection.find():
    label = classify_document(doc.get("Product Description", ""))
    # Update the document with its new category label
    collection.update_one({"_id": doc["_id"]}, {"$set": {"Doc_Category": label}})