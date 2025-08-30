import os, csv, pymongo
client = pymongo.MongoClient("mongodb+srv://ADMIN12:rf9lrC4x3PZgbI7M@realestateproject.sqf4qmt.mongodb.net")
col    = client["realestateproject"]["test"]

with open("realestate.csv", "w", newline="") as f:
    writer = None
    for doc in col.find({}, {"_id": 0}):      # drop _id if not needed
        if writer is None:                    # write header once
            writer = csv.DictWriter(f, fieldnames=doc.keys())
            writer.writeheader()
        writer.writerow(doc)
