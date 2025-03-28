from pymongo import MongoClient

# Replace the uri string with your MongoDB deployment's connection string.
client = MongoClient("mongodb://127.0.0.1:27017/")

# Connect to the database
db = client['your_database_name']

# Access a collection
collection = db['your_collection_name']

# Example
# Insert some recipes
recipes = [
    {"name": "Pancakes", "ingredients": ["flour", "milk", "eggs", "sugar", "butter"], "prep_time": 20},
    {"name": "Spaghetti Carbonara", "ingredients": ["spaghetti", "eggs", "parmesan cheese", "bacon"], "prep_time": 30},
    {"name": "Chicken Curry", "ingredients": ["chicken", "curry powder", "coconut milk", "onions", "garlic"], "prep_time": 40}
]

collection.insert_many(recipes)

# Retrieve and print all recipes
for recipe in collection.find():
    print(recipe)