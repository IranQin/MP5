import json
import os

directory = 'tasks/stone_tools'

for filename in os.listdir(directory):
    if filename.endswith(".json"): 
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r') as f:
            data = json.load(f)

        new_data = []

        for item in data:
            new_item = {
                'task': item['task'].replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"),
                'quantity': item['quantity'],
                "material": None if item['material'] is None else {key.replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"): value for key, value in item['material'].items()},
                'tool': None if item['tool'] is None else item['tool'].replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"),
                'platform': None if item['platform'] is None else  item['platform'].replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"),
                'tips': item['tips'].replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"),
                'description': item['description'].replace('_', ' ').replace("sticks", "stick").replace("ores", "ore").replace("ingots", "ingot").replace("cobblestones", "cobblestone"),
            }
            new_data.append(new_item)

        with open(file_path, 'w') as f:
            json.dump(new_data, f, indent=4)