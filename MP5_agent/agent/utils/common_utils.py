import math
import openai
import logging
import requests
import json
import os
import re
from collections import Counter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def load_prompt(prompt):
    return load_text(f"prompts/{prompt}.txt")


def log_info(info, is_logging=True):
    if not is_logging:
        logging.disable(logging.CRITICAL + 1)

    logging.info(info)

    if not is_logging:
        logging.disable(logging.NOTSET)

    print(info)


def count_inventory(inventory_name_list, inventory_num_list):
    inventory_dict = {}
    for name, num in zip(inventory_name_list, inventory_num_list):
        if name == "air":
            continue

        if name not in inventory_dict:
            inventory_dict[name] = num
        else:
            inventory_dict[name] += num
    return inventory_dict


def share_memory(memory, events):
    inventory_name_list = events['inventory']['name'].tolist()
    inventory_num_list = events['inventory']['quantity'].tolist()
    memory.update_inventory(count_inventory(inventory_name_list, inventory_num_list))

def update_find_obj_name(obj_name):
    if obj_name == "log" or obj_name == "tree":
        return "wood"
    elif obj_name == "cobblestone":
        return "stone"
    elif obj_name =="diamond":
        return "diamond ore"
    else:
        return obj_name


def update_inventory_obj_name(obj_name):
    if obj_name == "wood" or obj_name == "tree":
        return "log"
    elif obj_name == "stone":
        return "cobblestone"
    elif obj_name =="diamond ore":
        return "diamond"
    else:
        return obj_name

def update_craft_num(craft_name, craft_num):
    if craft_name in ["stick", "planks", "bowl"]:
        craft_num = math.ceil(craft_num / 4) 
    elif craft_name in ["wooden_slab"]:
        craft_num = math.ceil(craft_num / 6) 
    elif craft_name in ["fence", "wooden_door", "iron_door"]:
        craft_num = math.ceil(craft_num / 3) 
    return craft_num


### For generating prompts
def list_dict_to_prompt(list_data):
    if len(list_data) == 0:
        return "- None\n"

    text = ""
    for dict_data in list_data:
        text += dict_to_prompt(dict_data)

    return text


def dict_to_prompt(dict_data):
    text = ""
    for key in dict_data:
        if dict_data[key]:
            text += f"- {key}: {dict_data[key]}\n"
        else:
            text += f"- {key}: None\n"
    return text


def task_to_description_prompt(task_description):
    return f"- description: {task_description}\n" 


def update_find_task_prompt(task_information, find_obj):
    if task_information["description"].lower().find("find") != -1:
        return task_information["description"].capitalize()
    else:
        return f"Find 1 {find_obj}."


### For simulating action
def simulate_mine(memory, obj, tool): 
    # Add the mined object to the inventory
    if obj == "wood" or obj =="tree":
        obj = "log"
    elif obj == "stone":
        obj = "cobblestone"
    elif obj == "diamond ore":
        obj = "diamond"

    if obj in memory.inventory:
        memory.inventory[obj] += 1
    else:
        memory.inventory[obj] = 1


def simulate_craft(memory, obj, materials, platform):

    # If all materials are available, craft the object
    for material, quantity in materials.items():
        quantity = int(quantity)
        memory.inventory[material] -= quantity
        if memory.inventory[material] == 0:
            del memory.inventory[material]

    # Add the crafted object to the inventory
    for item, quantity in obj.items():
        quantity = int(quantity)
        if item in memory.inventory:
            memory.inventory[item] += quantity
        else:
            memory.inventory[item] = quantity

def simulate_find(memory, percipient, task_information, file_path="../images/1.jpg", is_del=1):
    while True:
        memory.reset_current_environment_information()
        find_result = percipient.perceive(task_information=task_information, file_path=file_path, is_del=is_del)
        if find_result == 2:
            break