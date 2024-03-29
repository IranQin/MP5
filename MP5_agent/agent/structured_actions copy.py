# generate data of different biomes at different times of the day
from re import T
import minedojo
import random
import string
import numpy as np
from PIL import Image
import math
import pickle
from minedojo.sim import InventoryItem
import numpy as np
from minedojo.sim.mc_meta import mc as MC
import pdb
# work done by lqc and what remains to be done (short-term goals):
# 1. I have modified mine_ahead but haven't checked its effect.
# 2. I have modified underground strategy (so that the agent's current underground strategy
#    is to go in straight line no matter what), but this strategy isn't flexible and needs improvement
# 3. I have modified move_to_middle: may still be problems.
# 4. I have solved some problems with approach, but approach strategy still needs improvement. (modify try_forward, try_leftward and try_rightward)
# 5. I have modified the mine function to better aim: aim has improved but there may still be problems.
# 6. Need to devise strategy to deal with water.
# 7. If you want to observe and interact with entities, need to scan entity_name array in addition to the block_name array.
# 8. Counting of inventory items may sometimes be faulty.

dontstop = 0
steplen = 0.098 
seed = 0
vradius = 5# voxel observation radius should be consistent with lidar range
events = {}
prev_position = np.array([0,0,0])
explore_steps = 0 # keep as global variable
try_steps = 0
action_stack = []# each element is a tuple with 2 dimensions (dir,jumpornot)
stuck = 0
recently_approached_object_position = []# stores the position of the recently approached object

# Create_observation creates a txt file containing ground-truth observations provided by the minedojo interface
# under the directory SA_observation, note that you will have to create the SA_observation directory under the  
# work directory in advance.
def create_observation(env,ii):
    observation = ""
    events  = sleep(env)
    # Location Statistics
    if 'location_stats' in events:
        observation += f"Location Statistics:\n"
        observation += f"  - Position: {events['location_stats']['pos']}\n"
        observation += f"  - Compass (Yaw/Pitch): {events['location_stats']['yaw']}, {events['location_stats']['pitch']}\n"
        observation += f"  - Biome ID: {events['location_stats']['biome_id']}\n"
        observation += f"  - Rainfall: {events['location_stats']['rainfall']}\n"
        observation += f"  - Temperature: {events['location_stats']['temperature']}\n"
        observation += f"  - Can See Sky: {events['location_stats']['can_see_sky']}\n"
        observation += f"  - Is Raining: {events['location_stats']['is_raining']}\n"
        observation += f"  - Light Level: {events['location_stats']['light_level']}\n"
        observation += f"  - Sky Light Level: {events['location_stats']['sky_light_level']}\n"
        observation += f"  - Sun Brightness: {events['location_stats']['sun_brightness']}\n"
        observation += f"  - Sea Level: {events['location_stats']['sea_level']}\n\n"

  # Voxels
    if 'voxels' in events:
        observation += f"Voxels:\n"
        # observation += f"Looking Angle: {events['voxels']['cos_look_vec_angle']}"
        # observation += f"  - Block Names: {events['voxels']['block_name']}\n"
        # Create a 3D NumPy array of shape (21, 21, 21) for demonstration purposes


        # Iterate through the 3D array and append its elements to the string
        for i in range(events['voxels']['block_name'].shape[0]):
            for j in range(events['voxels']['block_name'].shape[1]):
                for k in range(events['voxels']['block_name'].shape[2]):
                    # Convert the element to a string and append it to 'observation'
                    observation += str(events['voxels']['block_name'][i, j, k]) + " "
                observation+='\n'
            observation +='\n'

        # You can remove the trailing space if needed
        # observation = observation.strip()

        # Print the resulting string
            
    
    if 'inventory' in events:
        observation += f"Inventory:\n"
        observation += f"  - Name: {events['inventory']['name']}\n"
        observation += f"  - Quantity: {events['inventory']['quantity']}\n"
        observation += f"  - Variant: {events['inventory']['variant']}\n"
        observation += f"  - Current Durability: {events['inventory']['cur_durability']}\n"
        observation += f"  - Max Durability: {events['inventory']['max_durability']}\n\n"


    # Nearby Tools
    if 'nearby_tools' in events:
        observation += f"Nearby Tools:\n"
        observation += f"  - Is Crafting Table Nearby: {events['nearby_tools']['table']}\n"
        observation += f"  - Is Furnace Nearby: {events['nearby_tools']['furnace']}\n\n"

    # Privileged Observation
    if 'rays' in events:
        observation += "LIDAR OBSERVATION\n"
        observation += "Lidar observations mainly include three parts: information about traced entities, properties of traced blocks, and directions of lidar rays themselves.\n"
        observation += f"  - Block Name: {events['rays']['block_name']}\n"
        observation += f"  - Block Distance: {events['rays']['block_distance']}\n"
        observation += f"  - Block Variant: {events['rays']['block_meta']}\n"
        observation += f"  - Block Position: ({events['rays']['traced_block_x']}, {events['rays']['traced_block_y']}, {events['rays']['traced_block_z']})\n"
        observation += f"  - Ray Yaw: {events['rays']['ray_yaw']}\n"
        observation += f"  - Ray Pitch: {events['rays']['ray_pitch']}\n"
        observation += f"  - Entity Name: {events['rays']['entity_name']}\n"
        observation += f"  - Entity Distance: {events['rays']['entity_distance']}\n"
    # with open (f"SA_observation/{ii}.txt",'w') as file:
    #     file.write(observation)
    # save_rgb_as_image(env,f"{ii}")
    return observation

# Saves a screenshot under SA_screenshots, the SA_screenshots directory must likewise be created in advance.
# Screenshot will be named by the string parameter you pass to the function.
def save_rgb_as_image(env,name):
    events  = sleep(env)
    rgb_frame = events["rgb"]
    file_path = f"../images/{name}.jpg"
    image = Image.fromarray(rgb_frame.transpose(1,2,0))
    image.save(file_path)

# Generates a random string of numbers of length 8
def generate_random_string(env,):
    events  = sleep(env)
    return ''.join(str(random.randint(0,9)) for _ in range(8))

# Decides if target object is within lidar range.
# This function should be replaced.
def lidar_detect(env,object):
    events  = sleep(env)
    if np.isin(object,events['rays']['block_name']):
        return True
    else:
        return False
    
def surrounding_voxel_detect(env,object):
    events  = sleep(env)
    detect_success = False
    # print(events['voxels']['block_name'])
    # right down
    if (events['voxels']['block_name'][vradius][vradius][vradius+1]==object):
        detect_success = True
    # right top
    if (events['voxels']['block_name'][vradius][vradius+1][vradius+1]==object):
        detect_success = True
    # forward down
    if (events['voxels']['block_name'][vradius+1][vradius][vradius]==object):
        detect_success = True
    # forward top
    if (events['voxels']['block_name'][vradius+1][vradius+1][vradius]==object):
        detect_success = True
    # left down
    if (events['voxels']['block_name'][vradius][vradius][vradius-1]==object):
        detect_success = True
    # left top
    if (events['voxels']['block_name'][vradius][vradius+1][vradius-1]==object):
        detect_success = True
    # top 
    if (events['voxels']['block_name'][vradius][vradius+2][vradius]==object):
        detect_success = True
    # down
    if (events['voxels']['block_name'][vradius][vradius-1][vradius]==object):
        detect_success = True
    return detect_success

    
# This function is not used.
def voxel_detect(env,object):
    events  = sleep(env)
    if np.isin(object,events['voxels']['block_name']):
        return True
    else:
        return False
    
# # older version
# def move_to_middle(env):# move to middle of block(upon which you are standing)
#     # moving forward
#     events  = sleep(env)
#     numerator = float(math.floor(events['location_stats']['pos'][0])+0.5-float(events['location_stats']['pos'][0]))
#     stepnum = numerator / steplen 
#     delta = stepnum - math.floor(stepnum)
#     if (delta>0.5):
#         stepnum = math.ceil(stepnum)
#     else:
#         stepnum = math.floor(stepnum) 
#     if (stepnum > 0):
#         for i in range(stepnum):
#             events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
#     elif (stepnum < 0):
#         for i in range(-stepnum):
#             events,_,_,_ = env.step([2,0,0,12,12,0,0,0])
#     # moving rightward
#     numerator = float(float(math.ceil(events['location_stats']['pos'][2])+0.5-float(events['location_stats']['pos'][2])))
#     stepnum = numerator / steplen 
#     delta = stepnum - math.floor(stepnum)
#     if (delta>0.5):
#         stepnum = math.ceil(stepnum)
#     else:
#         stepnum = math.floor(stepnum) 
#     if (stepnum > 0):
#         for i in range(stepnum):
#             events,_,_,_ = env.step([0,2,0,12,12,0,0,0])
#     elif (stepnum < 0):
#         for i in range(-stepnum):
#             events,_,_,_ = env.step([0,1,0,12,12,0,0,0])
#     print(f"present location is {events['location_stats']['pos']}")

def move_to_middle(env):# Move to middle of block(upon which you are standing).
    # moving forward
    events  = sleep(env)
    delta = events['location_stats']['pos'][0] - math.floor(events['location_stats']['pos'][0])
    if (delta<0.5):
        print(f"perpendicular delta is {delta}")
        cnt = 0
        while (events['location_stats']['pos'][0]<math.floor(events['location_stats']['pos'][0])+0.45):
            cnt += 1
            if cnt > 10:
                break
            events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
            events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            print(f"present x is {events['location_stats']['pos'][0]} and it should be larger than {math.floor(events['location_stats']['pos'][0])+0.45}")
            print(f"moving forward, present position is {events['location_stats']['pos'][0]}")
    else:
        cnt = 0
        while (events['location_stats']['pos'][0]>math.floor(events['location_stats']['pos'][0])+0.55):
            cnt += 1
            if cnt > 10:
                break
            events,_,_,_ = env.step([2,0,0,12,12,0,0,0])
            events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            print(f"moving back, present position is {events['location_stats']['pos'][0]}")
            print(f"present x is {events['location_stats']['pos'][0]} and it should be less than {math.floor(events['location_stats']['pos'][0])+0.55}")
    # moving rightward
    delta = events['location_stats']['pos'][2] - math.floor(events['location_stats']['pos'][2])
    print(f"horizontal delta is {delta}")
    if (delta<0.5):
        cnt = 0
        while (events['location_stats']['pos'][2]<math.floor(events['location_stats']['pos'][2])+0.45):
            cnt += 1
            if cnt > 10:
                break
            events,_,_,_ = env.step([0,2,0,12,12,0,0,0])
            events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            print(f"moving right, present position is {events['location_stats']['pos'][2]}")

    else:
        cnt = 0
        while (events['location_stats']['pos'][2]>math.floor(events['location_stats']['pos'][2])+0.55):
            cnt += 1
            
            if cnt > 10:
                break
            events,_,_,_ = env.step([0,1,0,12,12,0,0,0]) 
            events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            print(f"moving left, present position is {events['location_stats']['pos'][2]}")

    print(f"moving to middle, present location is {events['location_stats']['pos']}")

    
# This function returns object's relative location if desired object is within close proximity.
# The nearby function and the approach function can only target inanimate objects(tbd).
def nearby(env,object):
    events  = sleep(env)
    for i in range(vradius-1,vradius+2):
        for j in range(vradius-1,vradius+3):
            for k in range(vradius-1,vradius+2):
                # Convert the element to a string and append it to 'observation'
                if (object == events['voxels']['block_name'][i][j][k]):
                    new_tuple = (i,j,k)
                    return new_tuple
    new_tuple = (-1,-1,-1)
    return new_tuple

# # This function is not used.
# def mine_around(target,equipment):
#     events  = sleep(env)
#     if np.isin(target,events['rays']['block_name']):
#         indices = np.where(events['rays']['block_name'] == target)[0]
#         positions = []
#         for idx in indices:
#             if events['rays']['block_distance'][idx] < 3:
#                 print(f"check if {events['rays']['block_name'][idx]} is {target}")
#                 positions.append(idx)
#         if (positions):
#             print(f"mining around with positions {positions}")
#             for pos in positions:
#                 quotient, remainder = divmod(pos+1, 25)
#                 xangle, remainder = divmod((13-remainder),3)
#                 yangle,remainder = divmod((13-quotient),3)
#                 events,_,_,_ = env.step([0,0,0,12,12+xangle,0,0,0])
#                 events,_,_,_ = env.step([0,0,0,12+yangle,12,0,0,0])
#                 for i in range(10):
#                     events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
#                     save_rgb_as_image(env,f"{i}")
#                 events,_,_,_ = env.step([0,0,0,12,13,0,0,0])# turn slightly
#                 for i in range(10):
#                     events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
#                     save_rgb_as_image(env,f"again{i}")
#                 events,_,_,_ = env.step([0,0,0,12,11,0,0,0])# recover from slight turn
#                 events,_,_,_ = env.step([0,0,0,12,12-xangle,0,0,0])
#                 events,_,_,_ = env.step([0,0,0,12-yangle,12,0,0,0])# return to initial angle
#     elif np.isin(object,events['rays']['entity_name']):
#         indices = np.where(events['rays']['entity_name'] == target)[0]
#         positions = []
#         for idx in indices:
#             if events['rays']['entity_distance'][idx] < 2:
#                 positions.append(indices[idx])
#         if (positions):
#             print(f"mining around")
#             for pos in positions:
#                 quotient, remainder = divmod(pos+1, 25)
#                 xangle, remainder = divmod((13-remainder),3)
#                 yangle,remainder = divmod((13-quotient),3)
#                 events,_,_,_ = env.step([0,0,0,12,12+xangle,0,0,0])
#                 events,_,_,_ = env.step([0,0,0,12+yangle,12,0,0,0])
#                 for i in range(10):
#                     events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
#                     save_rgb_as_image(env,f"{i}")
#                 events,_,_,_ = env.step([0,0,0,12,13,0,0,0])# turn slightly
#                 for i in range(10):
#                     events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
#                     save_rgb_as_image(env,f"again{i}")
#                 events,_,_,_ = env.step([0,0,0,12,11,0,0,0])# recover from slight turn
#                 events,_,_,_ = env.step([0,0,0,12,12-xangle,0,0,0])
#                 events,_,_,_ = env.step([0,0,0,12-yangle,12,0,0,0])# return to initial angle


# Action_mine is the structured action for mining a target number of desired objects using specified equipment.
# The function loops infinitely, executing explore -> approach -> mine until the number of objects satisfy the agent's need (>= goalum)
# if agent is above ground. If agent is underground, the function executes explore -> mine in the loop, omitting the approach phase as it is
# assumed that agent must be in mining distance of desired object if he can see it underground.
def action_mine(env,object,equipment,underground,goal_num):
    events  = sleep(env)
    # for i in range(goal_num):
    #     print(f"the {i}th mine in {goal_num} times")
    #     explore_above_ground(object,underground,10)
    #     approach(object,underground)
    #     mine(object,equipment)
    
    inventory = events['inventory']['name'].tolist()
    indices = []
    totalnum = 0
    prevnum = -1# num of objects in inventory directly after last mine
    target_object = ""
    trynum = 0
    if object == "wood":
        target_object = "log"
    elif object == "stone":
        target_object = "cobblestone"
    elif object =="diamond ore":
        target_object = "diamond"
    else:
        target_object = object
    try:
        indices = [index for index, value in enumerate(inventory) if value == target_object]
    except ValueError:
        indices = []
        print(f"no equipment {equipment} found")
    if (indices):
        events = sleep(env)
        totalnum = 0
        for index in indices:
            totalnum += events['inventory']['quantity'][index]
    if (not underground):
        while (totalnum < goal_num):
            trynum += 1
            delta = totalnum-prevnum
            if (delta == 0):
                explore_above_ground(object,underground,50)
            else:
                explore_above_ground(object,underground,10)
            prevnum = totalnum
            if (approach(object,underground)):  
                # pdb.set_trace()
                mine(object,equipment,underground)

            events = sleep(env)
            inventory = events['inventory']['name'].tolist()
            try:
                indices = [index for index, value in enumerate(inventory) if value == target_object]
            except ValueError:
                indices = []
                print(f"no object {target_object} found")
            if (indices):
                print(indices)
                events = sleep(env)
                totalnum = 0
                for index in indices:
                    totalnum += events['inventory']['quantity'][index]
            
            print(f"total number of {target_object} is {totalnum}, trynum is {trynum}")
    else:
        while (totalnum < goal_num):
            trynum += 1
            delta = totalnum-prevnum
            if (delta == 0):
                explore_above_ground(object,underground,50)
            else:
                explore_above_ground(object,underground,10)

            prevnum = totalnum  
            mine(object,equipment,underground)
            events = sleep(env)
            inventory = events['inventory']['name'].tolist()

            try:
                indices = [index for index, value in enumerate(inventory) if value == target_object]
            except ValueError:
                indices = []
                print(f"no object {target_object} found")
            if (indices):
                events = sleep(env)
                totalnum = 0
                for index in indices:
                    totalnum += events['inventory']['quantity'][index]

            print(f"total number of {target_object} is {totalnum}, trynum is {trynum}")
    mine_ahead(env,)  # for craft
    # change back

# Function for mining a target with tools as specified by the parameter 'equipment'.
# Only to be used when agent is within mining distance of the target.
# Note that the target parameter in this function and the object parameter in action_mine may not be the same!(difference between wood and log)
# Target parameter is the object's name in the voxel array, object parameter is the object's name in inventory.
def mine(target,equipment,underground,env):
    print(f"executing mining of {target} with {equipment}")
    events  = sleep(env)
    global dontstop

    # cb_inventory_index = events['inventory']['name'].tolist().index(equipment)  #################for debug, could be changed
    # events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index])
    # if (np.isin(target,events['rays']['block_name']) or np.isin(object,events['rays']['entity_name'])):
    #     block_within_range = 1
    
    inventory = events['inventory']['name'].tolist()
    try:
        cb_inventory_index = inventory.index(equipment)
    except ValueError:
        cb_inventory_index = -1 
        print(f"no equipment {equipment} found")
    if (cb_inventory_index != -1):
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip tool
    print(f"found equipment {equipment}")

    if not underground:
        loopnum = 2
        for i in range (loopnum):
            if np.isin(target,events['rays']['block_name']):
                indices = np.where(events['rays']['block_name'] == target)[0]
                positions = [indices[0]]
                pos = positions[0]
                for idx in indices:
                    if events['rays']['block_distance'][idx]<3:
                        pos = idx
                        break
                print(f"{len(positions)}  positions found")
                if (positions):
                    observation = ""
                    for i in range(events['voxels']['block_name'].shape[0]):
                        for j in range(events['voxels']['block_name'].shape[1]):
                            for k in range(events['voxels']['block_name'].shape[2]):
                                # Convert the element to a string and append it to 'observation'
                                observation += str(events['voxels']['block_name'][i, j, k]) + " "
                            observation+='\n'
                        observation +='\n'
                    # print (observation)
                quotient, remainder = divmod(pos+1, 25)
                xangle, remainder_ = divmod((13-remainder),3)
                # if (remainder_ == 2):
                #     xangle += 1
                xangle -= 1# notsure
                yangle,remainder_ = divmod((13-quotient),3)
            
                # if (remainder_ == 2):
                #     yangle += 1
                yangle -= 1# notsure
                events,_,_,_ = env.step([0,0,0,12,12+xangle,0,0,0])
                events,_,_,_ = env.step([0,0,0,12+yangle,12,0,0,0])
                flag = 0
                for i in range(3):
                    if np.isin(target, events["delta_inv"]["inc_name_by_other"]):
                        flag = 1
                    events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                    # save_rgb_as_image(env,f"{i}")
                if (not flag):
                    dontstop = 1
                    print(f"mining successful")
                else:
                    dontstop = 0
                    print(f"mining UNsuccessful")

                events,_,_,_ = env.step([0,0,0,12,13,0,0,0])# turn slightly
                for i in range(3):
                    events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                    # save_rgb_as_image(env,f"again{i}")
                events,_,_,_ = env.step([0,0,0,12,11,0,0,0])# recover from slight turn
                events,_,_,_ = env.step([0,0,0,12,12-xangle,0,0,0])
                events,_,_,_ = env.step([0,0,0,12-yangle,12,0,0,0])# return to initial angle
                # positions = []
                # indices = np.where(events['rays']['block_name'] == target)[0]
                # for idx in indices:
                #     if events['rays']['block_distance'][idx] < 3:
                #         # smallest_element = events['rays']['block_distance'][idx]
                #         positions.append(idx)
                # if (positions):
                #     block_within_range = 1
                # else:
                #     block_within_range = 0
                    
            elif np.isin(object,events['rays']['entity_name']):
                indices = np.where(events['rays']['entity_name'] == target)[0]
                smallest_element = 100
                pos = indices[0]
                # for idx in indices:
                #     if events['rays']['entity_distance'][idx] < smallest_element:
                #         smallest_element = events['rays']['entity_distance'][idx]
                #         pos = idx
                quotient, remainder = divmod(pos+1, 25)
                xangle, remainder = divmod((13-remainder),3)
                yangle,remainder = divmod((13-quotient),3)
                events,_,_,_ = env.step([0,0,0,12,12+xangle,0,0,0])
                events,_,_,_ = env.step([0,0,0,12+yangle,12,0,0,0])
                for i in range(10):
                    events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                    # save_rgb_as_image(env,f"{i}")
                events,_,_,_ = env.step([0,0,0,12,13,0,0,0])# turn slightly
                for i in range(10):
                    events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                    # save_rgb_as_image(env,f"again{i}")
                events,_,_,_ = env.step([0,0,0,12,11,0,0,0])# recover from slight turn
                events,_,_,_ = env.step([0,0,0,12,12-xangle,0,0,0])
                events,_,_,_ = env.step([0,0,0,12-yangle,12,0,0,0])# return to initial angle
            else:
                print("not in range")
                # save_rgb_as_image(env,"not_in_range")
        # mine_around(object,equipment)

        print(f"Present inventory:{events['inventory']['name']}")
        print(f"Present inventory:{events['inventory']['quantity']}")
        if (not underground):
            explore_above_ground_none(env,"non-existent",0,8)                     # for underground, might bug for groud
    else:
        events = sleep(env)
        # right down
        if (events['voxels']['block_name'][vradius][vradius][vradius+1]==target):
            events,_,_,_ = env.step([0,0,0,12,18,0,0,0])
            events,_,_,_ = env.step([0,0,0,15,12,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius][vradius+1]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,9,12,0,0,0])
            events,_,_,_ = env.step([0,0,0,12,6,0,0,0])
            sleep(env)
        # right top
        if (events['voxels']['block_name'][vradius][vradius+1][vradius+1]==target):
            events,_,_,_ = env.step([0,0,0,12,18,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius+1][vradius+1]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,12,6,0,0,0])
            sleep(env)
        # forward down
        if (events['voxels']['block_name'][vradius+1][vradius][vradius]==target):
            events,_,_,_ = env.step([0,0,0,15,12,0,0,0])
            while (events['voxels']['block_name'][vradius+1][vradius][vradius]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,9,12,0,0,0])
            sleep(env)
        # forward top
        if (events['voxels']['block_name'][vradius+1][vradius+1][vradius]==target):
            while (events['voxels']['block_name'][vradius+1][vradius+1][vradius]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            sleep(env)
        # left down
        if (events['voxels']['block_name'][vradius][vradius][vradius-1]==target):
            events,_,_,_ = env.step([0,0,0,12,6,0,0,0])
            events,_,_,_ = env.step([0,0,0,15,12,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius][vradius-1]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,9,12,0,0,0])
            events,_,_,_ = env.step([0,0,0,12,18,0,0,0])
            sleep(env)
        # left top
        if (events['voxels']['block_name'][vradius][vradius+1][vradius-1]==target):
            events,_,_,_ = env.step([0,0,0,12,6,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius+1][vradius-1]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,12,18,0,0,0])
            sleep(env)
        # top
        if (events['voxels']['block_name'][vradius][vradius+2][vradius]==target):
            events,_,_,_ = env.step([0,0,0,6,12,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius+2][vradius]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,18,12,0,0,0])
            sleep(env)
        # down
        if (events['voxels']['block_name'][vradius][vradius-1][vradius]==target):
            events,_,_,_ = env.step([0,0,0,18,12,0,0,0])
            while (events['voxels']['block_name'][vradius][vradius-1][vradius]==target):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,_,_,_ = env.step([0,0,0,6,12,0,0,0])
            sleep(env)

    mine_ahead(env)   

     # equipe dirt
    inventory = events['inventory']['name'].tolist()
    try:
        cb_inventory_index = inventory.index('dirt')
    except ValueError:
        cb_inventory_index = -1 
        print(f"no equipment dirt found")
    if (cb_inventory_index != -1):
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip tool

    name = events['inventory']['name'].tolist()
    num = events['inventory']['quantity'].tolist()

    return name, num

# A function that enables agent to clear itself of obstacles in front of him,
# ONLY TO BE USED WHEN AGENT IS ABOVE GROUND! (as it is pretty destructive and does not gauge the extent of destruction until the function is finished)
def mine_ahead_aboveground(env):
    print('trying to mine ahead')
    for i in range(5):
        events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine ahead

    events,reward,ended,addinfo = env.step([0,0,0,10,12,0,0,0])
    for i in range(5):
        events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine 30 degrees upwards
    events,reward,ended,addinfo = env.step([0,0,0,8,12,0,0,0])
    for i in range(5):
        events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine above head
    events,reward,ended,addinfo = env.step([0,0,0,18,12,0,0,0])

    # events,reward,ended,addinfo = env.step([0,0,0,15,12,0,0,0])
    # for i in range(3):
    #     events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine 45 degree downwards
    # events,reward,ended,addinfo = env.step([0,0,0,9,12,0,0,0])

    events,reward,ended,addinfo = env.step([0,0,0,12,13,0,0,0])
    for i in range(5):
        events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine left 15 degrees
    events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
    for i in range(5):
        events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])# mine right 15 degrees
    events,reward,ended,addinfo = env.step([0,0,0,12,13,0,0,0])

# A function that mines the two blocks right in front of the agent's body and head to form a tunnel for passage.
# Did not check implementation, may have to rewrite and modify.
# Currently this function is only used when agent is exploring under ground.
# Agent facing north -> direction = 0; Agent facing west -> direction = 1; 
# Agent facing east -> direction = 2; Agent facing south -> direction = 3; 
######## only refine direction=0, other to be done
def mine_ahead(env,direction = 0):
    events = sleep(env)
    print('trying to mine')
    # move_to_middle(env)
    # equipe iron pickaxe in deep 
    if (events['location_stats']['pos'][1]<20):
        inventory = events['inventory']['name'].tolist()
        try:
            cb_inventory_index = inventory.index('iron pickaxe')
        except ValueError:
            cb_inventory_index = -1 
            print(f"no equipment iron pickaxe found")
        if (cb_inventory_index != -1):
            events = sleep(env)
            events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip tool
    
    if (direction == 0):
        if (events['voxels']['block_name'][vradius+1][vradius+1][vradius] not in ["air", "water"]) or (events['voxels']['block_name'][vradius+1][vradius][vradius] not in ["air", "water"]):
            while (events['voxels']['block_name'][vradius+1][vradius+1][vradius] not in ["air", "water"]):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,reward,ended,addinfo = env.step([0,0,0,15,12,0,0,0])
            while (events['voxels']['block_name'][vradius+1][vradius][vradius] not in ["air", "water"]):
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,reward,ended,addinfo = env.step([0,0,0,9,12,0,0,0])
    elif (direction == 1):
        while (events['voxels']['block_name'][vradius][vradius][vradius-1] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,10,12,0,0,0])
        while (events['voxels']['block_name'][vradius][vradius+1][vradius-1] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,14,12,0,0,0])
    elif (direction == 2):
        while (events['voxels']['block_name'][vradius][vradius][vradius+1] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,10,12,0,0,0])
        while (events['voxels']['block_name'][vradius][vradius+1][vradius+1] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,14,12,0,0,0])
    else:
        while (events['voxels']['block_name'][vradius-1][vradius][vradius] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,10,12,0,0,0])
        while (events['voxels']['block_name'][vradius-1][vradius+1][vradius] not in ["air", "water"]):
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,14,12,0,0,0])

# Function enabling the agent to move one block. Underground specifies if agent is underground or not.
# The direction parameter should fall within [0,3], 0: ahead along the positive direction of the x axis, 1: left, 2: right, 3: backward
# Regardless of whether agent is aboveground, agent will always WALK when jumpornot = 0.
# Jumpornot = 1 when underground = 0: agent will jump towards the given direction instead of walking.
# Jumpornot = 1 when underground = 1: agent will utilize mine_ahead to mine its way towards the given direction instead of walking.
def move_one_block(env,movedir=0,underground=0,jumpornot = 0):
    if underground:
        print(f"MOVEONEBLOCK: movedir is {movedir}, underground is {underground}, jumpornot is {jumpornot}")
    global explore_steps
    events  = sleep(env)
    # print(f"move one block in direction {movedir} ; whether it is necessary to jump:{jumpornot}\n")
    if (underground == 0):
        if ( not jumpornot):
            if (movedir == 0):
                try_num = 0    
                while (events['location_stats']['pos'][0]<events['location_stats']['pos'][0]+0.5):              #debug not sure
                # while (events['location_stats']['pos'][0]<math.ceil(events['location_stats']['pos'][0])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                    # print(f"present x is {events['location_stats']['pos'][0]} and it should be larger than {math.floor(events['location_stats']['pos'][0])+0.44}")
            elif (movedir == 2):# right
                try_num = 0
                while (events['location_stats']['pos'][0]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,2,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            elif (movedir == 1):# left
                try_num = 0
                while (events['location_stats']['pos'][2]<math.ceil(events['location_stats']['pos'][2])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,1,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            else:
                try_num = 0
                while (events['location_stats']['pos'][2]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([2,0,0,12,12,0,0,0]) 
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])       
        elif jumpornot:
            if (movedir == 0):
                try_num = 0    
                # while (events['location_stats']['pos'][0]<math.ceil(events['location_stats']['pos'][0])+0.45):
                while (events['location_stats']['pos'][0]<events['location_stats']['pos'][0]+0.5):               #debug, not sure
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([1,0,1,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                    # print(f"present x is {events['location_stats']['pos'][0]} and it should be larger than {math.floor(events['location_stats']['pos'][0])+0.44}")
            elif (movedir == 2):# right
                try_num = 0
                while (events['location_stats']['pos'][0]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,2,1,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            elif (movedir == 1):# left
                try_num = 0
                while (events['location_stats']['pos'][2]<math.ceil(events['location_stats']['pos'][2])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,1,1,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            else:
                try_num = 0
                while (events['location_stats']['pos'][2]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([2,0,1,12,12,0,0,0])  
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
        action_tuple = (movedir, jumpornot)
        if movedir == 0 or movedir == 2:
            action_stack.append(action_tuple)
    else:
        if ( not jumpornot):
            if (movedir == 0):
                try_num = 0    
                while (events['location_stats']['pos'][0]<math.ceil(events['location_stats']['pos'][0])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                    # print(f"present x is {events['location_stats']['pos'][0]} and it should be larger than {math.floor(events['location_stats']['pos'][0])+0.44}")
            elif (movedir == 2):# right
                try_num = 0
                while (events['location_stats']['pos'][0]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,2,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            elif (movedir == 1):# left
                try_num = 0
                while (events['location_stats']['pos'][2]<math.ceil(events['location_stats']['pos'][2])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([0,1,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
            else:
                try_num = 0
                while (events['location_stats']['pos'][2]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([2,0,0,12,12,0,0,0])  
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])      
        else:
            if (movedir == 0):
                try_num = 0    
                mine_ahead(env,0)
                while (events['location_stats']['pos'][0]<math.ceil(events['location_stats']['pos'][0])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    # mine_ahead(env,0)
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                    # print(f"present x is {events['location_stats']['pos'][0]} and it should be larger than {math.floor(events['location_stats']['pos'][0])+0.44}")
            elif (movedir == 2):# right
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                mine_ahead(env,2)
                try_num = 0
                while (events['location_stats']['pos'][0]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])

                
            elif (movedir == 1):# left
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,10,0,0,0])
                mine_ahead(env,1)
                try_num = 0
                while (events['location_stats']['pos'][2]<math.ceil(events['location_stats']['pos'][2])+0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,14,0,0,0])
                
            else:
                events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])
                try_num = 0
                while (events['location_stats']['pos'][2]>math.floor(events['location_stats']['pos'][0])-0.45):
                    if (try_num > 20):
                        return
                    try_num += 1
                    mine_ahead(env,3)
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0])
                    # events,_,_,_ = env.step([0,0,0,12,12,0,0,0])  
                events,reward,ended,addinfo = env.step([0,0,0,12,18,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,18,0,0,0])
                events,reward,ended,addinfo = env.step([0,0,0,12,18,0,0,0])
                
        action_tuple = (movedir, jumpornot)
        if movedir == 0 or movedir == 2:
            action_stack.append(action_tuple)


# This function decides whether agent can reach block ahead of him.
# If agent can reach the block ahead of him, he will call the function move_one_block and move ahead, returning true.
# If not, the function will return false and the agent will do nothing.
# Both explore_above_ground and approach call this function, the approach parameter will tell the function
# whether the function is called in explore or approach. It is necessary to distinguish between the 
# two scenarios as the agent has very different standards of path accessibility in the two cases.
# In approach, try_forward will nearly always return true as agent is desperate to get to target object.
def try_forward(env,underground,approach=0):
    events  = sleep(env)
    print(f"explore_steps is {explore_steps}, front floor is {events['voxels']['block_name'][vradius+1][vradius-1][vradius]}, block in front of body is {events['voxels']['block_name'][vradius+1][vradius][vradius]}, in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}")
    if (approach and (not underground)):
        events = sleep(env)
        if (events['voxels']['block_name'][vradius+1][vradius][vradius] == "water"):
            move_one_block(env,0,0,1)
        elif (events['voxels']['block_name'][vradius+1][vradius+1][vradius]!=("lava" or "air")):
            print(f"approaching and met with obstacle , have to mine")
            move_to_middle(env)
            mine_ahead_aboveground(env)
            prev_pos = events['location_stats']['pos']
            move_one_block(env,0,0,1)
            pres_pos = events['location_stats']['pos']
            are_equal = (prev_pos == pres_pos).all()
            if are_equal:
                return False # return false only if agent finds itself stuck
            return True
        else:
            move_one_block(env,0,0,1)
            return True
    if (not underground):
        if (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air")) and events['voxels']['block_name'][vradius+1][vradius][vradius]== "air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air":
            if (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("lava")):
                move_one_block(env,0,0,0)
                return True
            else:
                print(f"lava")
                return False
        elif (events['voxels']['block_name'][vradius+1][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius+1][vradius][vradius]=="water" or events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="water" or events['voxels']['block_name'][vradius+1][vradius+2][vradius]=="water" or events['voxels']['block_name'][vradius][vradius+1][vradius]=="water"):
            move_one_block(env,0,0,1)
            return True
        elif (events['voxels']['block_name'][vradius+1][vradius][vradius]!="air" and events['voxels']['block_name'][vradius+1][vradius][vradius]!=("lava" )) and events['voxels']['block_name'][vradius+1][vradius+1][vradius]== "air" and events['voxels']['block_name'][vradius+1][vradius+2][vradius]=="air":
            print("can jump one block to move ahead\n")
            move_one_block(env,0,0,1)
            return True
        elif (events['voxels']['block_name'][vradius+1][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius+1][vradius][vradius]=="water"):
            move_one_block(env,0,0,1)# jump if water is a head of you
        else:
            no_climb = True
            for i in range(2):
                if (events['voxels']['block_name'][vradius+1][vradius+i][vradius]=="air"):
                    continue
                else:
                    no_climb=False
                    break
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):#cgd
                    if (events['voxels']['block_name'][vradius+1][i][vradius]=="lava"):#cgd
                        print(f"can't move because of lava")
                        return False
                    if (events['voxels']['block_name'][vradius+1][i][vradius]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return True
                else:
                    move_one_block(env,0,0,0)
                    return True
            print(f"can't move because of climb")
            return False
    else:
        if (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("lava" or "water")) and events['voxels']['block_name'][vradius+1][vradius][vradius]== "air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air":
            if (events['voxels']['block_name'][vradius+2][vradius-1][vradius]!=("water") and events['voxels']['block_name'][vradius+2][vradius][vradius]!=("water") and events['voxels']['block_name'][vradius+2][vradius+1][vradius]!=("water") and events['voxels']['block_name'][vradius+2][vradius+2][vradius]!=("water") and events['voxels']['block_name'][vradius+2][vradius+3][vradius]!=("water")):
                move_one_block(env,0,0,0)
                return True
            else:
                return False
        if (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water")) and events['voxels']['block_name'][vradius+1][vradius][vradius]== "air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air":
            if (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("lava")):
                move_one_block(env,0,0,0)
                return True
            else:
                return False
        elif ((events['voxels']['block_name'][vradius+1][vradius-1][vradius]==("lava" or "water")) or events['voxels']['block_name'][vradius+1][vradius][vradius] == ("lava" or "water") or events['voxels']['block_name'][vradius+1][vradius+1][vradius]==("lava" or "water")) or events['voxels']['block_name'][vradius+2][vradius-1][vradius]==("lava" or "water") or events['voxels']['block_name'][vradius+2][vradius][vradius]==("lava" or "water") or events['voxels']['block_name'][vradius+2][vradius+1][vradius]==("lava" or "water"):
            return False# lava right in front of you
        elif ((events['voxels']['block_name'][vradius+1][vradius-1][vradius]!="lava") and events['voxels']['block_name'][vradius+1][vradius][vradius] != "lava" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]!="lava"):
            move_one_block(env,0,1,1)#solid blocks ahead, have to mine them
        
        else:
            no_climb = True
            for i in range(2):
                if (events['voxels']['block_name'][vradius+1][vradius-1+i][vradius]=="air"):
                    continue
                else:
                    no_climb=False
                    break
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):#cgd
                    if (events['voxels']['block_name'][vradius+1][i][vradius]=="lava"):#cgd
                        return False
                    if (events['voxels']['block_name'][vradius+1][i][vradius]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return False
                else:
                    move_one_block(env,0,0,0)
                    return True
            return False

# This function decides whether agent can reach block behind him.
# If agent can reach the block, he will call the function move_one_block and move backwards, returning true.
# If not, the function will return false and the agent will do nothing.
# TRY_BACKWARD is implemented very CRUDELY, feel free to modify and rewrite.
def try_backward(env,underground):# tbd: have to complete and may probably be made better use of
    events  = sleep(env)
    print(f"blocks behind: are {events['voxels']['block_name'][vradius-1][vradius-1][vradius]}, {events['voxels']['block_name'][vradius-1][vradius][vradius]},{events['voxels']['block_name'][vradius-1][vradius-1][vradius]}")
    if (events['voxels']['block_name'][vradius-1][vradius-1][vradius]!=("air" or "water")) and events['voxels']['block_name'][vradius-1][vradius][vradius]== "air" and events['voxels']['block_name'][vradius-1][vradius+1][vradius]=="air":
        if (events['voxels']['block_name'][vradius-1][vradius-1][vradius]!=("lava")):
            move_one_block(env,3,0,0)
            return True
        else:
            return False
    elif (events['voxels']['block_name'][vradius-1][vradius][vradius]!="air" and events['voxels']['block_name'][vradius-1][vradius][vradius]!=("lava" )) and events['voxels']['block_name'][vradius-1][vradius+1][vradius]== "air" and events['voxels']['block_name'][vradius-1][vradius+2][vradius]=="air":
        print("can jump one block to backward\n")
        move_one_block(env,3,0,1)
        return True
    else:
        no_climb = True
        for i in range(2):
            if (events['voxels']['block_name'][vradius-1][vradius+i][vradius]=="air"):
                continue
            else:
                no_climb=False
                break
        if (no_climb):
            no_footing = True
            for i in range(vradius - 1, -1, -1):#cgd
                if (events['voxels']['block_name'][vradius-1][i][vradius]=="lava"):#cgd
                    return False
                if (events['voxels']['block_name'][vradius-1][i][vradius]=="air"):
                    continue
                else:
                    no_footing = False
                    break
            if no_footing:
                return False
            else:
                move_one_block(env,3,0,0)
                return True
        return False
    

# This function decides whether agent can reach block on his left.
# If agent can reach the block, he will call the function move_one_block and move leftwards, returning true.
# If not, the function will return false and the agent will do nothing.
def try_leftward(env,underground):
    events  = sleep(env)
    # create_observation(env,"leftward_approach")
    if (approach):
        if events['voxels']['block_name'][vradius][vradius+1][vradius-1]!= "air"  :
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:
            print(f"mining to walk left in approach")
            observation = ""
            for i in range(events['voxels']['block_name'].shape[0]):
                for j in range(events['voxels']['block_name'].shape[1]):
                    for k in range(events['voxels']['block_name'].shape[2]):
                        # Convert the element to a string and append it to 'observation'
                        observation += str(events['voxels']['block_name'][i, j, k]) + " "
                    observation+='\n'
                observation +='\n'
            # print(f"have to mine because of observation:{observation}")
            events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])
            mine_ahead_aboveground(env)
            events,reward,ended,addinfo = env.step([0,0,0,12,18,0,0,0])
            move_one_block(env,1,0,0)
            return True
        else:
            if events['voxels']['block_name'][vradius][vradius+2][vradius-1]== "air" and events['voxels']['block_name'][vradius][vradius+2][vradius]== "air":
                move_one_block(env,1,0,1)
                return True
            else:
                move_one_block(env,1,0,0)
                return True
    if (underground):
        # print(f"explore_steps is {explore_steps}, front floor is {events['voxels']['block_name'][vradius][vradius-1][vradius+1]}, block in front of body is {events['voxels']['block_name'][vradius][vradius][vradius]}, in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}")
        if (events['voxels']['block_name'][vradius][vradius-1][vradius-1]!=("air" or "water")) and events['voxels']['block_name'][vradius][vradius][vradius-1]== "air" and events['voxels']['block_name'][vradius][vradius+1][vradius-1]=="air":
            if (events['voxels']['block_name'][vradius][vradius-1][vradius-1]!=("lava")):
                move_one_block(env,1,underground,0)
                return True 
            
        elif events['voxels']['block_name'][vradius][vradius][vradius-1]!=("air"  or "lava") and events['voxels']['block_name'][vradius][vradius+1][vradius-1]!= "air" and events['voxels']['block_name'][vradius][vradius+2][vradius-1]!="air":
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:
            observation = ""
            for i in range(events['voxels']['block_name'].shape[0]):
                for j in range(events['voxels']['block_name'].shape[1]):
                    for k in range(events['voxels']['block_name'].shape[2]):
                        # Convert the element to a string and append it to 'observation'
                        observation += str(events['voxels']['block_name'][i, j, k]) + " "
                    observation+='\n'
                observation +='\n'
            print(f"have to mine because of observation:{observation}")
            move_one_block(env,1,1,1)
            return True
        else:
            no_climb = True
            for i in range(2):
                if (events['voxels']['block_name'][vradius][vradius+i][vradius-1]=="air"):
                    continue
                else:
                    no_climb=False
                    break
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):
                    if (events['voxels']['block_name'][vradius][i][vradius-1]=="lava"):
                        return False
                    if (events['voxels']['block_name'][vradius][i][vradius-1]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return False
                else:
                    move_one_block(env,1,underground,0)
                    return True
            return False
    else:
        if (events['voxels']['block_name'][vradius][vradius-1][vradius-1]!=("lava")) and events['voxels']['block_name'][vradius][vradius][vradius-1]== "air" and events['voxels']['block_name'][vradius][vradius+1][vradius-1]=="air":
            if (events['voxels']['block_name'][vradius][vradius-1][vradius-1]!=("lava")):
                move_one_block(env,1,underground,0)
                return True
        elif (events['voxels']['block_name'][vradius][vradius][vradius-1]!="air" and events['voxels']['block_name'][vradius][vradius][vradius-1]!="lava" ) and events['voxels']['block_name'][vradius][vradius+1][vradius-1]!= "air" and events['voxels']['block_name'][vradius][vradius+2][vradius-1]!="air":
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:
                move_one_block(env,1,underground,1)
                return True
        elif ((events['voxels']['block_name'][vradius][vradius-1][vradius-1]==("lava" or "water")) or events['voxels']['block_name'][vradius][vradius][vradius-1] == ("lava" or "water") or events['voxels']['block_name'][vradius][vradius+1][vradius-1]==("lava" or "water"))or events['voxels']['block_name'][vradius][vradius-1][vradius-2]==("lava" or "water") or events['voxels']['block_name'][vradius][vradius][vradius-2]==("lava" or "water") or events['voxels']['block_name'][vradius][vradius+1][vradius-2]==("lava" or "water"):
            return False# lava right in front of you
        elif ((events['voxels']['block_name'][vradius][vradius-1][vradius-1]!="lava") and events['voxels']['block_name'][vradius][vradius][vradius-1] != "lava" and events['voxels']['block_name'][vradius][vradius+1][vradius-1]!="lava"):
            move_one_block(env,1,1,1)#solid blocks ahead, have to mine them
        
        else:
            no_climb = True
            for i in range(2):
                if (events['voxels']['block_name'][vradius][vradius-1+i][vradius-1]=="air"):
                    continue
                else:
                    no_climb=False
                    break
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):
                    if (events['voxels']['block_name'][vradius][i][vradius-1]=="lava"):
                        return False
                    if (events['voxels']['block_name'][vradius][i][vradius-1]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return False
                else:
                    move_one_block(env,1,0,0)
                    return True
            return False
        
# This function decides whether agent can reach block on his right.
# If agent can reach the block, he will call the function move_one_block and move rightwards, returning true.
# If not, the function will return false and the agent will do nothing.
def try_rightward(env,underground,approach = 0):
    events  = sleep(env)
    # create_observation(env,"righward_approach")
    if (approach):
        if events['voxels']['block_name'][vradius][vradius+1][vradius+1]!= "air" :
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:
                observation = ""
                for i in range(events['voxels']['block_name'].shape[0]):
                    for j in range(events['voxels']['block_name'].shape[1]):
                        for k in range(events['voxels']['block_name'].shape[2]):
                            # Convert the element to a string and append it to 'observation'
                            observation += str(events['voxels']['block_name'][i, j, k]) + " "
                        observation+='\n'
                    observation +='\n'
                print(f"mine rightward because of observation:{observation}")
                events,reward,ended,addinfo = env.step([0,0,0,12,18,0,0,0])
                mine_ahead_aboveground(env)
                events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])
                move_one_block(env,2,0,0)
                return True
        else:
            if events['voxels']['block_name'][vradius][vradius+2][vradius+1]== "air" and events['voxels']['block_name'][vradius][vradius+2][vradius]== "air":
                move_one_block(env,2,0,1)
                print(f"jumping rightward")
                return True
            else:
                move_one_block(env,2,0,0)
                print(f"walking rightward")
                return True
    if (not underground):
        if (events['voxels']['block_name'][vradius][vradius-1][vradius+1]!=("air" or "water")) and events['voxels']['block_name'][vradius][vradius][vradius+1]== "air" and events['voxels']['block_name'][vradius][vradius+1][vradius+1]=="air":
            if (events['voxels']['block_name'][vradius][vradius-1][vradius+1]!=("lava")):
                move_one_block(env,2,0,0)
                return True
        elif (events['voxels']['block_name'][vradius][vradius][vradius+1]!="air" and events['voxels']['block_name'][vradius][vradius][vradius+1]!=("lava" )) and events['voxels']['block_name'][vradius][vradius+1][vradius+1]== "air" and events['voxels']['block_name'][vradius][vradius+2][vradius+1]=="air" and events['voxels']['block_name'][vradius][vradius+2][vradius]=="air":
            print("can jump one block to move rightward\n")
            move_one_block(env,2,0,1)
            return True
        elif (events['voxels']['block_name'][vradius][vradius-1][vradius+1]=="water" or events['voxels']['block_name'][vradius][vradius][vradius+1]=="water"  or events['voxels']['block_name'][vradius][vradius+1][vradius]=="water"):
            move_one_block(env,2,0,1)# jump if water is on your right
        elif  events['voxels']['block_name'][vradius][vradius+1][vradius+1]!= "air" :
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:x
                return False
        else:
            no_climb = True
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):
                    if (events['voxels']['block_name'][vradius][i][vradius+1]=="lava"):
                        return False
                    if (events['voxels']['block_name'][vradius][i][vradius+1]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return False
                else:
                    move_one_block(env,2,0,0)
                    return True
            return False
    else:
        if (events['voxels']['block_name'][vradius][vradius-1][vradius+1]!=("lava")) and events['voxels']['block_name'][vradius][vradius][vradius+1]== "air" and events['voxels']['block_name'][vradius][vradius+1][vradius+1]=="air":
            if (events['voxels']['block_name'][vradius][vradius-1][vradius+1]!=("lava")):
                move_one_block(env,2,1,0)
                return True
        elif (events['voxels']['block_name'][vradius][vradius][vradius+1]!="air" and events['voxels']['block_name'][vradius][vradius][vradius+1]!="lava" ) and events['voxels']['block_name'][vradius][vradius+1][vradius+1]== "air" and events['voxels']['block_name'][vradius][vradius+2][vradius+1]=="air":
            # if (events['voxels']['block_name'][vradius][vradius-1][vradius]=="water" or events['voxels']['block_name'][vradius][vradius][vradius]=="water"):
            #     return False
            # else:
                move_one_block(env,2,1,1)
                return True
        elif ((events['voxels']['block_name'][vradius][vradius-1][vradius+1]==("lava" or "water")) or events['voxels']['block_name'][vradius][vradius][vradius+1] == ("lava" or "water") or events['voxels']['block_name'][vradius][vradius+1][vradius+1]==("lava" or "water"))or events['voxels']['block_name'][vradius][vradius-1][vradius+2]==("lava" or "water") or events['voxels']['block_name'][vradius][vradius][vradius+2]==("lava" or "water") or events['voxels']['block_name'][vradius][vradius+1][vradius+2]==("lava" or "water"):
            return False# lava right in front of you
        elif ((events['voxels']['block_name'][vradius][vradius-1][vradius+1]!="lava") and events['voxels']['block_name'][vradius][vradius][vradius+1] != "lava" and events['voxels']['block_name'][vradius][vradius+1][vradius+1]!="lava"):
            move_one_block(env,2,1,1)#solid blocks ahead, have to mine them
        
        else:
            no_climb = True
            for i in range(2):
                if (events['voxels']['block_name'][vradius][vradius-1+i][vradius+1]=="air"):
                    continue
                else:
                    no_climb=False
                    break
            if (no_climb):
                no_footing = True
                for i in range(vradius - 1, -1, -1):
                    if (events['voxels']['block_name'][vradius][i][vradius+1]=="lava"):
                        return False
                    if (events['voxels']['block_name'][vradius][i][vradius+1]=="air"):
                        continue
                    else:
                        no_footing = False
                        break
                if no_footing:
                    return False
                else:
                    move_one_block(env,2,0,0)
                    return True
            return False
    
direction = 0
retry_times = 0
# the function name explore_above_ground may be misleading, it is actually a function
# which can be used both in above-ground and underground scenarios,
# set parameter underground to 0 if above ground, set it to 1 if underground.
# once the agent has explored max_try_steps number of steps, it will stop regardless of whether object is within range.
def explore_above_ground(env,object,underground,memory,percipient,task_information,max_try_steps=10000):
    global explore_steps
    # events  = sleep(env)
    global stuck
    global direction
    global prev_position
    global retry_times
    global dontstop
    events = sleep(env)
    print(f"exploring once, front floor{events['voxels']['block_name'][vradius+1][vradius-1][vradius]}\n block in front of body is {events['voxels']['block_name'][vradius+1][vradius][vradius]} \n and block in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}\n ")
    
    # # make sure you drop tool into inventory before exploring so as not to waste them
    # inventory = events['inventory']['name'].tolist()
    # cb_inventory_index = inventory.index('air')
    # events,_,_,_ = env.step([0,0,0,16,12,0,0,0])
    # events = sleep(env)
    # events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip crafting tabsle
    if (not underground):
        find_time = 0
        for i in range(max_try_steps):
            find_time+=1
            if i >= 2 and dontstop == 1:
                dontstop = 0
            # create_observation(env,f"observation_{i}")
            print(f"try step is {i} and dir is {direction}")
            print(f"explore step is {explore_steps} and position is {events['location_stats']['pos']}")

            save_rgb_as_image(env,f"{i}")

            # if lidar_detect(env,object) and (i!=0 and dontstop == 0):
            # # if voxel_detect(env,object) and (i!=0 and dontstop == 0):
            #     print("Found object {object}!!!yay")
            #     return True
            print(find_time)
            print(explore_steps)
            if find_time%10==0:
                file_path = f"../images/{i}.jpg"
                
                memory.reset_current_environment_information()
                find_result = percipient.perceive(task_information=task_information, find_obj=object, file_path=file_path)
                if find_result == 2:
                    print("Find successfully!")
                    return True


            if explore_steps >= 10000 :
                print("explore steps exceed limit")
                explore_steps = 0
                return False
            if (prev_position[0] != events['location_stats']['pos'][0]) or (prev_position[2] != events['location_stats']['pos'][2]):
                stuck = 0
            else:
                stuck += 1
            prev_position = events['location_stats']['pos']
    
            if stuck>2:
                print(f"stuck!!!")
                stuck = 0
                observation = ""
                for i in range(events['voxels']['block_name'].shape[0]):
                    for j in range(events['voxels']['block_name'].shape[1]):
                        for k in range(events['voxels']['block_name'].shape[2]):
                            # Convert the element to a string and append it to 'observation'
                            observation += str(events['voxels']['block_name'][i, j, k]) + " "
                        observation+='\n'
                    observation +='\n'
                # print(f"stuck observation:{observation}")
                mined_ahead = 0
                
                while (True):# an issue here
                    retry_times += 1
                    if (retry_times > 2):
                        observation = ""
                        for i in range(events['voxels']['block_name'].shape[0]):
                            for j in range(events['voxels']['block_name'].shape[1]):
                                for k in range(events['voxels']['block_name'].shape[2]):
                                    # Convert the element to a string and append it to 'observation'
                                    observation += str(events['voxels']['block_name'][i, j, k]) + " "
                                observation+='\n'
                            observation +='\n'
                        print(f"had to mine ahead")
                        move_to_middle(env)
                        mine_ahead_aboveground(env)
                        mined_ahead = 1
                        stuck = 0
                        direction = 0
                        retry_times = 0
                        break
                    print(f"retry times is {retry_times}")

                    while(action_stack and action_stack[-1][0]==2 and mined_ahead == 0):#action's movedir was "go right", which means that agent had no choice but go right
                        action_tuple = action_stack.pop()
                        if (action_tuple[0] == 2):
                            if not (try_leftward(env,underground)):
                                print(f"had to mine ahead because can't go left")
                                move_to_middle(env)
                                mine_ahead_aboveground(env)
                                mined_ahead = 1
                                direction = 0
                                retry_times = 0
                                stuck = 0
                                break
                        
                        elif (action_tuple[0] == 0):
                            if (not try_backward(env,underground)):
                                print(f"had to mine ahead because can't go back")
                                move_to_middle(env)
                                mine_ahead_aboveground(env)
                                mined_ahead = 1
                                direction = 0
                                retry_times = 0
                                stuck = 0
                                break
                            retry_times = 0
                            stuck = 0
                        # move_one_block(env,3-action_tuple[0],0,1-action_tuple[1])
                    if (not action_stack ) and (not mined_ahead):
                        print("Exception encountered, may be stuck permanently!! but i will venture a step forward")# tbd: think of some other way to let agent extricate himself
                        move_to_middle(env)
                        mine_ahead_aboveground(env)
                        direction = 0
                        # return
                    elif (not mined_ahead):
                        action_tuple = action_stack.pop()
                        # print(f"action_tuple's direction is {action_tuple[0]}")
                        move_one_block(env,3,0,1-action_tuple[1])
                        stuck = 0
                        if (try_rightward(env,underground,0)):
                            print(f"moved rightward in new endeavor and position now is {events['location_stats']['pos']}")
                            direction = 0
                            break
                if (not action_stack):
                    print("Exception encountered, stuck permanently!!")# tbd: think of some other way to let agent extricate himself
                    # return
            if direction == 0:
                if (not try_forward(env,underground)):
                    print(f"meant to go forward, rightward instead")
                    observation = ""
                    for i in range(events['voxels']['block_name'].shape[0]):
                        for j in range(events['voxels']['block_name'].shape[1]):
                            for k in range(events['voxels']['block_name'].shape[2]):
                                # Convert the element to a string and append it to 'observation'
                                observation += str(events['voxels']['block_name'][i, j, k]) + " "
                            observation+='\n'
                        observation +='\n'
                    print(observation)
                    direction = 2
                    continue
                    # record this position in stack
                    # add another 
                else:
                    print("went forward as planned, want to continue forward")
                    direction = 0
                    continue
            if direction == 1:
                if (not try_leftward(env,underground)):
                    direction = 0
                    continue
                else:
                    direction = 0
                    continue
            if direction == 2:
                if (not try_rightward(env,underground,0)):
                    print(f"meant to go rightward, but can't")
                    observation = ""
                    for i in range(events['voxels']['block_name'].shape[0]):
                        for j in range(events['voxels']['block_name'].shape[1]):
                            for k in range(events['voxels']['block_name'].shape[2]):
                                # Convert the element to a string and append it to 'observation'
                                observation += str(events['voxels']['block_name'][i, j, k]) + " "
                            observation+='\n'
                        observation +='\n'
                    print(observation)
                    direction = 0
                    continue
                else:
                    print("went rightward as planned, want to go forward now")
                    direction = 0
                    continue
        print("end of exploration")
        return False
    else:
        for i in range(max_try_steps):
            # create_observation(env,f"observation_{i}")
            # move_one_block(env,0,1,1)
            events = sleep(env)
            print(f"try step is {i} and dir is {direction}")
            print(f"explore step is {explore_steps} and position is {events['location_stats']['pos']}")
            if surrounding_voxel_detect(env, object):
                print("Found object {object}!!!yay")
                return True
            if explore_steps >= 10000:
                print("explore steps exceed limit")
                explore_steps = 0
                return False
            # print(events['voxels']['block_name'])
            move_one_block(env,0,1,1)

# go out to top             
def go_out(env):
    events = sleep(env)
    curlevel = events['location_stats']['pos'][1]
    out_level = curlevel + 10
    go_up(out_level)
    for i in range(10):
        events,_,_,_ = env.step([1,0,0,12,12,0,0,0])

def go_down_to_y_level(env,goal_level,equipment = ""):
    events  = sleep(env)
    move_to_middle(env)
    inventory = events['inventory']['name'].tolist()
    try:
        cb_inventory_index = inventory.index(equipment)
    except ValueError:
        cb_inventory_index = -1 
        print(f"no equipment {equipment} found")
    if (cb_inventory_index != -1):
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip tool
        print(f"found equipment {equipment}")

    curlevel = events['location_stats']['pos'][1]
    pre_level = curlevel
    if (curlevel > goal_level):
        events,reward,ended,addinfo = env.step([0,0,0,18,12,0,0,0])  
        while (curlevel > goal_level):
            print(f"present level is { events['location_stats']['pos'][1]}")
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            events,reward,ended,addinfo = env.step([0,0,0,12,12,3,0,0])
            sleep(env)
            curlevel = events['location_stats']['pos'][1]
            if (curlevel == pre_level):
                events,reward,ended,addinfo = env.step([1,0,0,12,12,0,0,0])
        events,reward,ended,addinfo = env.step([0,0,0,6,12,0,0,0])  
    else:
    # events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])  
        cb_inventory_index = events['inventory']['name'].tolist().index('dirt')  #################for debug, could be changed
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip dirt
        return
    cb_inventory_index = events['inventory']['name'].tolist().index('dirt')  #################for debug, could be changed
    events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip dirt

    
# The function for approaching a desired object once it has already been sighted.
def approach(env,object,underground):# tbd: scanning blocknames not enough if you want to approach live entity
    if underground:
        return True
    print(f"executing approach")
    events  = sleep(env)
    events = sleep(env)
    object_x = events['location_stats']['pos'][0]
    object_y = events['location_stats']['pos'][1]
    object_z = events['location_stats']['pos'][2]
    x_found = 0
    for x_index in range(vradius, vradius*2+1):
        for y_index in range(0, vradius*2+1):
            for z_index in range(0, vradius*2 + 1):
                if events['voxels']['block_name'][x_index][y_index][z_index] == object:
                    x_found = True
                    print("found block")
                    print(f"x coordinate is {events['location_stats']['pos'][2]}, z_index is {z_index}")
                    object_z = z_index-vradius+events['location_stats']['pos'][2]
                    break  # Exit the innermost loop if x is found
            if x_found:
                object_y = y_index-vradius+events['location_stats']['pos'][1]
                break  # Exit the middle loop if x is found
        if x_found:
            object_x = x_index-vradius+events['location_stats']['pos'][0]
            break  # Exit the outermost loop if x is found
    print(f"present position is {events['location_stats']['pos'][0]},{events['location_stats']['pos'][1]},{events['location_stats']['pos'][2]}")
    # print(f"approaching, walking right. present position is {events['location_stats']['pos']}, goal position is {object_x},{object_y},{object_z}")
    print(f"goal position is {object_x},{object_y},{object_z}")
    if (not x_found):
        try_forward(env,underground,1)
        return False
    else:
        observation = ""
        for i in range(events['voxels']['block_name'].shape[0]):
            for j in range(events['voxels']['block_name'].shape[1]):
                for k in range(events['voxels']['block_name'].shape[2]):
                    # Convert the element to a string and append it to 'observation'
                    observation += str(events['voxels']['block_name'][i, j, k]) + " "
                observation+='\n'
            observation +='\n'
        print(observation)
    
    if (events['location_stats']['pos'][2]<object_z-1):
        try_num = 0
        print(f"have to move right")
        while (events['location_stats']['pos'][2]<object_z-1):
            print(f"present position is {events['location_stats']['pos'][0]},{events['location_stats']['pos'][1]},{events['location_stats']['pos'][2]}")
            print(f"approaching, walking right. present position is {events['location_stats']['pos']}, goal position is {object_x},{object_y},{object_z}")
            try_num += 1
            # if (nearby(env,object)):
            #     print(f"found desired {object} nearby")
            #     mine(object,"")
            try_rightward(env,underground,1)
            events = sleep(env)
            if (try_num>10):
                observation = ""
                for i in range(events['voxels']['block_name'].shape[0]):
                    for j in range(events['voxels']['block_name'].shape[1]):
                        for k in range(events['voxels']['block_name'].shape[2]):
                            # Convert the element to a string and append it to 'observation'
                            observation += str(events['voxels']['block_name'][i, j, k]) + " "
                        observation+='\n'
                    observation +='\n'
                print("stuck trying to go right in APPROACH! \n with observation ")
                return False
                break
    elif (events['location_stats']['pos'][2]>object_z+1):
        print(f"have to move left")
        try_num = 0
        observation = ""
        for i in range(events['voxels']['block_name'].shape[0]):
            for j in range(events['voxels']['block_name'].shape[1]):
                for k in range(events['voxels']['block_name'].shape[2]):
                    # Convert the element to a string and append it to 'observation'
                    observation += str(events['voxels']['block_name'][i, j, k]) + " "
                observation+='\n'
            observation +='\n'
        while (events['location_stats']['pos'][2]>object_z+1):
            print(f"approaching, walking left. present position is {events['location_stats']['pos']}, goal position is {object_x},{object_y},{object_z}\n with observation")
            # if (nearby(env,object)):
            #     mine(object,"")
            try_num += 1
            try_leftward(env,underground)
            events = sleep(env)
            if (try_num>10):
                print("stuck trying to go left in APPROACH!")
                return False
                break 
    else:
        return True
                
    
    print(f"finished moving sideways")
    # tbd: should select closest block instead of random block
    try_num = 0
    # print("##############################################!")
    # print(events['location_stats']['pos'][0])
    # print(object_x-1)
    while (events['location_stats']['pos'][0]<object_x-1):
        print(f"approaching, walking ahead. present position is {events['location_stats']['pos']}, goal position is {object_x},{object_y},{object_z}")
        # if (nearby(env,object)):
        #     mine(object,"")
        # move_forward()
        if (try_forward(env,0,1)==False):
            try_rightward(env,0,1)
            break
        # events = sleep(env)
        # print("##############################################!")
        # print(events['location_stats']['pos'][0])
        # print(object_x-1)
        try_num += 1
        if (try_num>30):
            print("stuck trying to go left in APPROACH!\n with observation")
            break  
    if ((not nearby(env,object)) and events['location_stats']['pos'][0]<object_z-1):
        go_up(env,object_z-1)
    elif ((not nearby(env,object)) and events['location_stats']['pos'][0]>object_z+1):
        go_down_to_y_level(object_z+1)

    print(f"APPROACH ended:). present position is {events['location_stats']['pos']}, goal position is {object_x},{object_y},{object_z}")
    observation = ""
    for i in range(events['voxels']['block_name'].shape[0]):
        for j in range(events['voxels']['block_name'].shape[1]):
            for k in range(events['voxels']['block_name'].shape[2]):
                # Convert the element to a string and append it to 'observation'
                observation += str(events['voxels']['block_name'][i, j, k]) + " "
            observation+='\n'
        observation +='\n'
    print(f"observation is {observation}")
    # save_rgb_as_image(env,"approached_goal")
    newpos = []
    newpos.append(events['location_stats']['pos'][0])
    newpos.append(events['location_stats']['pos'][1])
    newpos.append(events['location_stats']['pos'][2])
    global recently_approached_object_position
    recently_approached_object_position = newpos
    
    # print("##############################################################")
    # print(events['location_stats']['pos'][0])
    # print(object_x-1)
    if np.isin(object,events['rays']['block_name']):# tbd: didn't consider entities
        indices = np.where(events['rays']['block_name'] == object)[0]
        # positions = [indices[0]]
        # pos = positions[0]
        for idx in indices:
            if events['rays']['block_distance'][idx]<3:
                return True
    return False

def move_forward(env):
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])
    _,_,_,_ = env.step([0,0,1,12,12,0,0,0])
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])
    _,_,_,_ = env.step([1,0,0,12,12,0,0,0])

# def attack(object, equipment):
#     right = 0
#     front = 0
#     curdir = 0# 1: 90 anticlockwise, 2: facing backwards, 3: 90 clockwise
#     if (recently_approached_object_position[0]>events['location_stats']['pos'][0]):
#         front = 1
#     elif (recently_approached_object_position[2]>events['location_stats']['pos'][2]):
#         right = 1
#     frontdistance = abs(recently_approached_object_position[0]-events['location_stats']['pos'][0])
#     rightdistance = abs(recently_approached_object_position[2]-events['location_stats']['pos'][2])
#     # if (front and right and frontdistance and rightdistance):
        
#     # elif (front and (not right) and frontdistance and rightdistance):
#     # elif (front):
#     # elif (right):
#     # elif (not right):
MC_ITEM_IDS = MC.MC_ITEM_IDS

# A function which allows the agent to do nothing for 'duration' timesteps.
def sleep(env, duration = 1):
    for i in range (duration):
        _,_,_,_ = env.step([0,0,0,12,12,0,0,0])
        _,_,_,_ = env.step([0,0,0,12,12,0,0,0])
        _,_,_,_ = env.step([0,0,0,12,12,0,0,0])
        _,_,_,_ = env.step([0,0,0,12,12,0,0,0])
        events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
    return events

# The structured action for craft, it first finds level ground for a crafting table 
# if crafting table is needed, then proceeds to craft the desired tool.
def action_craft(env, item, use_crafting_table,use_furnace,craft_num):
    """
    Craft item
    :env: minedojo env
    :item: item need to be crafted: string
    """
    events  = sleep(env)
    equipment = "dirt"
    inventory = events['inventory']['name'].tolist()
    try:
        cb_inventory_index = inventory.index(equipment)
    except ValueError:
        cb_inventory_index = -1 
        print(f"no equipment {equipment} found")
    if (cb_inventory_index != -1):
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip tool
        print(f"found equipment {equipment}")

    events,_,_,_ = env.step([0,0,0,12,12,0,0,0]) #get event
    recipy = MC.ALL_CRAFT_SMELT_ITEMS
    item_recipy_index = recipy.index(item)

    if (not use_crafting_table):
        if use_furnace:
            if (events['location_stats']['pos'][1]<=56):
                mine_ahead(env,)
            else:
                move_to_middle(env)
                events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                
                if not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air"):
                # if not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius-1]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius+1]=="air" and events['voxels']['block_name'][vradius+1][vradius][vradius+1]=="air" and   events['voxels']['block_name'][vradius+1][vradius][vradius-1]=="air"):
                    explore_numb = 0
                    mine_ahead(env,)
                    events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                    while (explore_numb<20 and  (not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and 
                                                      events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and 
                                                      events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" and 
                                                      events['voxels']['block_name'][vradius+1][vradius+1][vradius-1]=="air" and 
                                                      events['voxels']['block_name'][vradius+1][vradius+1][vradius+1]=="air" and 
                                                      events['voxels']['block_name'][vradius+1][vradius][vradius+1]=="air" and   
                                                      events['voxels']['block_name'][vradius+1][vradius][vradius-1]=="air"))):
                        print(f"explore_numb is {explore_numb}, place not right for crafting")
                        explore_above_ground_none(env,"nothing",0,3)
                        explore_numb += 1
                        events = sleep(env)
                    while not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" ) and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" ):
                        mine_ahead(env,)
                        move_one_block(env,3,0,0)
                        move_one_block(env,3,0,0)
                        move_to_middle(env)
                        print(f"action_craft: taking a step back")
                        events = sleep(env)
                print(f"action_craft: front floor{events['voxels']['block_name'][vradius+1][vradius-1][vradius]}\n block in front of body is {events['voxels']['block_name'][vradius+1][vradius][vradius]} \n and block in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}\n begin crafting {item}")
                mine_ahead(env,)
                move_to_middle(env)
            cb_inventory_index = events['inventory']['name'].tolist().index('furnace')
            # cb_inventory_index = events['inventory']['name'].tolist().index('crafting table')
            
            events,_,_,_ = env.step([0,0,0,16,12,0,0,0])
            events = sleep(env)
            events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip furnace
            events = sleep(env)
            while events['inventory']['name'].tolist()[0]=='furnace':
                events,_,_,_ = env.step([0,0,0,12,12,6,0,0]) #backward until place furnace
                events,_,_,_ = env.step([2,0,0,12,12,0,0,0])
            
            cb_inventory_index = events['inventory']['name'].tolist().index('coal') #equip coal
            events,_,_,_ = env.step([0,0,0,12,12,1,0,0]) #use furnace
            for i in range(craft_num):
                events,_,_,_ = env.step([0,0,0,12,12,4,item_recipy_index,0]) #craft item by craft_num
            events = sleep(env)

            cb_inventory_index = events['inventory']['name'].tolist().index('stone pickaxe')  #################for debug, could be changed
            events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip stone pickaxe

            for i in range(10):# may have to modify
                events,_,_,_ = env.step([0,0,0,12,12,3,0,0]) #attack 8 times to get furnace
            events,_,_,_ = env.step([0,0,0,8,12,0,0,0]) #look forward 
            events = sleep(env)
            print(events['inventory']['name'].tolist())
            print('furnace' in events['inventory']['name'].tolist())

            if 'furnace' not in events['inventory']['name'].tolist():
                for i in range(10):
                    events,_,_,_ = env.step([1,0,0,12,12,0,0,0]) #get furnace
                for i in range(10):
                    events,_,_,_ = env.step([2,0,0,12,12,0,0,0]) #go back
        else:
            for i in range(craft_num):
                events,_,_,_ = env.step([0,0,0,12,12,4,item_recipy_index,0])
                events = sleep(env)
    else:
        if (events['location_stats']['pos'][1]<=56):
            mine_ahead(env,)
        else:
            move_to_middle(env)
            events,_,_,_ = env.step([0,0,0,12,12,0,0,0])

            print(events['voxels']['block_name'][vradius+1][vradius-1][vradius])
            print(events['voxels']['block_name'][vradius+1][vradius][vradius])
            if not (events['voxels']['block_name'][vradius+1][vradius][vradius]!=("water") and events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air"):
            # if not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius-1]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius+1]=="air" and events['voxels']['block_name'][vradius+1][vradius][vradius+1]=="air" and   events['voxels']['block_name'][vradius+1][vradius][vradius-1]=="air"):
                explore_numb = 0
                mine_ahead(env,)
                events,_,_,_ = env.step([0,0,0,12,12,0,0,0])
                while (explore_numb<20 and  (not (events['voxels']['block_name'][vradius+1][vradius][vradius]!=("water") and
                                                  events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" or "water") and 
                                                  events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and 
                                                  events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" and 
                                                  events['voxels']['block_name'][vradius+1][vradius+1][vradius-1]=="air" and 
                                                  events['voxels']['block_name'][vradius+1][vradius+1][vradius+1]=="air" and 
                                                  events['voxels']['block_name'][vradius+1][vradius][vradius+1]=="air" and   
                                                  events['voxels']['block_name'][vradius+1][vradius][vradius-1]=="air"))):
                    print(f"explore_numb is {explore_numb}, place not right for crafting")
                    explore_above_ground_none(env,"nothing",0,3)
                    explore_numb += 1
                    events = sleep(env)
                    print(events['voxels']['block_name'][vradius+1][vradius-1][vradius])

                while not (events['voxels']['block_name'][vradius+1][vradius-1][vradius]!=("air" ) and events['voxels']['block_name'][vradius+1][vradius][vradius]=="air" and events['voxels']['block_name'][vradius+1][vradius+1][vradius]=="air" ):
                    mine_ahead(env,)
                    move_one_block(env,3,0,0)
                    move_one_block(env,3,0,0)
                    move_to_middle(env)
                    print(f"action_craft: taking a step back")
                    events = sleep(env)
            print(f"action_craft: front floor{events['voxels']['block_name'][vradius+1][vradius-1][vradius]}\n block in front of body is {events['voxels']['block_name'][vradius+1][vradius][vradius]} \n and block in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}\n begin crafting {item}")
            mine_ahead(env,)
            move_to_middle(env)
        
        inventory = events['inventory']['name'].tolist()
        cb_inventory_index = inventory.index('crafting table')
        events,_,_,_ = env.step([0,0,0,16,12,0,0,0])
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip crafting tabsle
        # sleep(env)
        # while events['inventory']['name'].tolist()[0]=='crafting table':
        #     events,_,_,_ = env.step([0,0,0,12,12,6,0,0]) #place crafting table
        #     events,_,_,_ = env.step([2,0,0,12,12,0,0,0]) 
        events = sleep(env)
        events,_,_,_ = env.step([0,0,0,12,12,1,0,0]) #use
        for i in range(craft_num):
            events,_,_,_ = env.step([0,0,0,12,12,4,item_recipy_index,0]) #craft item
        events = sleep(env)

        cb_inventory_index = events['inventory']['name'].tolist().index('dirt')  #################for debug, could be changed
        events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip stone pickaxe
        
        for i in range(5):# may have to adjust
            events,_,_,_ = env.step([0,0,0,12,12,3,0,0]) #attack 5 times to get crafting table
        events,_,_,_ = env.step([0,0,0,8,12,0,0,0])
        events = sleep(env)

        print(events['inventory']['name'].tolist())
        print('crafting table' in events['inventory']['name'].tolist())

        if 'crafting table' not in events['inventory']['name'].tolist():
            for i in range(10):
                events,_,_,_ = env.step([1,0,0,12,12,0,0,0]) #get crafting table
            for i in range(10):
                events,_,_,_ = env.step([2,0,0,12,12,0,0,0]) #go back
    # print(events['inventory']['name'].tolist())
    name = events['inventory']['name'].tolist()
    num  = events['inventory']['quantity'].tolist()


    return name, num

def go_up(env, y_level):
    events,_,_,_ = env.step([0,0,0,12,12,0,0,0]) 
    cb_inventory_index = events['inventory']['name'].tolist().index('dirt')  #################for debug, could be changed
    events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip cobblestone

    for i in range(6):
        turn_up(env,1)
    curlevel = events['location_stats']['pos'][1]
    while curlevel<y_level:
        print(curlevel)
        for i in range(10):
                events,_,_,_ = env.step([0,0,0,12,12,3,0,0]) #attack 5 times

        if  events['inventory']['name'].tolist()[0]!='dirt':
            cb_inventory_index = events['inventory']['name'].tolist().index('dirt')  #################for debug, could be changed
            events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip cobblestone

        for i in range(12):
            turn_down(env,1)
        events,_,_,_ = env.step([0,0,1,12,12,0,0,0])
        for i in range(4):
            if  events['inventory']['name'].tolist()[0]=='dirt':
                events,_,_,_ = env.step([0,0,0,12,12,6,0,0])
        for i in range(12):
            turn_up(env,1)
        curlevel = events['location_stats']['pos'][1]

    for i in range(6):
        turn_down(env,1)

# Helper function.
def turn_up(env, angle):
    events,_,_,_ = env.step([0,0,0,12-angle,12,0,0,0])
    return events

# Helper function.
def turn_down(env, angle):
    events,_,_,_ = env.step([0,0,0,12+angle,12,0,0,0])
    return events

# the function name explore_above_ground may be misleading, it is actually a function
# which can be used both in above-ground and underground scenarios,
# set parameter underground to 0 if above ground, set it to 1 if underground.
# once the agent has explored max_try_steps number of steps, it will stop regardless of whether object is within range.
def explore_above_ground_none(env,object,underground,max_try_steps=10000):
    global explore_steps
    # events  = sleep(env)
    global stuck
    global direction
    global prev_position
    global retry_times
    global dontstop
    events = sleep(env)
    print(f"exploring once, front floor{events['voxels']['block_name'][vradius+1][vradius-1][vradius]}\n block in front of body is {events['voxels']['block_name'][vradius+1][vradius][vradius]} \n and block in front of head is {events['voxels']['block_name'][vradius+1][vradius+1][vradius]}\n ")
    
    # # make sure you drop tool into inventory before exploring so as not to waste them
    # inventory = events['inventory']['name'].tolist()
    # cb_inventory_index = inventory.index('air')
    # events,_,_,_ = env.step([0,0,0,16,12,0,0,0])
    # events = sleep(env)
    # events,_,_,_ = env.step([0,0,0,12,12,5,0,cb_inventory_index]) #equip crafting tabsle
    if (not underground):
        for i in range(max_try_steps):
            if i >= 2 and dontstop == 1:
                dontstop = 0
            # create_observation(env,f"observation_{i}")
            print(f"try step is {i} and dir is {direction}")
            print(f"explore step is {explore_steps} and position is {events['location_stats']['pos']}")

            # save_rgb_as_image(env,f"{i}")

            # if lidar_detect(env,object) and (i!=0 and dontstop == 0):
            # # if voxel_detect(env,object) and (i!=0 and dontstop == 0):
            #     print("Found object {object}!!!yay")
            #     return True
            
            if explore_steps >= 10000 :
                print("explore steps exceed limit")
                explore_steps = 0
                return False
            if (prev_position[0] != events['location_stats']['pos'][0]) or (prev_position[2] != events['location_stats']['pos'][2]):
                stuck = 0
            else:
                stuck += 1
            prev_position = events['location_stats']['pos']
    
            if stuck>2:
                print(f"stuck!!!")
                stuck = 0
                observation = ""
                for i in range(events['voxels']['block_name'].shape[0]):
                    for j in range(events['voxels']['block_name'].shape[1]):
                        for k in range(events['voxels']['block_name'].shape[2]):
                            # Convert the element to a string and append it to 'observation'
                            observation += str(events['voxels']['block_name'][i, j, k]) + " "
                        observation+='\n'
                    observation +='\n'
                # print(f"stuck observation:{observation}")
                mined_ahead = 0
                
                while (True):# an issue here
                    retry_times += 1
                    if (retry_times > 2):
                        observation = ""
                        for i in range(events['voxels']['block_name'].shape[0]):
                            for j in range(events['voxels']['block_name'].shape[1]):
                                for k in range(events['voxels']['block_name'].shape[2]):
                                    # Convert the element to a string and append it to 'observation'
                                    observation += str(events['voxels']['block_name'][i, j, k]) + " "
                                observation+='\n'
                            observation +='\n'
                        print(f"had to mine ahead")
                        move_to_middle(env)
                        mine_ahead_aboveground(env)
                        mined_ahead = 1
                        stuck = 0
                        direction = 0
                        retry_times = 0
                        break
                    print(f"retry times is {retry_times}")

                    while(action_stack and action_stack[-1][0]==2 and mined_ahead == 0):#action's movedir was "go right", which means that agent had no choice but go right
                        action_tuple = action_stack.pop()
                        if (action_tuple[0] == 2):
                            if not (try_leftward(env,underground)):
                                print(f"had to mine ahead because can't go left")
                                move_to_middle(env)
                                mine_ahead_aboveground(env)
                                mined_ahead = 1
                                direction = 0
                                retry_times = 0
                                stuck = 0
                                break
                        
                        elif (action_tuple[0] == 0):
                            if (not try_backward(env,underground)):
                                print(f"had to mine ahead because can't go back")
                                move_to_middle(env)
                                mine_ahead_aboveground(env)
                                mined_ahead = 1
                                direction = 0
                                retry_times = 0
                                stuck = 0
                                break
                            retry_times = 0
                            stuck = 0
                        # move_one_block(env,3-action_tuple[0],0,1-action_tuple[1])
                    if (not action_stack ) and (not mined_ahead):
                        print("Exception encountered, may be stuck permanently!! but i will venture a step forward")# tbd: think of some other way to let agent extricate himself
                        move_to_middle(env)
                        mine_ahead_aboveground(env)
                        direction = 0
                        return
                    elif (not mined_ahead):
                        action_tuple = action_stack.pop()
                        # print(f"action_tuple's direction is {action_tuple[0]}")
                        move_one_block(env,3,0,1-action_tuple[1])
                        stuck = 0
                        if (try_rightward(env,underground,0)):
                            print(f"moved rightward in new endeavor and position now is {events['location_stats']['pos']}")
                            direction = 0
                            break
                if (not action_stack):
                    print("Exception encountered, stuck permanently!!")# tbd: think of some other way to let agent extricate himself
                    return
            if direction == 0:
                if (not try_forward(env,underground)):
                    print(f"meant to go forward, rightward instead")
                    observation = ""
                    for i in range(events['voxels']['block_name'].shape[0]):
                        for j in range(events['voxels']['block_name'].shape[1]):
                            for k in range(events['voxels']['block_name'].shape[2]):
                                # Convert the element to a string and append it to 'observation'
                                observation += str(events['voxels']['block_name'][i, j, k]) + " "
                            observation+='\n'
                        observation +='\n'
                    print(observation)
                    direction = 2
                    continue
                    # record this position in stack
                    # add another 
                else:
                    print("went forward as planned, want to continue forward")
                    direction = 0
                    continue
            if direction == 1:
                if (not try_leftward(env,underground)):
                    direction = 0
                    continue
                else:
                    direction = 0
                    continue
            if direction == 2:
                if (not try_rightward(env,underground,0)):
                    print(f"meant to go rightward, but can't")
                    observation = ""
                    for i in range(events['voxels']['block_name'].shape[0]):
                        for j in range(events['voxels']['block_name'].shape[1]):
                            for k in range(events['voxels']['block_name'].shape[2]):
                                # Convert the element to a string and append it to 'observation'
                                observation += str(events['voxels']['block_name'][i, j, k]) + " "
                            observation+='\n'
                        observation +='\n'
                    print(observation)
                    direction = 0
                    continue
                else:
                    print("went rightward as planned, want to go forward now")
                    direction = 0
                    continue
        print("end of exploration")
        return False
    else:
        for i in range(max_try_steps):
            # create_observation(env,f"observation_{i}")
            # move_one_block(env,0,1,1)
            print(f"try step is {i} and dir is {direction}")
            print(f"explore step is {explore_steps} and position is {events['location_stats']['pos']}")
            if surrounding_voxel_detect(env, object):
                print("Found object {object}!!!yay")
                return True
            if explore_steps >= 10000:
                print("explore steps exceed limit")
                explore_steps = 0
                return False
            # print(events['voxels']['block_name'])
            move_one_block(env,0,1,1)






    
################################################################################################################################################################################################################

# # Do not modify configurations below rashly. 👇
# # If you want to adjust the lidar configuration, you would also have to modify the mine function.
# biome_string = "forest"
# # biome_string = "plains"
# # biome_string = "desert"
# seed = generate_random_string(env,)
# env = minedojo.make(
#     task_id="harvest",
#     image_size=(512, 820),
#     target_names="diamond",
#     target_quantities=100,
#     seed=3,
#     initial_mobs="sheep",
#     specified_biome = biome_string,
#     initial_mob_spawn_range_low=(-3, 1, -3),
#     initial_mob_spawn_range_high=(3, 3, 3),
#     spawn_rate=1,
#     break_speed_multiplier = 100.0,
#     spawn_range_low=(-10, -10, -10),
#     spawn_range_high=(10, 10, 10),
#     start_at_night = False,
#     world_seed = seed,
#     use_voxel = True,
#     voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),# doesn't really matter
#     use_lidar=True,
#     # task_id="harvest_milk",
#     lidar_rays=[
#             (np.pi * pitch / 180, np.pi * yaw / 180, 10) # ALERT: lidar range is now 10
#             for pitch in np.arange(-60, 60, 5)
#             for yaw in np.arange(-60, 60, 5)
#     ]
# )



# env.reset()
# env.set_inventory([InventoryItem(slot=9, name="dirt", variant=None, quantity=60),])
# # env.set_inventory([InventoryItem(slot=9, name="dirt", variant=None, quantity=60),InventoryItem(slot=1, name="diamond", variant=None, quantity=3),InventoryItem(slot=2, name="crafting_table", variant=None, quantity=1),InventoryItem(slot=5, name="stick", variant=None, quantity=3)])
# # env.set_inventory([InventoryItem(slot="mainhand", name="log", variant=None, quantity=0),InventoryItem(slot=1, name="log", variant=None, quantity=0),])
# # env.set_inventory([InventoryItem(slot="mainhand", name="log", variant=None, quantity=0),InventoryItem(slot=1, name="log", variant=None, quantity=0),InventoryItem(slot=5, name="wooden_pickaxe", variant=None, quantity=1),InventoryItem(slot=6, name="crafting_table", variant=None, quantity=1),InventoryItem(slot=7, name="stick", variant=None, quantity=10)])
# with open ("log.txt",'w') as file2:
#     file2.write(f"This is the world {seed} \n")
# events,reward,ended,addinfo = env.step([0,0,0,12,6,0,0,0])  # Facing north

# print(events['inventory']['name'])
# print(events['inventory']['quantity'])
# print(type(events['location_stats']['pos']))
# direction = 0

# # Do not modify the configurations above rashly. 👆


# # # action list for building wooden pickaxe
# # for i in range(4):
# #     explore_above_ground("wood",0)
# #     approach("wood",0)
# #     mine("wood","")
# #     print(f"print inventory:{events['inventory']['name']}\n Inventory quantity:{events['inventory']['quantity']}")
# #     print(f"{i}th mine finished")
# # for i in range(3):
# #     action_craft(env,"planks",0,0,1)

# # action_craft(env,"crafting_table",0,0,1)
# # action_craft(env,"stick",0,0,1)
# # action_craft(env,"wooden_pickaxe",1,0,1)
# # print(f"print inventory:{events['inventory']['name'].tolist()}\n Inventory quantity:{events['inventory']['quantity']}")
# # sleep(env,3)
# # # print(events)# make sure lidar rays cover desired object
# # print(seed)
# # env.close()

# # building iron pickaxe
# # for i in range(10):
# #     explore_above_ground("wood",0)
# #     approach("wood",0)
# #     mine("wood","")
# #     print(f"print inventory:{events['inventory']['name']}\n Inventory quantity:{events['inventory']['quantity']}")
# #     print(f"{i}th mine finished")


# # explore_above_ground("wood",0)
# # action_mine("Nan","",0,4)
# action_mine("wood","",0,4)
# print(f"print inventory:{events['inventory']['name']}")
# action_craft(env,"planks",0,0,4)
# action_craft(env,"stick",0,0,3)

# action_craft(env,"crafting_table",0,0,1)



# action_craft(env,"wooden_pickaxe",1,0,1)
# go_down_to_y_level(55,"wooden pickaxe")# y-level was 60 originally
# action_mine("stone","wooden pickaxe",1,11)
# action_craft(env,"stone_pickaxe",1,0,1)
# action_craft(env,"furnace",1,0,1)
# go_down_to_y_level(53,"stone pickaxe")

# # explore_above_ground("iron ore",1)
# action_mine("iron ore","stone pickaxe",1,3)

# action_craft(env,"iron_ingot",0,1,3)# craft iron ingots using furnace * 3

# # events = sleep(env)
# # print(f"print inventory:{events['inventory']['name']}")
# # pdb.set_trace()

# action_craft(env,"iron_pickaxe",1,0,1)

# go_down_to_y_level(14,"iron pickaxe")
# # explore_above_ground("diamond ore",1)
# action_mine("diamond ore","iron pickaxe",1,3)
# action_craft(env,"diamond_pickaxe",1,0,1)

# events = sleep(env)
# print(events['inventory']['name'])
# print(events['inventory']['quantity'])

# while(1):
#     pdb.set_trace()
#     sleep(env)

# env.close()



# # only craft with _