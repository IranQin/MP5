from utils import *
from planner import Planner
from patroller import Patroller
from memory import Memory
from performer import Performer
from percipient import Percipient
from minedojo.sim import InventoryItem
import minedojo
import random
import argparse
import numpy as np

def main():
    mllm_url = args.mllm_url
    openai_key = args.openai_key


    f_mkdir(f"../images"); f_remove("../video"); f_mkdir(f"../video")
    f_mkdir(f"../logs"); logging.basicConfig(filename=f'../logs/agent.log', filemode='w', level=logging.INFO, format='%(message)s')

    
    seed = random.randint(1,1000000000000)
    vradius = 5
    log_info(seed)
    biome_string = "forest"

    env = minedojo.make(
        task_id="harvest", target_names="diamond",
        image_size=(512, 820), 
        target_quantities=100, seed=3, 
        specified_biome = biome_string, 
        spawn_rate=1, 
        break_speed_multiplier = 100.0, 
        spawn_range_low=(-10, -10, -10), spawn_range_high=(10, 10, 10), 
        start_at_night = False, world_seed = seed, use_voxel = True, 
        voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius), # doesn't really matter
        use_lidar=True,
        lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 10) # ALERT: lidar range is now 10
                for pitch in np.arange(-60, 60, 5)
                for yaw in np.arange(-60, 60, 5)
        ]
    )


    env.reset()

    env.set_inventory([InventoryItem(slot=9, name="dirt", variant=None, quantity=6), InventoryItem(slot=16, name="coal", variant=None, quantity=3), InventoryItem(slot=17, name="coal", variant=None, quantity=3)])
    events, _, _, _ = env.step([0,0,0,12,6,0,0,0])


    underground = False
    model_name=args.gpt_model_name

    memory = Memory(openai_key=openai_key, use_history_workflow=False)
    patroller = Patroller(openai_key=openai_key, memory=memory, model_name=model_name)
    planner = Planner(openai_key=openai_key, memory=memory, model_name=model_name)
    
    # answer_method = active | caption
    # answer_model = mllm -> answer_mllm_url | gpt-vision -> gpt-4-vision-preview
    percipient = Percipient(openai_key=openai_key, memory=memory, question_model_name=model_name,
                            answer_method=args.answer_method, answer_model=args.answer_model, 
                            answer_mllm_url=mllm_url, answer_gpt_name="gpt-4-vision-preview")

    # sync the memory
    share_memory(memory=memory, events=events)

    with open(args.task, 'r') as f:
        task_list = json.load(f)
        for task_id, task_information in enumerate(task_list[::-1]):

            every_task_max_retries = 300
            check_result = {}

            log_info(f"Task: { task_information['description'] }")

            while every_task_max_retries >= 0:
                memory.reset_current_environment_information()

                log_info(f"My inventory: {memory.inventory}")

                if every_task_max_retries == 0:
                    log_info("************Failed to complete this task. Consider updating your prompt.************\n\n")
                    break

                ## Stage1: Workflow Decision
                workflow_dict = planner.get_workflow(task_information=task_information, underground=underground, check_result=check_result)

                ## Stage2: Interface & Update Inventory
                performer = Performer(memory=memory, percipient=percipient, checker=patroller)
                check_result, underground = performer.check_and_execute_workflow(env=env, workflow_dict=workflow_dict, task_information=task_information, underground=underground)
                # Fail halfway through
                if not check_result["success"]:
                    log_info(f"Action Preparation Failure: {check_result}")
                    every_task_max_retries -= 1
                    continue

                # Stage3: Check the final task result
                check_result = patroller.check_task_success(task_information=task_information)

                ## Stage4: Validation
                if check_result["success"]:
                    # Success: Put successful Workflow into Memory
                    memory.add_successful_workflow(task_information["task"], task_information["description"], workflow_dict["workflow"])
                    break
                else:
                    # Failure: Do not have sufficient materials, Feedback
                    every_task_max_retries -= 1
                    continue
    
    memory.reset_all()
    log_info("############ Successfully Finish All Tasks ############")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 1. mllm_url
    parser.add_argument(
        "--mllm_url",
        type=str,
        default='',
    )
    # 2. openai_key
    parser.add_argument(
        "--openai_key",
        type=str,
        required=True,
    )
    # 3. gpt_model_name
    parser.add_argument(
        "--gpt_model_name",
        type=str,
        required=True,
    )
    # 4. answer_method
    parser.add_argument(
        "--answer_method",
        type=str,
        default='active',
    )
    # 5. answer_model
    parser.add_argument(
        "--answer_model",
        type=str,
        default='mllm',
    )
    # 5. task
    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main()