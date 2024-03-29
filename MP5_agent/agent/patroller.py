from utils import *

class Patroller:
    def __init__(
        self,
        openai_key,
        memory,
        model_name="gpt-3.5-turbo-0613",
        temperature=0
    ):
        os.environ["OPENAI_API_KEY"] = openai_key
        openai.api_base ="https://api.chatweb.plus/v1"

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.memory = memory

        assert memory is not None, "Please input memory"


    def check_task_success(self, task_information, max_retries=5):
        check_system = load_prompt("check_system")
        check_query = load_prompt("check_query").format(
            task_information=task_to_description_prompt(task_information), 
            current_environment_information=list_dict_to_prompt(self.memory.current_environment_information,),
            inventory=self.memory.inventory
        )

        messages = [
            SystemMessage(content=check_system),
            HumanMessage(content=check_query)
        ]   

        try:
            check_info = self.llm(messages).content

            check_dict = fix_and_parse_json(check_info)
            assert check_dict["success"] in [True, False]
            if "suggestion" not in check_dict:
                check_dict["suggestion"] = ""

            log_info(f"Check Result: {check_dict}")

            if check_dict["success"]:
                log_info("************Workflow Success!************")
            else:
                log_info("************Workflow Failure!************")

            return check_dict

        except Exception as e:
            log_info(f"Error arises in Checker part: {e} Trying again!\n\n")
            return self.check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            ) 

    
    def check_action_preparation(self, action_name, inventory, args_dict):

        if action_name == "mine" or action_name == "fight" or action_name == "dig_down" or action_name == "dig_up" or action_name == "apply":
            tool = args_dict["tool"]
            if tool:
                if tool not in inventory or inventory[tool] <= 0:
                    check_dict = {
                        "feedback": f"You do not have 1 {tool} as the tool to complete the '{action_name}' action.",
                        "success": False,
                        "suggestion": f"Craft 1 {tool} on a crafting table as the platform first."
                    }
                    return check_dict
            check_dict = {
                        "feedback": f"You have 1 {tool} to complete the '{action_name}' action. Therefore, continue to do this action.",
                        "success": True,
                        "suggestion": f""
                    }
            return check_dict
        elif action_name == "equip":
            obj = args_dict["obj"]
            if obj:
                if obj not in inventory or inventory[obj] <= 0:
                    check_dict = {
                        "feedback": f"You do not have 1 {obj} to complete the '{action_name}' action.",
                        "success": False,
                        "suggestion": f"Craft 1 {obj} on a crafting table as the platform first."
                    }
                    return check_dict
            check_dict = {
                        "feedback": f"You have 1 {obj} to complete the '{action_name}' action.",
                        "success": True,
                        "suggestion": f""
                    }
            return check_dict
        
        elif action_name == "craft":
            platform = args_dict["platform"]
            if platform:
                if platform not in inventory or inventory[platform] <= 0:
                    check_dict = {
                        "feedback": f"You do not have {platform} to complete the '{action_name}' action.",
                        "success": False
                    }
                    if platform.lower().find("crafting") != -1:
                        check_dict["suggestion"] = f"Craft a {platform} using 4 planks. If you do not have enough planks, please craft 4 planks using 1 log first."
                    elif platform.lower().find("furnace") != -1:
                        check_dict["suggestion"] = f"Craft a {platform} using 8 cobblestone. If you do not have enough cobblestone, please mine 8 cobblestone using a wooden pickaxe as the tool, primarily found at level 55 first."
                    return check_dict
            
            materials = args_dict["materials"]

            for material, quantity in materials.items():
                material = update_inventory_obj_name(material)
                quantity = int(quantity)
                if material not in inventory:
                    check_dict = {
                        "feedback": f"You do not have {material} to complete the '{action_name}' action. You need {quantity} {material} but you do not have {material} in your inventory.",
                        "success": False,
                        "suggestion": f"Mine or Craft enough {material} first."
                    }
                    return check_dict
                
                elif inventory[material] < quantity:
                    check_dict = {
                        "feedback": f"You do not have enough {material} to complete the '{action_name}' action. You need {quantity} {material} but you only have {inventory[material]} {material} in your inventory.",
                        "success": False,
                        "suggestion": f"Mine ot Craft enough {material} first."
                    }
                    return check_dict

            check_dict = {
                        "feedback": f"You have enough materials to complete the '{action_name}' action.",
                        "success": True,
                        "suggestion": f""
                    }
            return check_dict

        else:
            # find and move_to
            check_dict = {
                        "feedback": f"",
                        "success": True,
                        "suggestion": f""
                    }
            return check_dict
    

