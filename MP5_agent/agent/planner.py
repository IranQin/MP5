from utils import *

class Planner:
    def __init__(
        self,
        openai_key,
        memory,
        model_name="gpt-4-0613",
        temperature=0
    ):
        os.environ["OPENAI_API_KEY"] = openai_key

        openai.api_base ="https://api.chatweb.plus/v1"

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )

        self.memory = memory
        assert  self.memory is not None, "Please input memory"
    
    def get_workflow(self, task_information, underground, check_result, max_retries=5):

        if max_retries == 0:
            log_info("************Failed to get workflow. Consider updating your prompt.************\n\n")
            return {}

        try:
            structured_action_system = load_prompt("structured_action_system")
            task_information_string = dict_to_prompt(task_information)
            current_environment_information=""
            
            if underground:
                current_environment_information += "- position: underground\n"
            else:
                current_environment_information += "- position: ground\n"

            reference_plan_string = list_dict_to_prompt(self.memory.seach_workflows(task_information["description"]))

            structured_action_query = load_prompt("structured_action_query").format(
                task_information=task_information_string, 
                current_environment_information=current_environment_information,
                inventory=self.memory.inventory,
                reference_plan=reference_plan_string
            )


            if len(check_result) == 0:
                structured_action_query += "\nPlan your workflow. Remember to follow the response format."
            else:
                structured_action_query += f"""The previous workflow failed. 
                The reason for the failure: {check_result["feedback"]}.
                A suggested recommendations: {check_result["suggestion"]}. 
                re-plan your workflow. Remember to follow the response format."""
                
            messages = [
                SystemMessage(content=structured_action_system),
                HumanMessage(content=structured_action_query)
            ]

            workflow_dict = self.llm(messages).content
            log_info(f"Create Workflow Result: {workflow_dict}")

            return fix_and_parse_json(workflow_dict)
        except Exception as e:
            log_info(f"Error arises in Plan Workflow part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.get_workflow(
                task_information, 
                underground, 
                check_result, 
                max_retries=max_retries - 1
            )