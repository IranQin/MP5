from utils import *

class Percipient:
    def __init__(
        self,
        openai_key,
        memory,
        question_model_name="gpt-4-0613",
        answer_method="active",     # active or caption
        answer_model="mllm",        # mllm or gpt-vision
        answer_mllm_url=None,       # mllm url
        answer_gpt_name=None,       # gpt-4-vision-preview
        temperature=0
    ):
        openai.api_base ="https://api.chatweb.plus/v1"
        self.perception_llm = ChatOpenAI(
            model_name=question_model_name,
            temperature=temperature
        )

        self.memory = memory
        self.answer_method = answer_method
        self.answer_model = answer_model

        assert self.answer_method in ["active", "caption"], "Please input correct method of perception"
        assert self.answer_model in ["mllm", "gpt-vision"], "Please input correct model of perception"
        assert self.memory is not None, "Please input memory"

        if self.answer_model == "mllm":
            assert answer_mllm_url is not None, "Please input mllm url"
            self.mllm = MineLLM(answer_mllm_url=answer_mllm_url)

        elif self.answer_model == "gpt-vision":
            assert answer_gpt_name is not None, "Please input gpt vison name"
            self.mllm = ChatOpenAIVision(method=self.answer_method, model_name=answer_gpt_name, openai_key=openai_key)
        else:
            raise ValueError("Percipient's answer mllm is incorrect.")


    def perceive(self, task_information, find_obj, file_path, max_retries=5):
        if self.answer_method == "active":
            return self.activate_perception(task_information=task_information, find_obj=find_obj, file_path=file_path, max_retries=max_retries)
        elif self.answer_method == "caption":
            return self.caption_perception(task_information=task_information, find_obj=find_obj, file_path=file_path, max_retries=max_retries)
        else:
            raise ValueError("Percipient's answer method is incorrect.")


    def get_perception_question(self, task_information, find_obj):
        # Active Perception
        active_perception_system = load_prompt("active_perception_system")
        active_perception_query = load_prompt("active_perception_query").format(
            task_information=task_to_description_prompt(update_find_task_prompt(task_information, find_obj)), 
            current_environment_information=list_dict_to_prompt(self.memory.current_environment_information)
        )

        messages = [
            SystemMessage(content=active_perception_system),
            HumanMessage(content=active_perception_query)
        ]   

        question_info = self.perception_llm(messages).content
        return fix_and_parse_json(question_info)



    def check_perception_question(self, question_dict):
        question_dict["status"] = int(question_dict["status"])
        ## Check if the Active Perception is over
        assert question_dict["status"] in [0, 1, 2]


        ## Success
        if question_dict["status"] == 2:
            return 2

        ## Failure
        if question_dict["status"] == 0:
            return 0

        ## Asked before
        for information in self.memory.current_environment_information:
            question_type = question_dict["query"]["type"]
            if question_type in information["type"] or information["type"] in question_type:
                return 3

        # Continue asking
        return 1
    

    def activate_perception(self, task_information, find_obj, file_path, max_retries=5):

        # Active Perception Failure
        if max_retries == 0:
            log_info("************Failed to get activate perception. Consider updating your prompt.************\n\n")
            return 

        try:
            ## Get the environment information
            while True:
                ## Current Environment State
                log_info(f"Current Environment Information: \n{list_dict_to_prompt(self.memory.current_environment_information)}")

                ## Active Perception
                question_dict = self.get_perception_question(task_information, find_obj)

                ## Check if the Active Perception is over
                check_result = self.check_perception_question(question_dict)
                if check_result == 2:
                    log_info(f"Active Perception Success: {question_dict}")
                    log_info("************Active Perception Finish!************\n")
                    return 2
                elif check_result == 3:
                    log_info(f"************Active Perception Failure: The question about {question_dict['query']['type']} was asked before. Continue finding************\n")
                    return 0
                elif check_result == 0:
                    log_info(f"************Active Perception Failure: {question_dict['thoughts']}***********\n")
                    return 0     
                        
                ## Active Perception Question
                log_info(f"GPT Question: {question_dict['query']['question']}")

                ## Interact with MLLM
                answer = self.mllm.query(question_dict['query']['question'], file_path)

                log_info(f"MLLM Answer: {answer}")

                ## record the information
                self.memory.current_environment_information.append({
                        "type": question_dict['query']['type'],
                        "info": answer
                    })

        except Exception as e:
            log_info(f"Error arises in Active Perception part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.activate_perception(
                task_information=task_information,
                find_obj=find_obj,
                file_path=file_path,
                max_retries=max_retries - 1,
            )



    def check_caption_perception(self, task_information, find_obj):
        # Caption Perception
        check_caption_perception_system = load_prompt("check_caption_perception_system")
        check_caption_perception_query = load_prompt("check_caption_perception_query").format(
            task_information=task_to_description_prompt(update_find_task_prompt(task_information, find_obj)), 
            current_environment_information=list_dict_to_prompt(self.memory.current_environment_information)
        )

        messages = [
            SystemMessage(content=check_caption_perception_system),
            HumanMessage(content=check_caption_perception_query)
        ]   

        question_info = self.perception_llm(messages).content
        return fix_and_parse_json(question_info)

    def caption_perception(self, task_information, find_obj, file_path, max_retries=5):

        # Caption Perception Failure
        if max_retries == 0:
            log_info("************Failed to get activate perception. Consider updating your prompt.************\n\n")
            return

        try:
            ## Interact with MLLM
            answer = self.mllm.query("Could you describe this Minecraft image?", file_path)

            log_info(f"MLLM Answer: {answer}")

            ## record the information
            self.memory.current_environment_information.append({
                    "type": "environment caption",
                    "info": answer
                })
            
            ## Check if finish perception
            check_dict = self.check_caption_perception(task_information, find_obj)
            if check_dict["status"] == 2:
                log_info(f"Active Perception Success: {check_dict}")
                log_info("************Active Perception Finish!************\n")
                return 2
            else:
                log_info(f"************Active Perception Failure: {check_dict['thoughts']}***********\n")
                return 0 

        except Exception as e:
            log_info(f"Error arises in Caption Perception part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.caption_perception(
                task_information=task_information,
                find_obj=find_obj,
                file_path=file_path,
                max_retries=max_retries - 1,
            )