from utils import *

openai_key = "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"

os.environ["OPENAI_API_KEY"] = openai_key

llm = ChatOpenAI(
            model_name="gpt-4-0613",
            temperature=0
        )


def transform_dict(new_dict):

    # Active Perception
    create_task_description_system = load_prompt("create_task_description_system")
    create_task_description_query = load_prompt("create_task_description_query").format(
        task_information=dict_to_prompt(new_dict), 
    )

    messages = [
        SystemMessage(content=create_task_description_system),
        HumanMessage(content=create_task_description_query)
    ]

    new_dict["description"] = llm(messages).content
    print(new_dict["description"])

    return new_dict

def convert_json_files(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        old_dicts = json.load(input_file)

    new_dicts = [transform_dict(old_dict) for old_dict in old_dicts]

    with open(output_file_path, 'w') as output_file:
        json.dump(new_dicts, output_file)


input_file_path = 'tasks/blocks.json' 
output_file_path = 'tasks/task_blocks.json'

convert_json_files(input_file_path, output_file_path)

