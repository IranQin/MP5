from utils import *

class Memory:
    
    def __init__(
        self,
        openai_key,
        model_name="gpt-3.5-turbo-0613",
        ckpt_dir="./",
        ckpt_id=0,
        use_history_workflow=False,
        retrieval_top_k=2,
        temperature=0
    ):
        ## Long Memory
        self.inventory = {}
        ## Short Memory
        self.current_environment_information = []
        self.feedback = []

        os.environ["OPENAI_API_KEY"] = openai_key
        openai.api_base ="https://api.chatweb.plus/v1"
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )

        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.ckpt_id = ckpt_id


        f_mkdir(f"{self.ckpt_dir}/memory/workflow_vectordb_{self.ckpt_id}")
        self.workflow_vectordb = Chroma(
            collection_name=f"workflow_vectordb_{self.ckpt_id}",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/memory/workflow_vectordb_{self.ckpt_id}",
        )

        
        if use_history_workflow:
            log_info(f"***********Loading Workflow from {ckpt_dir}memory***********")
            self.workflows = load_json(f"{ckpt_dir}/memory/workflows_{self.ckpt_id}.json")

            for task_name, value in self.workflows.items():
                self.add_successful_workflow(task_name, value["task_description"], value["successful_workflow"], update_json=False)
        else:
            self.workflows = {}
        
        assert self.workflow_vectordb._collection.count() == len(self.workflows), (
            f"workflows's vectordb is not synced with workflows_{self.ckpt_id}.json.\n"
            f"There are {self.workflow_vectordb._collection.count()} workflows in vectordb but {len(self.workflows)} workflows in workflows_{self.ckpt_id}.json.\n"
            f"Did you set use_history_workflow=False when initializing the manager?\n"
            f"You may need to manually delete the workflow_vectordb directory for running from scratch."
        )
    

    def add_successful_workflow(self, task_name, task_description, successful_workflow, update_json=True):

        if task_name in self.workflows:
            self.workflow_vectordb._collection.delete(ids=[task_name])

        self.workflow_vectordb.add_texts(
            texts=[task_description],
            ids=[task_name],
            metadatas=[{"task_name": task_name}],
        )

        if update_json:
            self.workflows[task_name] = {
                "task_description": task_description,
                "successful_workflow": successful_workflow,
            }

            assert self.workflow_vectordb._collection.count() == len(
                self.workflows
            ), f"workflow_vectordb is not synced with workflows_{self.ckpt_id}.json"

            dump_json(self.workflows, f"{self.ckpt_dir}/memory/workflows_{self.ckpt_id}.json")

        self.workflow_vectordb.persist()

        log_info(f"Adding a successful workflow about '{task_description}' to the memory\n\n")

        return True

    def seach_workflows(self, query):

        k = min(self.workflow_vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        log_info(f"Workflow Memory retrieving for {k} workflows")
        docs_and_scores = self.workflow_vectordb.similarity_search_with_score(query, k=k)
        log_info(
            f"Workflow Memory is seaching workflows: "
            f"{', '.join([doc.metadata['task_name'] for doc, _ in docs_and_scores])}"
        )
        
        workflows = []
        for doc, _ in docs_and_scores:
            workflows.append(
                {
                    "task_description": self.workflows[doc.metadata["task_name"]]["task_description"],
                    "workflow": self.workflows[doc.metadata["task_name"]]["successful_workflow"]
                }
            )
        return workflows

    def get_all_successful_workflows(self):
        return self.workflow_vectordb.get()

    def update_inventory(self, new_inventory):
        self.inventory = new_inventory

    def update_workflows(self, new_workflows):
        self.workflows = new_workflows

    def update_current_environment_information(self, new_info):
        self.current_environment_information.append(new_info)

    def update_feedback(self, new_feedback):
        self.feedback.append(new_feedback)

    def update_all(self, new_inventory, new_workflows, new_info, new_feedback):
        self.update_inventory(new_inventory)
        self.update_workflows(new_workflows)
        self.update_current_environment_information(new_info)
        self.update_feedback(new_feedback)

    def reset_inventory(self):
        self.inventory = {}

    def reset_workflows(self):
        self.workflows = {}

    def reset_current_environment_information(self):
        self.current_environment_information = []

    def reset_feedback(self):
        self.feedback = []

    def reset_all(self):
        self.reset_inventory()
        self.reset_workflows()
        self.reset_current_environment_information()
        self.reset_feedback()
        # Clear Successful Workflow Memory in Chroma
        self.workflow_vectordb.delete_collection()
