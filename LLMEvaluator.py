from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.utils.math import cosine_similarity
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import BasePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.translate import meteor_score as nltk_meteor
from HF_download_utils import set_hf_cache
from pydantic import BaseModel, Field
import os, sys, json, ast
from typing import *
import re, math, time
import rich

# custom HF download path for bert score embeddings
set_hf_cache("./temp") 
from datetime import datetime
from transformers import logging
logging.set_verbosity_error()
from bert_score import score as bert_score

from config import init_LLM, init_embedding, init_VecDB
from CLI_Format import CLI_input, CLI_next, CLI_print
from LogWriter import LogWriter, remove_special_symbol
from DatabaseManager import VDBManager
import MainGraph as MG
import BaselineGraph as BG
from PromptTools import SystemPromptConfig, to_text

embed = init_embedding(normalize_embeddings=True)

def embed_str(input_str: str) -> List[float]:
    ''' Embed string '''
    assert isinstance(input_str, str), "Input is not string"
    return embed.embed_query(input_str)

def load_test_qa(file:str, q_key:str="question", a_key:str="answer") -> List[Optional[Dict[Literal["question", "answer", "id"], str | int]]]:
    ''' 
    Read Question and Answer pair for RAG testing / Agent Reply from a text file.
    Each line in file should contain Json, Dict object like string
    return : [{ "question" : "foo1", "answer" : "foo2", "id" : "pre given ids or line in file" }, ...]
    '''
    assert os.path.exists(file), f"File not exist : {file}"
    file = os.path.abspath(file)

    qa_pairs = []
    line_loc = 0
    with open(file, mode='r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            line_loc += 1
            qa_item = None

            # check line
            if line == "": # EOF
                break  
            line = line.strip()
            if not line: # skip empty
                continue 

            # extract data
            try:
                qa_item = json.loads(line)
            except json.JSONDecodeError:
                pass
            
            if qa_item is None:
                try:
                    qa_item = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    rich.print(f"Invalid Data at {line_loc} : {line}")
                    continue
            
            if isinstance(qa_item, dict):
                if "id" not in qa_item:
                    qa_item.update({"id" : str(line_loc)}) # for checking raw data
                if q_key in qa_item and a_key in qa_item:
                    qa_pairs.append(qa_item)
                    continue
            
            rich.print(f"Invalid Data at {line_loc} : {line}")
            continue

    rich.print(f"Loaded QA {len(qa_pairs)} items.")
    return qa_pairs

def load_prediction(file:str) -> List[Optional[Dict[Literal["question", "reference", "id", "prediction"], str | int]]]:
    ''' 
    Load pre-generate prediction ans test case from a text file.
    Each line in file should contain JSON string
    return : [{ "question" : "foo1", "reference" : "foo2", "id" : "pre given ids", "prediction" : "llm generated answer" }, ...]
    '''
    assert os.path.exists(file), f"File not exist : {file}"
    file = os.path.abspath(file)

    expected_keys = ("question", "reference", "id", "prediction")

    test_case = []
    line_loc = 0
    with open(file, mode='r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            line_loc += 1
            new_test_case = None

            # check line
            if line == "": # EOF
                break  
            line = line.strip()
            if not line: # skip empty
                continue 

            # extract data
            try:
                new_test_case = json.loads(line)
            except json.JSONDecodeError:
                pass
            
            if new_test_case is None:
                try:
                    new_test_case = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    rich.print(f"Invalid Data at {line_loc} : {line}")
                    continue
            
            if isinstance(new_test_case, dict) and all(k in new_test_case for k in expected_keys):
                test_case.append(new_test_case)
                continue
            
            rich.print(f"Invalid Data at {line_loc} : {line}")
            continue

    rich.print(f"Loaded pre-generate predictions {len(test_case)} items.")
    return test_case

class TestCaseLoader:
    def __init__(self, test_name:Optional[str]=None, qa_file:Optional[str]=None):
        self.test_name = test_name
        self.test_items = load_test_qa(qa_file, q_key="question", a_key="answer") if qa_file else []
        pass

    def __sizeof__(self):
        return len(self.test_items)

    def load(self, qa_file:str):
        ''' load test cases from txt files, append new to existed test cases '''
        self.test_items.extend(load_test_qa(qa_file, q_key="question", a_key="answer"))
    
    def clear(self):
        ''' clean all test item '''
        self.test_items.clear()

    def get(self) -> Generator[Dict[Literal["question", "answer", "id"], int], None, None]:
        ''' Get test item '''
        for tc in self.test_items:
            yield tc

class EvaluationInput(BaseModel):
    ''' Defines Evaluation input, test graph need to have same parameter in Start State '''
    input: str
    prompt_config: Dict

class GraphEvaluator:
    def __init__(self, graph: CompiledStateGraph, prompt_config: Optional[Dict | type[BaseModel]]=None):
        ''' 
        Execute all kinds of test for Compiled Graph,
        - StartState should be one input and one invoke_config
            - input and invoke_config will assamble as input prompt
            - example : {"input" : quesiton string, "invoke_config" : Dict}
        - invoke_config : sets optional variable slot in Start State  
        '''
        assert graph != None, "Please set a valid LangChain Graph"
        self.graph: CompiledStateGraph = graph
        self.prompt_config = prompt_config.model_dump() if isinstance(prompt_config, BaseModel) else prompt_config
        # check if config is in corrent format
        test_evaluation_input = EvaluationInput(input="Test Question", prompt_config=self.prompt_config)
        #
        self.score_board = {"test_amount" : 0, "correct": 0, "incorrect": 0}
        self.incorrect_items = []
        self.correct_items = []
        self.graph_name = graph.get_name()
    
    def get_response(self, question:str) -> str:
        ''' get question response from graph '''
        evaluation_input = EvaluationInput(input=question, prompt_config=self.prompt_config)
        response = self.graph.invoke(evaluation_input)
        response = response.get("output_msg", "EMPTY RESPONSE")
        return to_text(response)
    
    def __split_sections(self, artical:str, expected_title:List[str]):
        ''' split artical into sections between titles, return a dict with title as key, section as value'''
        result = {}
        if not artical or not isinstance(artical, str):
            return result
        pattern = rf"({'|'.join(map(re.escape, expected_title))})\s*:\s*"
        parts = re.split(pattern, artical)
        for i in range(1, len(parts), 2):
            key = parts[i].strip()
            value = parts[i+1].strip() if i+1 < len(parts) else ""
            result[key] = value
        return result

    def answer_relevancy_eval(self, reference:str, prediction:Optional[str]=None) -> Dict[Literal["score"], int]:
        ''' Calculate similarity between llm predicttion and expected answer, checks realitive '''
        prediction = prediction or "EMPTY PREDICTION"
        er, ep = embed_str(reference), embed_str(prediction)
        similarity = cosine_similarity([er], [ep])
        return {"score": round(similarity.item(), 4)}
    
    def meteor_eval(self, reference:str, prediction:Optional[str]=None) -> Dict[Literal["score"], int]:
        ''' Calculate METEOR score between reference and prediction '''
        if prediction is None or len(prediction) == 0:
            return {"score" : 0} # Empty string
        tokenize_r = [ s for s in reference.split(" ")]
        tokenize_p = [ s for s in prediction.split(" ")]
        score = round(nltk_meteor.meteor_score(references=[tokenize_r], hypothesis=tokenize_p), 4)
        return {"score" : score}

    def llm_eval(self, judge_llm: BaseChatModel, question:str, reference:str, prediction:Optional[str]=None) -> Dict[Literal["value", "score", "completion", "reasoning sections"], int | str]:
        ''' 
        Eval a single QA result by judge_llm using langchian evalrator.
        - reference : expected answer
        - prediction : llm output response
        Five key section generated by judge_llm.
        - key sections : 
            - QUESTION
            - CONTEXT
            - STUDENT ANSWER
            - EXPLANATION
            - GRADE 
        '''
        if judge_llm is None:
            rich.print("Skip LLM Evaluation (empty judge llm)")
            return {"reasoning" : "SKIP", "value" : None, "score" : None}
        assert judge_llm != None, "judge_llm is not set"
        # llm grading
        prediction = prediction or "EMPTY PREDICTION"
        evaluator = load_evaluator(EvaluatorType.COT_QA, llm=judge_llm) # LLM QA evaluation with reason
        result = evaluator.evaluate_strings(input=question, prediction=prediction, reference=reference) # expected reasoning, value, score in Dict
        assert all(key in ("reasoning", "value", "score") for key in result) , f"Dict item missing : {str(result)}"
        # generate result
        result["value"] = "INCORRECT" if result["value"] is None else result["value"] # extracted from GRADE
        result["score"] = 0 if result["score"] is None else result["score"]
        key_sections = ("QUESTION", "CONTEXT", "STUDENT ANSWER", "EXPLANATION", "GRADE")
        complete_section = [key for key in key_sections if key in result["reasoning"]] # check if all sections are successfully generated
        result["completion"] = len(complete_section) / len(key_sections)
        result["reasoning sections"] = self.__split_sections(result["reasoning"], complete_section)
        # trimm reason
        return result
    
    def BERTScore_eval(self, reference:str, prediction:Optional[str]=None, lang:Optional[str]="en", model_type:Optional[str]=None) ->  Dict[Literal["precision", "recall", "f1"], float]:
        ''' Calculate BERT Score between reference and prediction '''
        pred = prediction or "EMPTY PREDICTION"
        cands = [pred]
        refs  = [reference]
        #
        kwargs = { "rescale_with_baseline": True }
        if lang:
            kwargs.update({"lang" : lang})
        if model_type:
            kwargs.update({"lang" : None, "model_type" : model_type})
        #
        P, R, F1 = bert_score(cands=cands, refs=refs, **kwargs)
        result = {"precision" : round(P.item(), 4), "recall" : round(R.item(), 4), "f1" : round(F1.item(), 4)}
        return  result
    
    def groundedness_eval(self, reference:str, claims:Optional[List[str]]=None, similarity_threshold:float=0.75) -> Dict[Literal["score", "support", "unsupport"], int | str]:
        ''' Calcualte similuarity between claims and reference  '''
        er = embed_str(reference)
        support   = []
        unsupport = []
        for c in claims:
            ec = embed_str(c)
            similarity = cosine_similarity([er], [ec])
            if similarity >= similarity_threshold:
                support.append(c)
            else:
                unsupport.append(c)
        score = round(len(support) / max(1, len(claims)), 4)
        return {"score" : score, "support" : support, "unsupport": unsupport}
    
    def generate_prediction(self, test_case: TestCaseLoader, save_path:Optional[str]=None) -> str:
        ''' Generate llm response from test case questions without evaluate score, save all predictions in log file and return file path'''
        if save_path:
            assert os.path.exists(os.path.dirname(save_path)), f"Path does not exist : {save_path}"    
            save_path = os.path.abspath(save_path)
        else:
            prediction_folder = os.path.abspath("./test_log")
            os.makedirs(prediction_folder, exist_ok=True)
            create_time = datetime.now().strftime("%Y%m%d")
            save_path = os.path.join(prediction_folder, remove_special_symbol(f"{create_time}_{test_case.test_name}_{self.graph_name}_prediction") + ".json")
        with open(save_path, mode="w", encoding="utf-8") as answer_file:
            answer_file.truncate(0)
        for tc in test_case.get():
            qutestion = tc["question"]
            reference = tc["answer"]
            id        = tc["id"]
            prediction = self.get_response(qutestion)
            log_msg = {"question" : qutestion, "reference" : reference, "id" : id, "prediction" : prediction}
            write_msg = json.dumps(log_msg, ensure_ascii=False) + "\n"
            with open(save_path, mode="a", encoding="utf-8") as answer_file:
                answer_file.write(write_msg)
        rich.print(f"Saved all predictions at {save_path}")
        return save_path
    
    def run_eval_from_loader(self, test_case: TestCaseLoader, judge_llm: Optional[BaseChatModel]=None, logger: Optional[LogWriter]=None):
        ''' execute evaluation from test case loader '''
        self.score_board = {"test_amount" : 0, "correct": 0, "incorrect": 0}
        for tc in test_case.get():
            score = self.eval_all(tc, judge_llm, logger)
            if score > 0.5:
                self.score_board["correct"] += 1
            else:
                self.score_board["incorrect"] += 1
            self.score_board["test_amount"] += 1
        logger.write_log(self.score_board, "Final Result")
        return None
    
    def run_eval_from_file(self, answer_file: str, judge_llm: Optional[BaseChatModel]=None, logger: Optional[LogWriter]=None):
        ''' execute evaluation from test case loader '''
        self.score_board = {"test_amount" : 0, "correct": 0, "incorrect": 0}
        test_case = load_prediction(answer_file)
        for tc in test_case:
            score = self.eval_all(tc, judge_llm, logger)
            if score > 0.5:
                self.score_board["correct"] += 1
            else:
                self.score_board["incorrect"] += 1
            self.score_board["test_amount"] += 1
        logger.write_log(self.score_board, "Final Result")
        return None
    
    def eval_all(self, test_case: Dict[Literal["question", "reference", "id", "prediction"], str | int], judge_llm: Optional[BaseChatModel]=None, logger: Optional[LogWriter]=None):
        # get all required item from test case
        id         = test_case["id"]
        question   = test_case["question"]
        reference  = test_case["reference"]
        prediction = test_case["prediction"]
        # 
        correctness = self.llm_eval(judge_llm, question, reference, prediction)
        bert_score = self.BERTScore_eval(reference, prediction)
        sim_result = self.answer_relevancy_eval(reference, prediction)
        meteor_result = self.meteor_eval(reference, prediction)
        if isinstance(prediction, Dict) and "reference" in prediction.keys(): # check reference from RAG or other source exists
            groundedness = self.groundedness_eval(reference, prediction["reference"])
        else:
            groundedness = "SKIP"
        # merge results
        result = {
            "Correctness (LLM Eval)" : {"judge" : judge_llm.model, "result" : correctness['value'], "score" : correctness['score'], "completion" : correctness['completion']},
            "BERT Score" : bert_score,
            "Length" : {"reference" : len(reference), "prediction" : len(prediction)},
            "Answer Relevancy" : sim_result,
            "METEOR Score" : meteor_result,
            "Groundedness" : groundedness,
        }
        # display
        rich.print(f"ID : {id}\nResult : {str(result)}")
        # log
        if logger:
            logger.write_s_line(2)
            logger.write_log(result, message_section=f"Test Case ID: {id}")
            tc_context = f"Q : {question}\nA : {reference}\n"
            logger.write_log(tc_context)
            logger.write_log(prediction, message_section="LLM Prediction")
            logger.write_log(correctness['reasoning sections'], message_section="Correctness Reason")
            logger.write_s_line(2)
        return correctness["score"]
            

if __name__ == "__main__":
    # ===============================
    # Load Core Objects

    # -------------------------------
    # Load QA file
    qa_file = r"./prebuild_VDB/mini-wiki_question-answer_1.txt" # example : {'question': 'Was Abraham Lincoln the sixteenth President of the United States?', 'answer': 'yes', 'id': 0}
    test_case = TestCaseLoader("QA_Test")
    test_case.load(qa_file)
    for c in test_case.get():
        print(c)

    # -------------------------------
    # Init core object
    llm = init_LLM(LLM_temperature=0)
    vdb = init_VecDB()
    db_manager = VDBManager(vdb)

    # ===============================
    # Create Main Graph Evaluator

    # -------------------------------
    # Load database
    dataset_name = "mini-wiki"
    raw_data_file = "./BanchMark/Dataset/logs/20250724_rag-datasets_rag-mini-wikipedia-text-corpus.txt"
    if not db_manager.load("prebuild_VDB", dataset_name):
        db_manager.load_from_file(raw_data_file)
        db_manager.save("prebuild_VDB", dataset_name)
    CLI_print("Vector Database", f"Data amount: {db_manager.amount()}")

    # -------------------------------
    # MainGraph
    main_graph = MG.build_main_graph(llm, db_manager)
    AI_name = "JOHN"
    professional_role = "專業AI助理"
    main_graph_prompt_cfg = MG.SystemPromptConfig(
            AI_name = AI_name,
            professional_role = professional_role,
            chat_lang = "English",
            negitive_rule = "Reply with other langue",
            output_format = "Answer QUESTION with a short answer. As simple as possible."
        )

    # -------------------------------
    # Init Evaluator
    main_graph_evaluator = GraphEvaluator(main_graph, main_graph_prompt_cfg)
    
    # ===============================
    # Create Baseline Graph Evaluator

    # -------------------------------
    # Baseline Graph
    baseline_graph = BG.build_baseline_graph(llm)
    baseline_graph_prompt_cfg = BG.SystemPromptConfig(
        chat_lang="English",
        negitive_rule="Reply with other langue",
        output_format="Answer QUESTION with a short answer, as simple as possible"
    )
  
    # -------------------------------
    # Init Evaluator
    baseline_graph_evaluator = GraphEvaluator(baseline_graph, baseline_graph_prompt_cfg)

    # ===============================
    # Execute Evaluation

    # -------------------------------
    # Generate all predictions
    main_graph_prediction     = main_graph_evaluator.generate_prediction(test_case)
    baseline_graph_prediction = baseline_graph_evaluator.generate_prediction(test_case)

    # -------------------------------
    # close test llm
    os.system(f"ollama stop {llm.model}")
    del llm

    # -------------------------------
    # init judge llm
    judge_llm = init_LLM(LLM_model_name="gpt-oss:latest", LLM_temperature=0)

    # -------------------------------
    # use judge llm to evaluate
    main_graph_logger = LogWriter(log_name="MainGraph_Evaluation", log_folder_name="test_log")
    main_graph_logger.clear()

    baseline_graph_logger = LogWriter(log_name="BaselineGraph_Evaluation", log_folder_name="test_log")
    baseline_graph_logger.clear()

    main_graph_evaluator.run_eval_from_file(main_graph_prediction, judge_llm=judge_llm, logger=main_graph_logger)
    baseline_graph_evaluator.run_eval_from_file(baseline_graph_prediction, judge_llm=judge_llm, logger=baseline_graph_logger)

    # -------------------------------
    # close judge llm
    os.system(f"ollama stop {judge_llm.model}")
    del judge_llm

    