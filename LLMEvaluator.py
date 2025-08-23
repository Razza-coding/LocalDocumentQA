import os, sys, json, ast
# custom download path for bert score embeddings
PATH_ROOT     = os.path.abspath("./temp")
BASE_CACHE    = os.path.join(PATH_ROOT, "huggingface")
DATASET_CACHE = os.path.join(BASE_CACHE, "datasets")
HUB_CACHE     = os.path.join(BASE_CACHE, "hub")
os.makedirs(DATASET_CACHE, exist_ok=True)
os.makedirs(HUB_CACHE, exist_ok=True)
os.environ["HF_HOME"] = BASE_CACHE
os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE
os.environ["HF_HUB_CACHE"] = HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_CACHE, "transformers")
os.environ["XDG_CACHE_HOME"] = BASE_CACHE

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.utils.math import cosine_similarity
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import BasePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.translate import meteor_score as nltk_meteor
from datetime import datetime
from transformers import logging
logging.set_verbosity_error()
from bert_score import score as bert_score
from typing import *

import re, math, time
import rich

from config import init_LLM, build_embedding, init_VecDB
from CLI_Format import CLI_input, CLI_next, CLI_print
from LogWriter import LogWriter, remove_special_symbol

embed = build_embedding(normalize_embeddings=True)

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

def run_qa_test(self, llm:Any, score_func:Any = lambda x, GT : bool( str(GT).lower() in str(x).lower() ) ):
    ''' This need rewrite '''
    assert self.qa_pairs is not None, "QA test not loaded"
    score = {"correct" : 0, "wrong" : 0, "total" : 0}
    for qa in self.qa_pairs:
        reply = llm.invoke(input=qa["question"]).text()
        score["total"] += 1
        if score_func(reply, qa["answer"]):
            score["correct"] += 1
        else:
            score["wrong"] += 1
        rich.print(f"Score : {score} \n Question : {{ {qa['question'][:100]} }} \n Reply : {{ {reply[:100]} }}")
    return score

class TestCaseLoader:
    def __init__(self, test_name:Optional[str]=None, qa_file:Optional[str]=None):
        self.test_name = test_name
        self.test_items = load_test_qa(qa_file, q_key="question", a_key="answer") if qa_file else []
        pass

    def __sizeof__(self):
        return len(self.test_items)

    def load(self, qa_file:str):
        ''' load test cases from txt files '''
        ''' append new to existed test cases '''
        self.test_items.extend(load_test_qa(qa_file, q_key="question", a_key="answer"))
    
    def clear(self):
        ''' clean all test item '''
        self.test_items.clear()

    def get(self) -> Generator[Dict[Literal["question", "answer", "id"], int], None, None]:
        ''' Get test item '''
        for tc in self.test_items:
            yield tc

class LLMEvaluator:
    def __init__(self, llm: BaseChatModel | CompiledStateGraph):
        ''' Execute all kinds of test for LLM '''
        assert llm != None, "Please set a valid llm or LangChain Graph"
        self.llm: BaseChatModel | CompiledStateGraph = llm
        self.score_board = {"test_amount" : 0, "correct": 0, "incorrect": 0}
        self.incorrect_items = []
        self.correct_items = []
    
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

    def get_response(self, question:str) -> str:
        ''' get question response from llm, use a style template to get a shorter respond from llm '''
        style_template = "Answer QUESTION with a short answer. QUESTION: {question}"
        return self.llm.invoke(input=style_template.format(question=question)).text()

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
        ''' Eval a single QA result by judge_llm using langchian evalrator, reference : expected answer, prediction : llm output response '''
        ''' Four key section generated by judge_llm: QUESTION、CONTEXT、STUDENT ANSWER、EXPLANATION、GRADE '''
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
            save_path = os.path.join(prediction_folder, remove_special_symbol(f"{create_time}_{test_case.test_name}_{self.llm.model}_prediction") + ".json")
        with open(save_path, mode="w", encoding="utf-8") as af:
            for tc in test_case.get():
                qutestion = tc["question"]
                reference = tc["answer"]
                id        = tc["id"]
                prediction = self.get_response(qutestion)
                log_msg = {"question" : qutestion, "reference" : reference, "id" : id, "prediction" : prediction}
                write_msg = json.dumps(log_msg, ensure_ascii=False) + "\n"
                af.write(write_msg)
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

    qa_file = r"./prebuild_VDB/mini-wiki_question-answer_10.txt" # example : {'question': 'Was Abraham Lincoln the sixteenth President of the United States?', 'answer': 'yes', 'id': 0}

    llm = init_LLM(LLM_temperature=0)

    logger = LogWriter(log_name="LLM_Evaluation", log_folder_name="test_log")
    logger.clear()
    
    evaluator = LLMEvaluator(llm)
    test_case = TestCaseLoader("QA Test")

    # load test case
    test_case.load(qa_file)
    for c in test_case.get():
        print(c)

    # generate all predictions
    pre_generate_answer = evaluator.generate_prediction(test_case)
    # release test llm
    os.system(f"ollama stop {llm.model}")
    del llm
    # init judge llm
    judge_llm = init_LLM(LLM_model_name="gpt-oss:latest", LLM_temperature=0)
    # use judge llm to evaluate
    evaluator.run_eval_from_file(pre_generate_answer, judge_llm=judge_llm, logger=logger)
    # release judge llm
    os.system(f"ollama stop {judge_llm.model}")
    del judge_llm

    