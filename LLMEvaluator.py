from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.utils.math import cosine_similarity
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import BasePromptTemplate
from nltk.translate import meteor_score as nltk_meteor
from typing import *
import os, sys, json, ast
import re, math, time
import rich

from config import init_LLM, build_embedding
from CLI_Format import CLI_input, CLI_next, CLI_print
from LogWriter import LogWriter

embed = build_embedding(normalize_embeddings=True)

def embed_str(input_str: str) -> List[float]:
    ''' Embed string '''
    assert isinstance(input_str, str), "Input is not string"
    return embed.embed_query(input_str)

def load_test_qa(file:str, q_key:str, a_key:str) -> List[Optional[Dict]]:
    ''' 
    Read Question and Answer pair for RAG testing / Agent Reply from a text file.
    Each line in file should contain Json, Dict, List object like string
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
            
            if isinstance(qa_item, Dict):
                if "id" not in qa_item:
                    qa_item.update({"id" : str(line_loc)}) # for checking raw data
                if q_key in qa_item and a_key in qa_item:
                    qa_pairs.append(qa_item)
                    continue
            
            rich.print(f"Invalid Data at {line_loc} : {line}")
            continue

    rich.print(f"Loaded QA {len(qa_pairs)} items.")
    return qa_pairs

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
        self.test_items = load_test_qa(qa_file) if qa_file else []
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

    def get(self):
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

    def get_response(self, question:str):
        ''' get question response from llm '''
        return self.llm.invoke(question).text()

    def answer_relevancy_eval(self, reference:str, prediction:Optional[str]=None):
        ''' Calculate similarity between llm predicttion and expected answer, checks realitive '''
        prediction = prediction or "EMPTY PREDICTION"
        er, ep = embed_str(reference), embed_str(prediction)
        similarity = cosine_similarity([er], [ep])
        return {"score": round(similarity.item(), 4)}
    
    def meteor_eval(self, reference:str, prediction:Optional[str]=None):
        ''' Calculate METEOR score between reference and prediction '''
        if prediction is None or len(prediction) == 0:
            return {"score" : 0} # Empty string
        tokenize_r = [ s for s in reference.split(" ")]
        tokenize_p = [ s for s in prediction.split(" ")]
        score = round(nltk_meteor.meteor_score(references=[tokenize_r], hypothesis=tokenize_p), 4)
        return {"score" : score}

    def llm_eval(self, judge_llm: BaseChatModel, question:str, reference:str, prediction:Optional[str]=None):
        ''' Eval a single QA result by judge_llm, reference : expected answer, prediction : llm output response '''
        if judge_llm is None:
            rich.print("Skip LLM Evaluation (empty judge llm)")
            return {"reasoning" : "SKIP", "value" : None, "score" : None}
        assert judge_llm != None, "judge_llm is not set"
        prediction = prediction or "EMPTY PREDICTION"
        evaluator = load_evaluator(EvaluatorType.COT_QA, llm=judge_llm) # LLM QA evaluation with reason
        result = evaluator.evaluate_strings(input=question, prediction=prediction, reference=reference) # expected reasoning, value, score in Dict
        assert all(key in ("reasoning", "value", "score") for key in result) , f"Dict item missing : {str(result)}"
        result["value"] = "INCORRECT" if result["value"] is None else result["value"]
        result["score"] = 0 if result["score"] is None else result["score"]
        return result
    
    def groundedness_eval(self, reference:str, claims:Optional[List[str]]=None, similarity_threshold:float=0.75):
        ''' Calcualte similuarity between claims and reference,  '''
        er = embed_str(reference)
        support   = []
        unsupport = []
        for c in claims:
            ec = embed_str(c)
            similarity = cosine_similarity(er, ec)
            if similarity >= similarity_threshold:
                support.append(c)
            else:
                unsupport.append(c)
        score = round(len(support) / max(1, len(claims)), 4)
        return {"score" : score, "support" : support, "unsupport": unsupport}

    def run_test(self, test_case: TestCaseLoader, judge_llm: Optional[BaseChatModel]=None, logger: Optional[LogWriter]=None):
        for tc in test_case.get():
            id = tc["id"]
            question   = tc["question"]
            reference  = tc["answer"]
            prediction = self.get_response(question)
            # 
            sim_result = self.answer_relevancy_eval(reference, prediction)
            meteor_result = self.meteor_eval(reference, prediction)
            correctness = self.llm_eval(judge_llm, question, reference, prediction)
            if isinstance(prediction, Dict) and "reference" in prediction.keys(): # check reference from RAG or other source exists
                groundedness = self.groundedness_eval(reference, prediction["reference"])
            else:
                groundedness = "SKIP"
            # merge results
            result = {
                "Correctness (LLM Eval)" : f"Result : {correctness['value']} Score : {correctness['score']}",
                "Answer Relevancy" : sim_result["score"],
                "METEOR Score" : meteor_result,
                "Groundedness" : groundedness
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
                logger.write_log(correctness['reasoning'], message_section="Correctness Reason")
                logger.write_s_line(2)
            

if __name__ == "__main__":
    qa_file = r"./prebuild_VDB/mini-wiki_question-answer_100.txt" # example : {'question': 'Was Abraham Lincoln the sixteenth President of the United States?', 'answer': 'yes', 'id': 0}

    llm = init_LLM(LLM_temperature=0)
    judge_llm = init_LLM(LLM_model_name="gpt-oss:latest", LLM_temperature=0)

    logger = LogWriter(log_name="LLM_Evaluation", log_folder_name="test_log")
    logger.clear()
    
    evaluator = LLMEvaluator(llm)
    case = TestCaseLoader("QA Test")
    case.load(qa_file)

    # list test came
    for c in case.get():
        print(c)
    
    #
    evaluator.run_test(case, judge_llm=judge_llm, logger=logger)

    