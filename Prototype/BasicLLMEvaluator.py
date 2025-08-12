from __future__ import annotations
from typing import *
import os, sys, json, ast
import re, math, time
import rich
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.prompts import BasePromptTemplate
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import init_LLM, build_embedding
from CLI_Format import CLI_input, CLI_next, CLI_print
from LogWriter import LogWriter

'''
Execute various LLM / RAG system tests.
Tested Tools:
 - LangChain Evaluator
 - DeepVal
 - self written test code
'''

def load_test_qa(file:str, q_key:str, a_key:str) -> List[Optional[Dict]]:
    ''' 
    Read Question and Answer pair for RAG testing / Agent Reply from a text file.
    Each line in file should contain Json, Dict, List object like string
    return : [{ "question" : "foo1", "answer" : "foo2", "location" : "line in file" }, ...]
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
                if q_key in qa_item and a_key in qa_item:
                    qa_item.update({"location" : str(line_loc)}) # for checking raw data
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

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return (dot / (na * nb) + 1.0) / 2.0 

_SENT_SPLIT = re.compile(r'(?<=[。！？!?．.;:])\s+|[\n\r]+')
def _split_into_claims(text: str, min_len: int = 3) -> list[str]:
    ''' Split sentence into multi claim sentences '''
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and len(p.strip()) >= min_len]
    return parts if parts else ([text.strip()] if text.strip() else [])

def generate_answer(
    gen_llm: Any,
    question: str,
    max_tokens: Optional[int] = None,
    timeout_s: float = 30.0,
    logger: Any | None = None
) -> str:
    start = time.time()
    try:
        out = gen_llm.invoke(question)
        text = getattr(out, "content", None) or getattr(out, "text", None) or str(out)
    except Exception as e:
        text = f"[ERROR_GENERATE]: {e}"
    if logger:
        logger.write_log(log_message=f"Generated Answer: {text[:200]}", message_section="GEN", add_time=True)
    return str(text).strip()

# LLM Scoring
def llm_eval_score(
    judge_llm: Any,
    answer: str,
    reference: str | None,
    question: str | None,
    temperature: float = 0.0,
    logger: Any | None = None
) -> Tuple[float, str]:
    sys_rules = (
        "You are a strict evaluator. Score the candidate answer from 0.0 to 1.0.\n"
        "Scoring rubric:\n"
        "- 1.0: Fully correct and directly answers the question.\n"
        "- 0.7: Mostly correct; minor omissions.\n"
        "- 0.4: Partially correct or vague.\n"
        "- 0.0: Incorrect or irrelevant.\n"
        "Return JSON: {\"score\": <float>, \"reason\": \"...\"}"
    )
    parts = []
    if question: parts.append(f"Question:\n{question}")
    if reference: parts.append(f"Reference:\n{reference}")
    parts.append(f"Answer:\n{answer}")
    user_msg = "\n\n".join(parts)

    prompt = f"{sys_rules}\n\n{user_msg}\n\nRespond ONLY with JSON."
    try:
        judge = judge_llm
        out = judge.invoke(prompt)
        text = getattr(out, "content", None) or getattr(out, "text", None) or str(out)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = float(max(0.0, min(1.0, float(data.get("score", 0.0)))))
            reason = str(data.get("reason", "")).strip()
        else:
            score, reason = 0.0, f"[PARSE_ERROR] raw: {text[:200]}"
    except Exception as e:
        score, reason = 0.0, f"[JUDGE_ERROR] {e}"
    if logger:
        logger.write_log(log_message=f"Judge Score: {score}, Reason: {reason}", message_section="LLM_EVAL", add_time=True)
    return score, reason

# similarity
def embedding_similarity(
    text_a: str,
    text_b: str,
    embed_fn: Callable[[str], list[float]],
    logger: Any | None = None
) -> float:
    ''' evaluate similarity between llm prediction and actual answer '''
    try:
        va = embed_fn(text_a) or []
        vb = embed_fn(text_b) or []
        sim = _cosine(va, vb)
    except Exception:
        sim = 0.0
    if logger:
        logger.write_log(log_message=f"Embedding Similarity: {sim}", message_section="SIM", add_time=True)
    return sim

# goundedness
def groundedness_score(
    answer: str,
    references: list[str],
    embed_fn: Callable[[str], list[float]],
    similarity_threshold: float = 0.75,
    logger: Any | None = None
) -> Tuple[float, list[str]]:
    ''' split and embed llm output sentence into claims, embed all reference sentences'''
    ''' calculate similuarity between claims and refs, check if any claim matches any refs (similarity over threshold) '''
    # get claim
    claims = _split_into_claims(answer)
    if not claims:
        return 0.0, []
    unsupported: list[str] = []
    # get reference
    ref_vecs = [(r, embed_fn(r)) for r in references] if references else []
    # match
    for c in claims:
        vc = embed_fn(c)
        best = 0.0
        for _, vr in ref_vecs:
            best = max(best, _cosine(vc, vr))
        if best < similarity_threshold:
            unsupported.append(c)
    # calculate supported percentage
    score = (len(claims) - len(unsupported)) / max(1, len(claims))
    if logger:
        logger.write_log(
            log_message=f"Groundedness: {score}, Unsupported: {unsupported}",
            message_section="GROUND",
            add_time=True
        )
    return float(score), unsupported

# Task Completion
def task_completion_score(
    answer: str,
    schema: dict | None = None,
    required_fields: list[str] | None = None,
    logger: Any | None = None
) -> float:
    ''' check output structure valid '''
    if not schema and not required_fields:
        return 1.0
    try:
        data = json.loads(answer)
    except Exception:
        if logger:
            logger.write_log(log_message="TaskCompletion: JSON parse error", message_section="TASK", add_time=True)
        return 0.0
    passed = 1.0
    if required_fields:
        for k in required_fields:
            if k not in data:
                passed = 0.0
                break
    if logger:
        logger.write_log(log_message=f"TaskCompletion Score: {passed}", message_section="TASK", add_time=True)
    return float(passed)

# ---------- 異常檢測 ----------
def detect_anomalies(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    logger: Any | None = None
) -> list[str]:
    flags: list[str] = []
    if "judge" in thresholds and metrics.get("judge", 1.0) < thresholds["judge"]:
        flags.append("low_judge")
    if "sim" in thresholds and metrics.get("sim", 1.0) < thresholds["sim"]:
        flags.append("low_similarity")
    if "grounded" in thresholds and metrics.get("grounded", 1.0) < thresholds["grounded"]:
        flags.append("ungrounded_claims")
    if "task" in thresholds and metrics.get("task", 1.0) < thresholds["task"]:
        flags.append("incomplete_task")
    if logger:
        logger.write_log(log_message=f"Anomalies: {flags}", message_section="ANOM", add_time=True)
    return flags

if __name__ == "__main__":
    # test subject
    local_llm = init_LLM(LLM_temperature=0)
    judge_llm = local_llm
    embedding = build_embedding()
    def embed_fn(str:str) -> str:
        return embedding.embed_documents([str])[0]
    
    # logger
    logger = LogWriter(log_name="mini-wiki_QA_test", log_folder_name="test_log", root_folder="Prototype")
    logger.clear()
    logger.write_log(log_message="Start Test", message_section="QA test initial stage", add_time=True)

    # Load test QA
    qa_file = r"./prebuild_VDB/mini-wiki_question-answer_100.txt" # example : {'question': 'Was Abraham Lincoln the sixteenth President of the United States?', 'answer': 'yes', 'id': 0}
    qa_pairs = load_test_qa(qa_file, q_key="question", a_key="answer") 
    #
    logger.write_log(log_message=f"QA source file : {qa_file}")
    logger.write_log(log_message=f"QA amount      : {len(qa_pairs)}")
    CLI_print("VDB Test", f"{qa_pairs[:1]}", "test qa")
    CLI_next()


    # Use 1 qa as execute example
    question = qa_pairs[0]["question"]
    answer   = qa_pairs[0]["answer"]
    prediction = local_llm.invoke(question).text()
    CLI_print(message=prediction)

    # use LangChain QA EvalChain
    qa_eval = QAEvalChain.from_llm(llm=judge_llm)
    result = qa_eval.evaluate(
        examples=[{"query": question, "answer": answer}],
        predictions=[{"result": prediction}],
        )
    print(result)

    # use LangChain load_evalator
    evaluator = load_evaluator(evaluator=EvaluatorType.QA, llm=local_llm)
    result = evaluator.evaluate_strings(
        input=question,
        prediction=prediction,
        reference=answer
    )
    print(result)

    # QA with explain (note: check \envs\LLMbuilder\Lib\site-packages\langchain\evaluation\qa\eval_prompt.py for pre made template)
    evaluator = load_evaluator(evaluator=EvaluatorType.COT_QA, llm=local_llm)
    result = evaluator.evaluate_strings(
        input=question,
        prediction=prediction,
        reference=answer
    )
    print(result)


    # use DeepEval
    tc = LLMTestCase(
        input=question or "",
        actual_output=answer,
        expected_output=prediction or "",
    )    
    metric = GEval(
        name="LLM-Judge",
        criteria="Score 0-1 by correctness and directness.",
        model=None, # use cmd to set up model: deepeval set-ollama gemma3:4b
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    metric.measure(tc)
    score = float(metric.score or 0.0)
    reason = str(metric.reason or "").strip()
    metric = {"metric":"deepeval_llm_judge","score":score,"reason":reason}
    rich.print(metric)
    if logger:
        logger.write_log(
            log_message=json.dumps(metric, ensure_ascii=False),
            message_section="DEEP_EVAL",
            add_time=True
        )

    # self written eval
    qa = {"question" : question, "answer" : answer}
    judge_score, judge_reason = llm_eval_score(judge_llm, prediction, qa.get("answer"), qa.get("question"), logger=logger)
    sim_score = embedding_similarity(prediction, qa.get("answer",""), embed_fn, logger=logger)
    g_score, unsupported = groundedness_score(prediction, [qa.get("answer","")], embed_fn, logger=logger)

    # score matrix, flags
    metrics = {"judge": judge_score, "sim": sim_score, "grounded": g_score, "final": 0.5*judge_score + 0.3*sim_score + 0.2*g_score}
    flags = detect_anomalies(metrics, {"judge":0.6, "sim":0.7, "grounded":0.8, "task":1.0}, logger=logger)

    # log result
    logger.write_log(
        log_message=json.dumps({
            "id": qa.get("id"),
            "question": qa["question"],
            "prediction": prediction,
            "reference": qa.get("answer",""),
            "metrics": metrics,
            "judge_reason": judge_reason,
            "unsupported_claims": unsupported,
            "anomalies": flags
        }, ensure_ascii=False),
        message_section="EVAL_RESULT",
        add_time=True
    )
