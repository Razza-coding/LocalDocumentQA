from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from agentic_chunker import AgenticChunker
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from LogWriter import LogWriter
from config import init_LLM, build_embedding

'''
Prototype for testing different text split method
Split Methods:
 - Fix size / Manual Split
 - Recursive Char Split
 - Semetic
 - LLM based
 - Agentic grouping

Source:
 - https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
 - https://www.youtube.com/watch?v=pIGRwMjhMaQ
'''

# ===============================
# init

logger = LogWriter(log_name="text_splitter", log_folder_name="test_log", root_folder=os.path.join(os.path.abspath("."), "Prototype"))
logger.clear()
local_llm = init_LLM()
sentence_embedding = build_embedding(model_device='cuda')

# ===============================
# test inputs
# simple sentence
text = "Text splitting in LangChain is a critical feature that facilitates the division of large texts into smaller, manageable segments."
# document
with open(os.path.join(os.path.abspath("."), "Prototype/test_log", "test_document.txt"), 'r', encoding='utf-8') as file:
    file_text = file.read()
# structured text
markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""
python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""
javascript_text = """
// Function is called, the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
  return a * b;
}
"""

# ===============================
# 1. Character Text Splitting

# -------------------------------
# Manual Splitting
# fix size split, no overlap
chunk_size = 35 
#
chunks = []
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
logger.write_log(documents, "Character Text Splitting")

# -------------------------------
# Automatic Text Splitting
# fix size with overlap splitting
chunk_size = 35
overlap = 10
#
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=overlap,
    separator='',
    strip_whitespace=False
)
documents = text_splitter.create_documents([text])
logger.write_log(documents)

# ===============================
# 2. Recursive Character Text Splitting
# Use target separators to split chunks, LangChain default symbol is ["\n\n", "\n", " ", ""]
# Steps:
#  - Split by "\n\n"
#  - Check chunk size
#  - Split by "\n"
#  - continue till all chunk size is small enough or all symbol is used
# Separators can be customized
chunk_size = 450
overlap = 0
#
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
logger.write_log(text_splitter.create_documents([file_text]), "Recursive Character Text Splitting") 

# ===============================
# 3. Document Specific Splitting
logger.write_log("", "")

# -------------------------------
# Document Specific Splitting - Markdown
from langchain.text_splitter import MarkdownTextSplitter
#
chunk_size = 40
overlap = 0
#
splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
logger.write_log(splitter.create_documents([markdown_text]), "Document Specific Splitting - MarkDown")

# -------------------------------
# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter
#
chunk_size = 100
overlap = 0
#
python_splitter = PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
logger.write_log(python_splitter.create_documents([python_text]), "Document Specific Splitting - Python")

# -------------------------------
# Document Specific Splitting - Javascript
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
#
chunk_size = 65
overlap = 0
#
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=chunk_size, chunk_overlap=overlap
)
logger.write_log(js_splitter.create_documents([javascript_text]), "Document Specific Splitting - JavaScript")


# ===============================
# 4. Semantic Chunking
# Percentile 
#  - all differences between sentences are calculated, 
#  - and then any difference greater than the X percentile is split

from langchain_experimental.text_splitter import SemanticChunker
#
breakpoint_threshold_type="percentile"  # "standard_deviation", "interquartile"
#
text_splitter = SemanticChunker(sentence_embedding)
text_splitter = SemanticChunker(
    sentence_embedding, breakpoint_threshold_type=breakpoint_threshold_type
)
documents = text_splitter.create_documents([file_text])
logger.write_log(documents, "Semantic Chunking")

# ===============================
# 5. Agentic Chunking

# -------------------------------
# Proposition-Based Chunking
#
# After splitting text into paragraphs (either semantic chunking or recursive chunking), further splits the paragraphs with AI Agent
# Creates smaller, multipile sentence that have meaning by itself
# https://arxiv.org/pdf/2312.06648.pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from pydantic import BaseModel
from typing import List

# https://smith.langchain.com/hub/wfh/proposal-indexing?organizationId=65e2223e-316a-5256-b012-5033801a97fa
proposal_indexing_template = ChatPromptTemplate.from_messages([
    SystemMessage(
'''Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]
'''),
    HumanMessagePromptTemplate.from_template(
'''Decompose the following:
{input}
''')
])
extraction_prompt = PromptTemplate.from_template(
    """Extract sentences from the following text and output them in JSON format.
Only return the JSON result—do not include any explanations or additional text.

If no sentences are found, return:
{{"sentences": []}}

Text:
{input}
"""
)

# Create Proposition Extraction Function
class Sentences(BaseModel):
    sentences: List[str]
parser = PydanticOutputParser(pydantic_object=Sentences)
obj = hub.pull("wfh/proposal-indexing") # same template as proposal_indexing_template
llm_chunking_runnable = obj | local_llm
fact_extraction_runnable = extraction_prompt | local_llm | parser
def get_propositions(text):
    # LLM list out some facts in text, but yet to be format
    unformated_chunk = llm_chunking_runnable.invoke({"input" : text}).content
    try:
        # extract formated data from string
        propositions = fact_extraction_runnable.invoke({"input" : unformated_chunk}).sentences
    except:
        # fall back, put unextracted_chunk for debug
        propositions = [unformated_chunk]
    return propositions

# Use Recursive Splitter, then use LLM to execute Proposition-Based Chunking
chunk_size = 450
overlap = 0
#
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
paragraphs = text_splitter.split_text(file_text)
# Extract Propositions
text_propositions = []
for i, para in enumerate(paragraphs[:10]):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print(f"Done with {i}")
#
logger.write_log(f"You have {len(text_propositions)} propositions", "Proposition-Based Chunking")
logger.write_log(text_propositions)


# -------------------------------
#  Agent Grouping
# Use LLM to group proposition together by similarity, create a group if no match found
ac = AgenticChunker(llm=local_llm)
ac.add_propositions(text_propositions)
chunks = ac.get_chunks(get_type='list_of_strings')

logger.write_log(ac.pretty_print_chunks(), "Agent Grouping")
logger.write_log(chunks)
