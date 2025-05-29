# Self Learning Note for LangChain
## Element of LLM Application
```
Question
Context
Query Vector
Database
Respond
```
## Chains
#### Components
```
Prompts
LLM / Chat Model
Output Parsers
Tools
General Function
Valuable Outputs (basicly text that contains information)
```
#### Pipeline
```
data pipeline
LLM chain
CI / CD
```
#### Prompt
```
Prompt Template
Predefined recipes
```
#### Loader
```
Document Loader
Youtube Loader
```
## LCEL (LangChain Expression Language)
#### Runnable
```
"|" operator
Runnable protocal (invoke, lambda, batch, stream)
```
#### Runnable Objects
```
RunnableSequence
RunnableLambda
RunnablePassthriugh
RunnableParallel
```
## Splitter & Retrievers
#### Retrievers
```
An interface, does not store data bt itself
```
#### Splitter
```
Chunking data
Recursive Character Text Splitter
```
#### vector store
```
Store chuncked data into database
```
## RAG
```
Indexing
Retrieve and Generate
```
#### Precision & Recall
```
Precision = Relevant search documents / All search   documents
Recall    = Relevant search documents / All relevent documents
```
## Tool
```
Interfaces for agent, chain, LLM to interact with additional function or data
Toolkits (set of tools for specific task)
```
## Agent
```
LLM decide what sequence of action to execute
Chain and Agent is different
  Chain's action is hard coded
  Agent's action is decided by LLM
End condition strongly depends on task topic
```

## Advence Topics
```
LangGraph
LangServe
LangSmith
```