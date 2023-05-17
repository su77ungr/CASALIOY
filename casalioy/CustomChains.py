"""Custom chains for LLM"""

from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.qa_generation.prompt import PROMPT_SELECTOR
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

from casalioy.load_env import (
    model_n_ctx,
    n_forward_documents,
    n_retrieve_documents,
)
from casalioy.utils import print_HTML


class BaseQA:
    """base class for Question-Answering"""

    def __init__(self, llm: BaseLanguageModel, retriever: VectorStoreRetriever, prompt: PromptTemplate = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or self.default_prompt
        self.retriever.search_kwargs = {**self.retriever.search_kwargs, "k": n_forward_documents, "fetch_k": n_retrieve_documents}

    @property
    def default_prompt(self) -> PromptTemplate:
        """the default prompt"""
        return PROMPT_SELECTOR.get_prompt(self.llm)

    def fetch_documents(self, search: str) -> list[Document]:
        """fetch documents from retriever"""
        return self.retriever.get_relevant_documents(search)

    def __call__(self, input_str: str) -> dict:
        """ask a question, return results"""
        return {"result": self.llm.predict(self.default_prompt.format_prompt(question=input_str).to_string())}


class StuffQA(BaseQA):
    """custom QA close to a stuff chain
    compared to the default stuff chain which may exceed the context size, this chain loads as many documents as allowed by the context size.
    Since it uses all the context size, it's meant for a "one-shot" question, not leaving space for a follow-up question which exactly contains the previous one.
    """

    @property
    def default_prompt(self) -> PromptTemplate:
        """the default prompt"""
        prompt = """HUMAN:
Answer the question using ONLY the given extracts from (possibly unrelated and irrelevant) documents, not your own knowledge.
If you are unsure of the answer or if it isn't provided in the extracts, answer "Unknown[STOP]".
Conclude your answer with "[STOP]" when you're finished.

Question: {question}

--------------
Here are the extracts:
{context}

--------------
Remark: do not repeat the question !

ASSISTANT:
"""
        return PromptTemplate(template=prompt, input_variables=["context", "question"])

    @staticmethod
    def context_prompt_str(documents: list[Document]) -> str:
        """the document's prompt"""
        prompt = "".join(f"Extract {i + 1}: {document.page_content}\n\n" for i, document in enumerate(documents))
        return prompt.strip()

    def __call__(self, input_str: str) -> dict:
        all_documents, documents = self.fetch_documents(input_str), []
        for document in all_documents:
            documents.append(document)
            context_str = self.context_prompt_str(documents)
            if (
                self.llm.get_num_tokens(self.prompt.format_prompt(question=input_str, context=context_str).to_string())
                > model_n_ctx - self.llm.dict()["max_tokens"]
            ):
                documents.pop()
                break
        print_HTML("<r>Stuffed {n} documents in the context</r>", n=len(documents))
        context_str = self.context_prompt_str(documents)
        formatted_prompt = self.prompt.format_prompt(question=input_str, context=context_str).to_string()
        return {"result": self.llm.predict(formatted_prompt), "source_documents": documents}


class RefineQA(BaseQA):
    """custom QA close to a refine chain"""

    @property
    def default_prompt(self) -> PromptTemplate:
        """the default prompt"""
        prompt = f"""HUMAN:
Answer the question using ONLY the given extracts from a (possibly irrelevant) document, not your own knowledge.
If you are unsure of the answer or if it isn't provided in the extract, answer "Unknown[STOP]".
Conclude your answer with "[STOP]" when you're finished.
Avoid adding any extraneous information.

Question:
-----------------
{{question}}

Extract:
-----------------
{{context}}

ASSISTANT:
"""
        return PromptTemplate(template=prompt, input_variables=["context", "question"])

    @property
    def refine_prompt(self) -> PromptTemplate:
        """prompt to use for the refining step"""
        prompt = f"""HUMAN:
Refine the original answer to the question using the new (possibly irrelevant) document extract.
Use ONLY the information from the extract and the previous answer, not your own knowledge.
The extract may not be relevant at all to the question.
Conclude your answer with "[STOP]" when you're finished.
Avoid adding any extraneous information.

Question:
-----------------
{{question}}

Original answer:
-----------------
{{previous_answer}}

New extract:
-----------------
{{context}}

Reminder:
-----------------
If the extract is not relevant or helpful, don't even talk about it. Simply copy the original answer, without adding anything.
Do not copy the question.

ASSISTANT:
"""
        return PromptTemplate(template=prompt, input_variables=["context", "question", "previous_answer"])

    def __call__(self, input_str: str) -> dict:
        """ask a question"""
        documents = self.fetch_documents(input_str)
        last_answer, score = None, None
        for i, doc in enumerate(documents):
            print_HTML("<r>Refining from document {i}/{N}</r>", i=i + 1, N=len(documents))
            prompt = self.default_prompt if i == 0 else self.refine_prompt
            if i == 0:
                formatted_prompt = prompt.format_prompt(question=input_str, context=doc.page_content)
            else:
                formatted_prompt = prompt.format_prompt(question=input_str, context=doc.page_content, previous_answer=last_answer)
            last_answer = self.llm.predict(formatted_prompt.to_string())
        return {
            "result": f"{last_answer}",
            "source_documents": documents,
        }
