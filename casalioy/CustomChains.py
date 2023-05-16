from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

from casalioy.load_env import (
    model_n_ctx,
    n_forward_documents,
    n_retrieve_documents,
)
from casalioy.utils import print_HTML


class StuffQA:
    """custom QA close to a stuff chain
    compared to the default stuff chain which may exceed the context size, this chain loads as many documents as allowed by the context size.
    Since it uses all the context size, it's meant for a "one-shot" question, not leaving space for a follow-up question which exactly contains the previous one.
    """

    def __init__(self, llm: BaseLanguageModel, retriever: VectorStoreRetriever, prompt: PromptTemplate = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or self.default_prompt
        self.retriever.search_kwargs = {**self.retriever.search_kwargs, "k": n_forward_documents, "fetch_k": n_retrieve_documents}

    @property
    def default_prompt(self) -> PromptTemplate:
        """the default prompt"""
        prompt = """HUMAN:
Answer the question using ONLY the given extracts from (possibly unrelated) documents, not your own knowledge.
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

    def fetch_documents(self, search: str) -> list[Document]:
        """fetch documents from retriever"""
        return self.retriever.get_relevant_documents(search)

    def __call__(self, input_str: str) -> dict:
        """ask a question"""
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
