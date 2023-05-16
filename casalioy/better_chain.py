from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

from casalioy.load_env import (
    get_embedding_model,
    model_n_ctx,
    model_path,
    model_stop,
    model_temp,
    n_forward_documents,
    n_gpu_layers,
    n_retrieve_documents,
    persist_directory,
    use_mlock,
)
from casalioy.startLLM import QASystem
from casalioy.utils import print_HTML


class StuffQA:
    """custom QA close to a stuff chain"""

    @property
    def default_prompt(self) -> PromptTemplate:
        """the default prompt"""
        prompt = """HUMAN: Answer the question using ONLY the given extracts from (possibly unrelated) documents.
If you are unsure of the answer, respond with "Unknown[STOP]".
Conclude your response with "[STOP]" to indicate the completion of the answer.
Each time you take information from an extract, cite it as EXACTLY "[E1]" for extract 1, "[E2]" for extract 2 etc.
Example: "This is part of the answer[E1], this is another part of the answer[E2]"

Question: {question}

{context}

ASSISTANT: """
        return PromptTemplate(template=prompt, input_variables=["context", "question"])

    @staticmethod
    def context_prompt_str(documents: list[Document]) -> str:
        """the document's prompt"""
        prompt = "Context\n------\n"
        for i, document in enumerate(documents):
            prompt += f"Extract {i + 1}: {document.page_content}\n\n"
        return prompt.strip()

    def __init__(self, llm: BaseLanguageModel, retriever: VectorStoreRetriever, prompt: PromptTemplate = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or self.default_prompt
        self.retriever.search_kwargs = {**self.retriever.search_kwargs, "k": n_forward_documents, "fetch_k": n_retrieve_documents}

    def fetch_documents(self, search: str) -> list[Document]:
        """fetch documents from retriever"""
        return self.retriever.get_relevant_documents(search)

    def __call__(self, input_str: str) -> str:
        """ask a question"""
        all_documents, documents = self.fetch_documents(input_str), []
        for document in all_documents:
            documents.append(document)
            context_str = self.context_prompt_str(documents)
            if (
                self.llm.get_num_tokens(self.prompt.format_prompt(question=input_str, context=context_str).to_string())
                > model_n_ctx - self.llm.dict()["max_tokens"] * 2
            ):
                documents.pop()
                break
        context_str = self.context_prompt_str(documents)
        formatted_prompt = self.prompt.format_prompt(question=input_str, context=context_str).to_string()
        print_HTML("<r>Asking: {question}</r>", question=formatted_prompt)
        return self.llm.predict(formatted_prompt)


qa_orig = QASystem(get_embedding_model()[0], persist_directory, model_path, model_n_ctx, model_temp, model_stop, use_mlock, n_gpu_layers)

qa = StuffQA(retriever=qa_orig.qdrant_langchain.as_retriever(search_type="mmr"), llm=qa_orig.llm)

res = qa("What is money ?")
print("answer", res)
