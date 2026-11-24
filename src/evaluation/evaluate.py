from ragas import RunConfig, evaluate, EvaluationDataset
from evaluation.evaluator import eval_llm, eval_embeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from rag.pipeline import RAGPipeline
from pathlib import Path


sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?",
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
]

def main():

    rag_pipeline = RAGPipeline()

    try:
        print("Initializing RAG pipeline...")
        rag_pipeline.initialize()
        print("RAG pipeline ready!")

        for file in Path("docs").glob("*.pdf"):
            rag_pipeline.upload_pdf(
                file_path=str(file),
                filename=file.name
            )

        query = "Who introduced the theory of relativity?"
        answer = rag_pipeline.ask(query)

        print(f"Query: {answer['question']}")
        print(f"Relevant Document: {answer['sources']}")
        print(f"Answer: {answer['answer']}")

        dataset = []

        for query, reference in zip(sample_queries, expected_responses):
            answer=rag_pipeline.ask(query)
            relevant_docs = list(dict.fromkeys(
                doc['content']
                for doc in answer['sources']
            ))
            response=answer['answer']
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": relevant_docs,
                    "response": response,
                    "reference": reference
                }
            )
    finally:
        rag_pipeline.weaviate_client.close()
        print("Shutting down application...")

    eval_dataset = EvaluationDataset.from_list(dataset)
    
    run_config = RunConfig(
        timeout=600,
        max_workers=1,
        max_retries=2,
        max_wait=10
    )
    
    metrics = [
        LLMContextRecall(), 
        Faithfulness(), 
        FactualCorrectness()
    ]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
        raise_exceptions=True
    )
    
    print("\nEvaluation results:")
    print(result)


if __name__ == "__main__":
    main()