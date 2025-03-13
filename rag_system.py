"""
Retrieval-Augmented Generation (RAG) system using our fine-tuned Llama model.
"""

import torch
from typing import List, Dict, Any, Optional
from zenml.logger import get_logger

# Import our modules
from config import TEST_QUESTIONS, SYSTEM_PROMPT, HF_TOKEN, MAX_NEW_TOKENS
from vectordb import ChromaVectorDB
from utils import extract_structured_answer

logger = get_logger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation system using Llama and ChromaDB."""
    
    def __init__(
        self,
        model_path: str,
        vectordb_collection: str = "rag_knowledge",
        persist_directory: str = "./chroma_data",
        top_k_results: int = 3
    ):
        """Initialize the RAG system.
        
        Args:
            model_path: Path to the fine-tuned model
            vectordb_collection: Name of the vector database collection
            persist_directory: Directory to persist the vector database
            top_k_results: Number of results to retrieve from the vector database
        """
        self.model_path = model_path
        self.top_k_results = top_k_results
        
        # Initialize the vector database
        self.vectordb = ChromaVectorDB(
            collection_name=vectordb_collection,
            persist_directory=persist_directory
        )
        
        # Initialize the model and tokenizer
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=HF_TOKEN,
        )
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
        )
        
        logger.info("Model and tokenizer initialized")
    
    def add_knowledge(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add knowledge to the vector database.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of ids for the added texts
        """
        return self.vectordb.add_texts(texts, metadatas=metadatas)
    
    def retrieve(self, query: str) -> Dict[str, List]:
        """Retrieve relevant information from the vector database.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        return self.vectordb.search(query, n_results=self.top_k_results)
    
    def generate(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = MAX_NEW_TOKENS,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """Generate a response to the question.
        
        Args:
            question: Question to answer
            temperature: Temperature for generation
            max_new_tokens: Maximum number of tokens to generate
            use_rag: Whether to use RAG or not
            
        Returns:
            Dictionary with generated text, reasoning, and answer
        """
        # Prepare prompt with or without RAG
        if use_rag and self.vectordb.get_collection_stats()["count"] > 0:
            # Retrieve relevant information
            results = self.retrieve(question)
            
            # Build context from retrieved documents
            context = "Here is some relevant information:\n\n"
            for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
                context += f"Document {i+1}: {doc}\n"
                if "source" in metadata:
                    context += f"Source: {metadata['source']}\n"
                context += "\n"
            
            # Prepare RAG prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nWith this information, please answer the following question: {question}"}
            ]
        else:
            # Standard prompt without RAG
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        
        # Format prompt
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate answer
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_params
            )
        
        # Decode the generated output
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        # Extract reasoning and answer
        reasoning, answer = extract_structured_answer(generated_text)
        
        # If no structured answer found, use the whole text
        if not answer:
            answer = generated_text
            
        return {
            "question": question,
            "full_response": generated_text,
            "reasoning": reasoning,
            "answer": answer,
            "rag_used": use_rag
        }
    
    def load_test_questions_knowledge(self) -> None:
        """Load test questions as knowledge for RAG."""
        # Prepare metadata for the questions (ChromaDB requires non-empty metadata)
        metadatas = [{"source": "test_question", "question_id": i} for i in range(len(TEST_QUESTIONS))]
        
        # Add questions to the vector database
        self.add_knowledge(TEST_QUESTIONS, metadatas=metadatas)
        
        logger.info(f"Loaded {len(TEST_QUESTIONS)} test questions as knowledge")
    
    def load_sample_answers_knowledge(self, answers: List[str]) -> None:
        """Load sample answers as knowledge for RAG.
        
        Args:
            answers: List of answers to add
        """
        if len(answers) != len(TEST_QUESTIONS):
            raise ValueError("Number of answers must match number of test questions")
        
        # Prepare metadata for the answers
        metadatas = [
            {"source": "sample_answer", "question_id": i, "for_question": q}
            for i, (q, a) in enumerate(zip(TEST_QUESTIONS, answers))
        ]
        
        # Add answers to the vector database
        self.add_knowledge(answers, metadatas=metadatas)
        
        logger.info(f"Loaded {len(answers)} sample answers as knowledge")

def main():
    """Demo the RAG system."""
    # Initialize the RAG system with the fine-tuned model
    rag = RAGSystem(
        model_path="llama_finetuned",  # Path to the fine-tuned model
        vectordb_collection="rag_demo",
        persist_directory="./chroma_data"
    )
    
    # Load test questions as knowledge
    rag.load_test_questions_knowledge()
    
    # Generate answers without and with RAG
    question = "What happens if you crack your knuckles a lot?"
    
    print(f"\nQuestion: {question}")
    print("\nGenerating answer without RAG...")
    result_without_rag = rag.generate(question, use_rag=False)
    
    print("Answer without RAG:")
    print(f"  {result_without_rag['answer']}")
    
    print("\nGenerating answer with RAG...")
    result_with_rag = rag.generate(question, use_rag=True)
    
    print("Answer with RAG:")
    print(f"  {result_with_rag['answer']}")
    
    # Compare answers
    print("\nComparison:")
    print("  Without RAG:", result_without_rag['answer'][:100], "...")
    print("  With RAG:", result_with_rag['answer'][:100], "...")

if __name__ == "__main__":
    main()