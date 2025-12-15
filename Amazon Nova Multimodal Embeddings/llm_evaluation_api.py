# eval_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import asyncio

import os

os.environ['AWS_PROFILE'] = 'devan2'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import AmazonBedrockModel
import hashlib
import json
import boto3

profile_name = "devan2"
region_name = 'us-east-1'

boto3.setup_default_session(profile_name=profile_name, region_name=region_name)

#making fastapi object, gonna be used for the RAG eval of the llm response
app = FastAPI(title="RAG Evaluation API", version="1.0.0")


# Initialize Bedrock judge with explicit profile
aws_judge = AmazonBedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    # profile_name=profile_name
)

# Request model
#Field -> specify default values, ... = no default, must provide
#literal -> only allow these types
class EvalRequest(BaseModel):
    query: str
    context: str
    response: str
    eval_type: Literal["text", "image", "combined"]
    threshold: float = 0.7
    expected_output: Optional[str] = None  # Required for Contextual Recall

# Response model
class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    contextual_relevancy: float
    contextual_recall: Optional[float] = None # for contextual recall, requires the expected output
    average_score: float
    faithfulness_reason: str
    answer_relevancy_reason: str
    contextual_relevancy_reason: str
    contextual_recall_reason: Optional[str] = None  # for contextual recall, requires the expected output

    passed: bool


# Simple in-memory cache (Redis for production)
# Store results here
eval_cache = {}

#making caching key, reuse previous results
def cache_key(req: EvalRequest) -> str:
    """Generate cache key from request"""
    content = f"{req.query}|{req.context}|{req.response}"

    #md5 hashing algo
    #encoding the string to bytes then hashing then converting that to hex
    return hashlib.md5(content.encode()).hexdigest()


#To check if API is running
#curl http://localhost:8000/health
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}



@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    """
    Evaluate LLM response using DeepEval metrics
    
    - **query**: The user's question
    - **context**: Retrieved context from RAG
    - **response**: LLM's generated response
    - **eval_type**: Type of retrieval (text/image/combined)
    - **threshold**: Minimum score to pass (0.0-1.0)
    """
    
    # Check cache
    key = cache_key(req)
    if key in eval_cache:
        return eval_cache[key]
    
    try:
   
       
        # Create test case object, using fields from inputted class
        test_case = LLMTestCase(
            input=req.query,
            actual_output=req.response,
            retrieval_context=[req.context],
            expected_output=req.expected_output  # Added for Contextual Recall
        )

                      
            

       
        # Define metrics

        # evaluating whether the actual_output factually aligns with the contents of your retrieval_context.
        # Does response STICK to context? (hallucinaion etc.)
        faithfulness = FaithfulnessMetric(model=aws_judge, threshold=req.threshold)

        # evaluating how relevant the actual_output of your LLM application is compared to the provided input
        # Does response answer the question?
        answer_relevancy = AnswerRelevancyMetric(model=aws_judge, threshold=req.threshold)

        # evaluating the overall relevance of the information presented in your retrieval_context for a given input
        # Is the retrieved context relevant to the query?
        contextual_relevancy = ContextualRelevancyMetric(model=aws_judge, threshold=req.threshold)

        # Evaluating the extent of which the retrieval_context aligns with the expected_output
        # Did we retrieve enough context?
        contextual_recall = None
        if req.expected_output:
            contextual_recall = ContextualRecallMetric(model=aws_judge, threshold=req.threshold)
            # measure happens in asyncio.gather, not here

        # Run evaluations sequentially to avoid async loop conflicts
        faithfulness.measure(test_case)
        answer_relevancy.measure(test_case)
        contextual_relevancy.measure(test_case)
        
        if contextual_recall:
            contextual_recall.measure(test_case)

        # Calculate average score from available metrics
        scores = [
            faithfulness.score, 
            answer_relevancy.score, 
            contextual_relevancy.score
        ]
        
        if contextual_recall:
            scores.append(contextual_recall.score)
            

            
        avg_score = sum(scores) / len(scores)
        
       
        # Packaging our result using defined class w/ pydantic
        result = EvalResponse(
            faithfulness=faithfulness.score,
            answer_relevancy=answer_relevancy.score,
            contextual_relevancy=contextual_relevancy.score,
            contextual_recall=contextual_recall.score if contextual_recall else None, #for context recall
            average_score=avg_score,

            #getattr for edge cases
            faithfulness_reason=getattr(faithfulness, 'reason', ''),
            answer_relevancy_reason=getattr(answer_relevancy, 'reason', ''),
            contextual_relevancy_reason=getattr(contextual_relevancy, 'reason', ''),
            contextual_recall_reason=getattr(contextual_recall, 'reason', '') if contextual_recall else None, #for context recall
            # factual_correctness_score= 
            passed=avg_score >= req.threshold
        )
      
        # Cache result
        eval_cache[key] = result
        
        return result
        
    except Exception as e:
        import traceback
        print("="*80)
        print("EVALUATION ERROR:")
        print(traceback.format_exc())  # Print full error
        print("="*80)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

#clearing saved cache
@app.delete("/cache")
async def clear_cache():
    """Clear evaluation cache"""
    eval_cache.clear()
    return {"message": "Cache cleared", "items_removed": len(eval_cache)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)