import json
import requests
from datetime import datetime

# reusing our other functions from indexing image contents
from indexing_image_contents import (
    load_vector_stores,
    query_text_index,
    query_image_index,
    get_combined_claude_response
)

def run_evaluation_tests():
    """
    Run evaluation tests using pre-defined test dataset.
    Tests the RAG system with ground truth answers.
    """
    
    print("=" * 80)
    print("STARTING RAG SYSTEM EVALUATION")
    print("=" * 80)
    
    # reusing vector stores
    print("\nLoading vector stores...")
    text_store, image_store = load_vector_stores()
    
    if not text_store or not image_store:
        print("Error: Could not load vector stores. Make sure indexes exist.")
        return
    
    print("Vector stores loaded successfully")
    
    # Load test cases from json file
    print("\nLoading test cases...")
    try:
        with open('test_dataset.json') as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print("Error: test_dataset.json not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in test_dataset.json")
        return
    
    # Store all results for saving to file
    all_results = []
    
    # Run each test case
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {idx}/{len(test_cases)}: {test['query']}")
        print(f"{'─' * 80}")
       
        try:
            # Step 1: Query both indexes (REUSING functions)
            print("Retrieving context...")
            text_results = query_text_index(text_store, test['query'], k=2)
            image_results = query_image_index(image_store, test['query'], k=2)
            
            # Step 2: Build context strings
            text_context = "\n\n".join([
                f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}:\n{doc.page_content}"
                for doc, _ in text_results
            ])
            
            image_context = "\n\n".join([
                f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}"
                for doc, _ in image_results
            ])
            
            print(f"   ✓ Retrieved {len(text_results)} text chunks, {len(image_results)} images")
            
            # Step 3: Get LLM response (REUSING function)
            print("Generating response...")
            response = get_combined_claude_response(
                query=test['query'],
                text_context=text_context,
                image_context=image_context,
                image_results=image_results
            )
            print(f"   ✓ Response generated ({len(response)} chars)")
            print(f"   DEBUG FULL RESPONSE: {response}")

           
            # Step 4: Evaluate with API (includes expected_output for all 5 metrics)
            print("Evaluating response...")
            
            combined_context = f"{text_context}\n\n{image_context}"
            
            eval_response = requests.post(
                "http://localhost:8000/evaluate",
                json={
                    "query": test['query'],
                    "context": combined_context,
                    "response": response,
                    "expected_output": test['expected_output'],  # Ground truth
                    "eval_type": "combined",
                    "threshold": 0.7
                },
                timeout=90  # Longer timeout for 5 metrics
            )
           
            eval_response.raise_for_status()
            # After eval_response.raise_for_status()
            metrics = eval_response.json()
            print(f"\nDEBUG REASONS:")
            print(f"   Answer Relevancy: {metrics.get('answer_relevancy_reason', 'N/A')}")
            print(f"   Contextual Relevancy: {metrics.get('contextual_relevancy_reason', 'N/A')}")
            print(f"   Contextual Recall: {metrics.get('contextual_recall_reason', 'N/A')}")

            metrics = eval_response.json()
            
            # Step 5: Display results
            print("\nEVALUATION RESULTS:")
            print(f"   • Faithfulness:          {metrics['faithfulness']:.1%}")
            print(f"   • Answer Relevancy:      {metrics['answer_relevancy']:.1%}")
            print(f"   • Contextual Relevancy:  {metrics['contextual_relevancy']:.1%}")
            print(f"   • Contextual Recall:     {metrics['contextual_recall']:.1%}")
            # print(f"   • Factual Correctness:   {metrics['factual_correctness']:.1%}")
            print(f"   • Average Score:         {metrics['average_score']:.1%}")
            print(f"   • Status:                {'passed' if metrics['passed'] else 'failed'}")
           
            # Store for JSON output
            all_results.append({
                'test_number': idx,
                'query': test['query'],
                'expected_output': test['expected_output'],
                'actual_response': response,
                'retrieved_slides_text': [doc.metadata['slide_number'] for doc, _ in text_results],
                'retrieved_slides_image': [doc.metadata['slide_number'] for doc, _ in image_results],
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to evaluation API")
            print("   Make sure the API is running: python llm_evaluation_api.py")
            continue
            
        except requests.exceptions.Timeout:
            print("Error: Evaluation API timeout (took >90 seconds)")
            continue
            
        except Exception as e:
            print(f"Error during test: {str(e)}")
            continue
    
    # Save results to JSON file
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(results_filename, 'w') as f:
            json.dump({
                'test_run_timestamp': datetime.now().isoformat(),
                'total_tests': len(test_cases),
                'completed_tests': len(all_results),
                'results': all_results
            }, f, indent=2)
        
        print(f"Results saved to: {results_filename}")
        
        # Calculate summary statistics
        if all_results:
            avg_faithfulness = sum(r['metrics']['faithfulness'] for r in all_results) / len(all_results)
            avg_answer_relevancy = sum(r['metrics']['answer_relevancy'] for r in all_results) / len(all_results)
            avg_contextual_relevancy = sum(r['metrics']['contextual_relevancy'] for r in all_results) / len(all_results)
            avg_contextual_recall = sum(r['metrics']['contextual_recall'] for r in all_results) / len(all_results)
            # avg_factual_correctness = sum(r['metrics']['factual_correctness'] for r in all_results) / len(all_results)
            passed_count = sum(1 for r in all_results if r['metrics']['passed'])
            
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(f"Tests Completed:         {len(all_results)}/{len(test_cases)}")
            print(f"Tests Passed:            {passed_count}/{len(all_results)} ({passed_count/len(all_results)*100:.1f}%)")
            print(f"\nAverage Scores:")
            print(f"   • Faithfulness:          {avg_faithfulness:.1%}")
            print(f"   • Answer Relevancy:      {avg_answer_relevancy:.1%}")
            print(f"   • Contextual Relevancy:  {avg_contextual_relevancy:.1%}")
            print(f"   • Contextual Recall:     {avg_contextual_recall:.1%}")
            # print(f"   • Factual Correctness:   {avg_factual_correctness:.1%}")
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Check if evaluation API is running
    print("Checking if evaluation API is running...")
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=5)
        if health_check.status_code == 200:
            print("Evaluation API is running\n")
            run_evaluation_tests()
        else:
            print("Evaluation API is not responding correctly")
    except requests.exceptions.ConnectionError:
        print("Evaluation API is not running!")
        print("   Please start it first: python llm_evaluation_api.py")