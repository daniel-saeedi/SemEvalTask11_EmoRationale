import argparse
import os
import json
import hashlib
from anthropic import Anthropic
import pandas as pd
from tqdm import tqdm
import ast
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize the API key and Anthropic client
api_key = config['api']['anthropic_api_key']
anthropic = Anthropic(api_key=api_key)

# Cache directory setup
CACHE_DIR = Path('prompt_cache')
CACHE_DIR.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize sentence transformer embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': device}
)

def get_cache_key(text, system_prompt):
    """Generate a unique cache key for a given text and system prompt."""
    combined = f"{text}||{system_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(cache_key):
    """Get cached response if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def cache_response(cache_key, response):
    """Cache the response."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(response, f)

def create_vector_store(csv_path):
    """Create a FAISS vector store from the training data."""
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    
    # Create metadata for each text
    metadatas = []
    for _, row in df.iterrows():
        emotions = []
        if row['joy'] == 1: emotions.append('joy')
        if row['fear'] == 1: emotions.append('fear')
        if row['anger'] == 1: emotions.append('anger')
        if row['sadness'] == 1: emotions.append('sadness')
        if row['disgust'] == 1: emotions.append('disgust')
        if row['surprise'] == 1: emotions.append('surprise')
        if not emotions: emotions.append('none')
        metadatas.append({'emotions': emotions})
    
    # Create and return the vector store
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def get_similar_examples(vector_store, query_text, k=5):
    """Retrieve k most similar examples for a given text."""
    results = vector_store.similarity_search_with_score(query_text, k=k)
    
    examples = ''
    for i, (doc, score) in enumerate(results):
        examples += f'Input {i}: {doc.page_content}\n'
        examples += f'Output {i}: [{", ".join(doc.metadata["emotions"])}]\n'
    
    return examples

def get_combined_prompt(examples):
    """Generate a single system prompt that combines reasoning and classification."""
    return f"""You are an emotion classification expert. Your task has two parts:

1. First, analyze the text and provide evidence for and against the presence of each emotion:
   joy, fear, anger, sadness, disgust, surprise

   Guidelines for analysis:
   - For each emotion, provide specific evidence from the text that supports or contradicts its presence
   - Consider both explicit emotional words and contextual implications
   - Base your analysis on linguistic patterns, word choice, and context
   - Be objective in your analysis

2. Then, based on your analysis, provide your final classification in the format: [emotion1, emotion2, ...]
   - Only include emotions that are clearly present
   - Use only these emotions: joy, fear, anger, sadness, disgust, surprise, none
   - Do not explain your choice, just provide the list

Format your response EXACTLY as follows:

Explanation:
(Your detailed analysis here)

Final Classification:
[emotion1, emotion2, ...]   

Here are some similar examples to help guide your analysis:
{examples}"""

def process_emotions(text, vector_store, system_prompt=None):
    """Process text and return emotion predictions and reasoning using RAG."""
    try:
        # Get similar examples
        examples = get_similar_examples(vector_store, text)
        
        # Use combined prompt
        if system_prompt is None:
            system_prompt = get_combined_prompt(examples)
        
        cache_key = get_cache_key(text, system_prompt)
        cached_response = get_cached_response(cache_key)
        
        if cached_response and isinstance(cached_response, dict):
            return cached_response
        
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Analyze this text and provide both the evidence analysis and final classification: {text}"}
            ]
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract emotions from the response
        if '[' not in response_text or ']' not in response_text:
            pred_emotions = ['none']
            reasoning = response_text
        else:
            # Get the last bracketed list in the response (final classification)
            last_bracket_start = response_text.rindex('[')
            last_bracket_end = response_text.rindex(']')
            
            # Split reasoning and classification
            reasoning = response_text[:last_bracket_start].strip()
            emotions_str = response_text[last_bracket_start+1:last_bracket_end]
            
            pred_emotions = [e.strip().lower() for e in emotions_str.split(',') if e.strip()]
            if not pred_emotions:
                pred_emotions = ['none']
            
            # Validate emotions
            valid_emotions = {'joy', 'fear', 'anger', 'sadness', 'disgust', 'surprise', 'none'}
            pred_emotions = [e for e in pred_emotions if e in valid_emotions]
            if not pred_emotions:
                pred_emotions = ['none']
        
        result = {
            'emotions': pred_emotions,
            'reasoning': reasoning
        }
        
        cache_response(cache_key, result)
        return result
        
    except Exception as e:
        print(f"Error processing emotions: {e}")
        return {'emotions': ['none'], 'reasoning': f"Error: {str(e)}"}

def classify_emotions_batch(input_csv, output_csv, training_csv='train.csv'):
    """Process a batch of texts from CSV using RAG and save emotion classifications."""
    try:
        # Load data
        df = pd.read_csv(input_csv)
        df_copy = df.copy()
        
        # Create vector store from training data
        print("Creating vector store from training data...")
        vector_store = create_vector_store(training_csv)
        
        # Initialize columns if they don't exist
        emotion_columns = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'surprise']
        for col in emotion_columns + ['reasoning']:
            if col not in df_copy.columns:
                df_copy[col] = 0  # Initialize with 0 for emotions, will be updated
        
        # Process all texts
        print("Processing texts...")
        for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy)):
            try:
                result = process_emotions(row['text'], vector_store)
                if not isinstance(result, dict):
                    print(f"Warning: Unexpected result type for row {idx}: {type(result)}")
                    result = {'emotions': ['none'], 'reasoning': f"Error: Invalid result type {type(result)}"}
                
                pred_emotions = result.get('emotions', ['none'])
                reasoning = result.get('reasoning', 'Error: No reasoning provided')
                
                # Update emotions (one by one)
                for emotion in emotion_columns:
                    df_copy.at[idx, emotion] = 1 if emotion in pred_emotions else 0
                
                # Update reasoning
                df_copy.at[idx, 'reasoning'] = reasoning
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                # Set default values for failed rows
                for emotion in emotion_columns:
                    df_copy.at[idx, emotion] = 0
                df_copy.at[idx, 'reasoning'] = f"Error: {str(e)}"
        
        # Save results
        output_df = df_copy.drop(columns=['text'])
        if 'id' in output_df.columns:
            output_df = output_df.set_index('id')
        output_df.to_csv(output_csv)
        print(f"Results saved to {output_csv}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise  # Re-raise the exception to see the full traceback

def chat_with_emotions(training_csv='train.csv'):
    """Interactive chat with RAG-based emotion classification."""
    print("Initializing vector store from training data...")
    vector_store = create_vector_store(training_csv)
    
    print("🤖 RAG-based Emotion Classification Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n🤖 Goodbye! Have a great day!")
            break
            
        try:
            # Get emotions using RAG
            result = process_emotions(user_input, vector_store)
            print("\n🤖 Analysis:", result['reasoning'])
            print("\n🤖 Detected emotions:", ', '.join(result['emotions']))
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_to_classify_path', type=str, required=True)
    parser.add_argument('--output_csv_prediction_path', type=str, required=True)
    parser.add_argument('--reference_csv_path', type=str, required=True)
    args = parser.parse_args()

    input_csv = args.input_csv_to_classify_path
    output_csv = args.output_csv_prediction_path
    reference_csv = args.reference_csv_path
    classify_emotions_batch(input_csv, output_csv, reference_csv)