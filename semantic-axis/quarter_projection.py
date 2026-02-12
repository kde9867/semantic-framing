import pandas as pd
import numpy as np
import pickle
import torch
from typing import List, Tuple, Optional, Dict
import os
import re
import glob
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm


TARGET_WORDS = [
    '출산', '육아', '아이', '아기', '임신', '엄마', '결혼', '애', '남편', '신생아', 
    '병원', '저출산', '아빠', '지원', '산모', '정책', '개월', '분만', '아가', '자녀', 
    '임산부', '돌', '둘째', '부모', '인구', '조리원', '부부', '가족', '수유', '첫째', 
    '정부', '애기', '기저귀', '출산율', '교육', '분유', '가정', '검사', '딸', '여성', 
    '산후', '휴직', '육아휴직', '젖병', '태동', '출생', '아들', '혼인', '초음파', '부모님', 
    '입원', '진통', '태명', '자연분만', '제왕절개', '통증', '수술', '생후', '신혼부부', '물티슈', 
    '이유식', '비혼', '아내', '통계', '제도', '고령화', '양육', '연애', '출생아', '애들', 
    '남성', '내진', '통계청', '복지', '가사', '아동', '출산률', '태동검사', '산부인과', '돌봄', 
    '유도분만', '독박육아', '주택', '저출생', '합계출산율', '가구', '주거', '출생신고', '맞벌이', '혼인신고', 
    '집안일', '쌍둥이', '촉진제', '이혼', '모유', '어린이집', '모유수유', '의사', '산후조리', '유치원', 
    '태아', '전문의', '젖', '산후조리원', '의학', '아이들', '태교', '진료', '소아과'
]

MODELS_BASE_PATH = ".path/model/quarter_model/"
TEST_DATA_PATH = ".path/semantic-framing/data/all_ranges_test.csv"
AXIS_PATH = ".path/semantic-framing/semantic-axis/micro_macro_axis_chunks_balanced.pkl"
SAVE_DIR = ".path/semantic-framing/semantic-axis/quarterly_projections/"

SAMPLE_SIZE = 20000
SLIDING_WINDOW_STRIDE = 256
TOP_K_START = 2  
TOP_K_END = 6

CHECKPOINT_FREQUENCY = 5
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def collect_quarterly_models(base_path: str) -> Dict[str, Dict[str, str]]:
    """Automatically collect quarterly model paths"""

    quarterly_models = defaultdict(dict)
    
    for range_name in ['range1', 'range2', 'range3']:
        pattern = os.path.join(base_path, f"{range_name}_*Q*_*")
        model_dirs = glob.glob(pattern)
        
        for model_dir in sorted(model_dirs):
            dir_name = os.path.basename(model_dir)
            
            match = re.search(r'(\d{4}Q\d)', dir_name)
            if match:
                quarter = match.group(1)
                
                final_path = os.path.join(model_dir, 'final_model')
                best_path = os.path.join(model_dir, 'best_model')
                
                if os.path.exists(final_path):
                    quarterly_models[range_name][quarter] = final_path
                elif os.path.exists(best_path):
                    quarterly_models[range_name][quarter] = best_path
    
    total_models = sum(len(q) for q in quarterly_models.values())
    print(f"Total {total_models} models found")
    for range_name in sorted(quarterly_models.keys()):
        quarters = sorted(quarterly_models[range_name].keys())
        print(f"  {range_name}: {len(quarters)} quarters ({quarters[0]} ~ {quarters[-1]})")
    print("="*80)
    
    return dict(quarterly_models)



def save_checkpoint(
    all_results: List[Dict],
    timestamp: str,
    model_count: int,
    total_models: int,
    checkpoint_dir: str
):
    """Save intermediate results"""
    checkpoint_name = f"checkpoint_{timestamp}_model_{model_count}of{total_models}.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'metadata': {
                'timestamp': timestamp,
                'checkpoint_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models_processed': model_count,
                'total_models': total_models,
                'total_results': len(all_results)
            }
        }, f)
    

def load_latest_checkpoint(checkpoint_dir: str, timestamp: str) -> Tuple[List[Dict], int]:
    """Load most recent checkpoint"""
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}_model_*.pkl")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return [], 0
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    
    print(f"\nCheckpoint found: {os.path.basename(latest_checkpoint)}")
    
    with open(latest_checkpoint, 'rb') as f:
        data = pickle.load(f)
    
    results = data['results']
    models_processed = data['metadata']['models_processed']
    
    return results, models_processed


def get_processed_models(all_results: List[Dict]) -> set:
    """Extract already processed (range, quarter) combinations"""
    processed = set()
    for result in all_results:
        processed.add((result['range'], result['quarter']))
    return processed


def is_valid_prediction(word: str, target_word: str) -> bool:
    """Check if predicted word is valid"""
    word = word.strip()
    
    if not word:
        return False
    if word.startswith('##'):
        return False
    if word.startswith('[') and word.endswith(']'):
        return False
    if word == target_word:
        return False
    if word in ['.', ',', '·', '?', '!', ':', ';', '-', '~', '/', '\\']:
        return False
    if re.match(r'^[^\w가-힣]+$', word):
        return False
    if len(word) < 2 and not re.search('[가-힣]', word):
        return False
    if re.search('[가-힣]', word) or (len(word) >= 2 and word.isalpha()):
        return True
    
    return False


def get_word_embeddings_from_model(
    words: List[str], 
    tokenizer, 
    model, 
    device
) -> List[np.ndarray]:
    """Extract word embeddings directly from quarterly model in batch"""
    if not words:
        return []
    
    embeddings = []
    
    batch_size = 32
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i+batch_size]
        
        inputs = tokenizer(
            batch_words, 
            return_tensors='pt', 
            padding=True,
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.bert(**inputs, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            embeddings.extend(batch_embeddings)
    
    return embeddings


def predict_masked_words(
    text: str,
    target_word: str,
    tokenizer,
    model,
    device
) -> Tuple[List[List[str]], bool]:
    """Predict masked words using period-specific model with sliding window"""
    masked_text = text.replace(target_word, tokenizer.mask_token)
    
    inputs_raw = tokenizer(
        masked_text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=512,
        stride=SLIDING_WINDOW_STRIDE,
        return_overflowing_tokens=True
    )
    
    model_inputs = {
        k: v.to(device) 
        for k, v in inputs_raw.items() 
        if k != 'overflow_to_sample_mapping'
    }
    
    if model_inputs['input_ids'].shape[0] == 0:
        return [], False
    
    all_predictions = []
    is_correct = False
    mask_token_id = tokenizer.mask_token_id
    
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits
    
    for i in range(logits.shape[0]):
        window_ids = model_inputs['input_ids'][i]
        window_logits = logits[i]
        
        mask_positions = (window_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        for pos in mask_positions:
            top_10_tokens = torch.topk(window_logits[pos], k=10).indices.tolist()
            top_10_words = [tokenizer.decode([t]).strip() for t in top_10_tokens]
            
            if target_word in top_10_words[:5]:
                is_correct = True
            
            valid_words = [
                w for w in top_10_words[TOP_K_START-1:TOP_K_END] 
                if is_valid_prediction(w, target_word)
            ]
            
            if valid_words:
                all_predictions.append(valid_words)
    
    return all_predictions, is_correct


def project_to_axis(embedding: np.ndarray, axis_vector: np.ndarray) -> float:
    """Project embedding onto axis using cosine similarity"""
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    axis_norm = axis_vector / (np.linalg.norm(axis_vector) + 1e-8)
    return float(np.dot(emb_norm, axis_norm))


def analyze_single_word(
    target_word: str,
    range_name: str,
    quarter: str,
    tokenizer,
    model,
    test_df: pd.DataFrame,
    axis_vector: np.ndarray,
    device,
    sample_size: int = 20000
) -> Optional[Dict]:
    """Analyze single keyword with specific quarterly model (direct embedding extraction)"""
    
    word_df = test_df[test_df['content'].str.contains(target_word, na=False)].copy()
    
    if len(word_df) == 0:
        return None
    
    sample_texts = word_df['content'].head(sample_size).tolist()
    
    results = {
        'projections': [],
        'predicted_words': [],
        'is_correct': [],
        'mask_count': []
    }
    
    correct_count = 0
    total_count = 0
    
    for text in sample_texts:
        if target_word not in text:
            continue
        
        all_mask_preds, is_correct = predict_masked_words(
            text, target_word, tokenizer, model, device
        )
        
        total_count += 1
        if is_correct:
            correct_count += 1
        
        if not all_mask_preds:
            continue
        
        all_words_for_embedding = []
        for mask_preds in all_mask_preds:
            all_words_for_embedding.extend(mask_preds)
        
        if all_words_for_embedding:
            word_embeddings = get_word_embeddings_from_model(
                all_words_for_embedding, tokenizer, model, device
            )
            
            if word_embeddings:
                avg_embedding = np.mean(word_embeddings, axis=0)
                projection = project_to_axis(avg_embedding, axis_vector)
                
                results['projections'].append(projection)
                results['predicted_words'].append(all_mask_preds)
                results['is_correct'].append(is_correct)
                results['mask_count'].append(len(all_mask_preds))
    
    if not results['projections']:
        return None
    
    projections = np.array(results['projections'])
    
    return {
        'target_word': target_word,
        'range': range_name,
        'quarter': quarter,
        'mean_projection': float(projections.mean()),
        'std_projection': float(projections.std()),
        'median_projection': float(np.median(projections)),
        'min_projection': float(projections.min()),
        'max_projection': float(projections.max()),
        'count': len(projections),
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'total_sentences': total_count,
        'correct_predictions': correct_count,
        'projections': results['projections'],
        'predicted_words': results['predicted_words'],
        'is_correct': results['is_correct'],
        'mask_count': results['mask_count']
    }


def main():
    """Main execution function - Ultra-optimized + Checkpoint version"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load axis
    with open(AXIS_PATH, 'rb') as f:
        axis_data = pickle.load(f)
        axis_vector = axis_data['axis_vector']
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect quarterly models
    quarterly_models = collect_quarterly_models(MODELS_BASE_PATH)
    total_models = sum(len(q) for q in quarterly_models.values())
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Check and load checkpoint
    all_results, models_processed = load_latest_checkpoint(CHECKPOINT_DIR, timestamp)
    processed_models = get_processed_models(all_results)
    
    if models_processed > 0:
        print(f"   Resuming from: {models_processed}/{total_models} models")
    else:
        print("   Starting new analysis")
    
    model_count = models_processed
    
    for range_name in sorted(quarterly_models.keys()):
        quarters_dict = quarterly_models[range_name]
        
        for quarter in sorted(quarters_dict.keys()):
            if (range_name, quarter) in processed_models:
                continue
            
            model_count += 1
            model_path = quarters_dict[quarter]

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = BertForMaskedLM.from_pretrained(model_path)
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
                continue

            quarter_results = []
            
            for target_word in tqdm(TARGET_WORDS, desc=f"   {quarter}", ncols=100, leave=False):
                try:
                    result = analyze_single_word(
                        target_word=target_word,
                        range_name=range_name,
                        quarter=quarter,
                        tokenizer=tokenizer,
                        model=model,
                        test_df=test_df,
                        axis_vector=axis_vector,
                        device=device,
                        sample_size=SAMPLE_SIZE
                    )
                    
                    if result is not None:
                        quarter_results.append(result)
                
                except Exception as e:
                    continue
            
            all_results.extend(quarter_results)
            
            del model, tokenizer
            torch.cuda.empty_cache()
            
            print(f"Completed: {len(quarter_results)}/{len(TARGET_WORDS)} keywords")
            
            if model_count % CHECKPOINT_FREQUENCY == 0:
                save_checkpoint(all_results, timestamp, model_count, total_models, CHECKPOINT_DIR)
    

    csv_rows = []
    for r in all_results:
        csv_rows.append({
            'target_word': r['target_word'],
            'range': r['range'],
            'quarter': r['quarter'],
            'mean_projection': r['mean_projection'],
            'std_projection': r['std_projection'],
            'median_projection': r['median_projection'],
            'min_projection': r['min_projection'],
            'max_projection': r['max_projection'],
            'count': r['count'],
            'accuracy': r['accuracy'],
            'total_sentences': r['total_sentences'],
            'correct_predictions': r['correct_predictions']
        })
    
    summary_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(SAVE_DIR, f'quarterly_summary_direct_{timestamp}.csv')
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV saved: {csv_path}")
    
    pkl_path = os.path.join(SAVE_DIR, f'quarterly_all_results_direct_{timestamp}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'metadata': {
                'timestamp': timestamp,
                'target_words': TARGET_WORDS,
                'quarterly_models': quarterly_models,
                'axis_path': AXIS_PATH,
                'embedding_method': 'direct_from_quarterly_model',
                'sample_size': SAMPLE_SIZE,
                'top_k_range': f'{TOP_K_START}-{TOP_K_END}',
                'total_analyses': len(all_results),
                'checkpoint_frequency': CHECKPOINT_FREQUENCY
            }
        }, f)


if __name__ == "__main__":
    main()