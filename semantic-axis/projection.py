import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import os
import re
from collections import Counter
from datetime import datetime

# Korean font settings
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


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

SAMPLE_SIZE = 20000
SAVE_DIR = ".path/semantic-framing/semantic-axis/"
SAVE_DIR_OUTPUT = ".path/semantic-framing/semantic-axis/projections/"
DIAGNOSTIC_MODE = True

os.makedirs(SAVE_DIR_OUTPUT, exist_ok=True)

EMBEDDING_MODEL_PATH = ".path/model/klue-bert-mlm-20251213_123829/best_model"

SLIDING_WINDOW_STRIDE = 256

TOP_K_START = 2  
TOP_K_END = 6    


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


def get_word_embedding(word: str, tokenizer, model, device) -> Optional[np.ndarray]:
    """Extract word embedding using Birth-Klue* model"""
    text = f"{word}"
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        word_embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

    return word_embedding


def predict_masked_words(
    text: str,
    target_word: str,
    pred_tokenizer,
    pred_model,
    device,
    top_k_start: int = 2,
    top_k_end: int = 6
) -> Tuple[List[List[str]], bool]:
    """Predict masked words using period-specific model with sliding window"""
    masked_text = text.replace(target_word, pred_tokenizer.mask_token)

    pred_inputs_raw = pred_tokenizer(
        masked_text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=512,
        stride=SLIDING_WINDOW_STRIDE,
        return_overflowing_tokens=True
    )

    model_inputs = {}
    for k, v in pred_inputs_raw.items():
        if k != 'overflow_to_sample_mapping':
            model_inputs[k] = v.to(device)

    mask_token_id = pred_tokenizer.mask_token_id
    input_ids = model_inputs['input_ids']

    if input_ids.shape[0] == 0:
        return [], False

    all_mask_predictions = []
    is_correct_any = False

    with torch.no_grad():
        outputs = pred_model(**model_inputs)
        all_logits = outputs.logits

    num_windows = all_logits.shape[0]
    for i in range(num_windows):
        window_input_ids = input_ids[i]
        window_logits = all_logits[i]

        mask_positions = (window_input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            continue

        for pos in mask_positions:
            pos_logits = window_logits[pos, :]

            top_10_tokens = torch.topk(pos_logits, k=10).indices.tolist()
            top_10_words = [pred_tokenizer.decode([t]).strip() for t in top_10_tokens]

            if target_word in top_10_words[:5]:
                is_correct_any = True

            top_2_to_6_words = []
            for rank_idx in range(top_k_start - 1, top_k_end):
                if rank_idx < len(top_10_words):
                    word = top_10_words[rank_idx]
                    if is_valid_prediction(word, target_word):
                        top_2_to_6_words.append(word)
            
            if len(top_2_to_6_words) > 0:
                all_mask_predictions.append(top_2_to_6_words)

    return all_mask_predictions, is_correct_any


def project_to_axis(embedding: np.ndarray, axis_vector: np.ndarray) -> float:
    """Project embedding onto axis using cosine similarity"""
    embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    axis_norm = axis_vector / (np.linalg.norm(axis_vector) + 1e-8)
    cosine_sim = np.dot(embedding_norm, axis_norm)
    return float(cosine_sim)


def analyze_single_word(
    target_word: str,
    test_df: pd.DataFrame,
    model_configs: Dict,
    axis_vector: np.ndarray,
    k_star_tokenizer,
    k_star_model,
    device,
    sample_size: int = 20000,
    diagnostic_mode: bool = True
) -> Optional[Dict]:
    """Perform complete analysis for a single word"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing word: '{target_word}'")
    print(f"{'='*80}")
    
    word_df = test_df[test_df['content'].str.contains(target_word, na=False)].copy()
    
    if len(word_df) == 0:
        return None
    
    sample_texts = word_df['content'].head(sample_size).tolist()
    
    results = {}
    period_word_counters = {period: Counter() for period in model_configs.keys()}
    
    for period, config in model_configs.items():
        print(f"\n--- Analyzing {config['label']} ---")
        
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = BertForMaskedLM.from_pretrained(config['path'])
        model.to(device)
        model.eval()
        
        period_results = {
            'projections': [],
            'texts': [],
            'predicted_words': [],
            'embeddings_norm': [],
            'is_correct': [],
            'mask_count': []
        }
        
        correct_count = 0
        total_count = 0
        diagnostic_samples = []
        
        for i, text in enumerate(sample_texts):
            if target_word not in text:
                continue
            
            all_mask_preds, is_correct = predict_masked_words(
                text=text,
                target_word=target_word,
                pred_tokenizer=tokenizer,
                pred_model=model,
                device=device,
                top_k_start=TOP_K_START,
                top_k_end=TOP_K_END
            )
            total_count += 1
            
            if is_correct:
                correct_count += 1
            
            if len(all_mask_preds) == 0:
                continue
            
            mask_projections = []
            
            for mask_preds in all_mask_preds:
                word_embeddings = []
                for word in mask_preds:
                    word_emb = get_word_embedding(word, k_star_tokenizer, k_star_model, device)
                    if word_emb is not None:
                        word_embeddings.append(word_emb)
                        period_word_counters[period][word] += 1
                
                if len(word_embeddings) == 0:
                    continue
                
                avg_embedding = np.mean(word_embeddings, axis=0)
                projection = project_to_axis(avg_embedding, axis_vector)
                mask_projections.append(projection)
            
            if len(mask_projections) == 0:
                continue
            
            final_projection = np.mean(mask_projections)
            
            all_words_flat = [word for mask_preds in all_mask_preds for word in mask_preds]
            all_embeddings = []
            for word in all_words_flat:
                emb = get_word_embedding(word, k_star_tokenizer, k_star_model, device)
                if emb is not None:
                    all_embeddings.append(emb)
            
            if len(all_embeddings) == 0:
                continue
            
            avg_all_embedding = np.mean(all_embeddings, axis=0)
            
            period_results['projections'].append(final_projection)
            period_results['texts'].append(text[:100] + '...')
            period_results['predicted_words'].append(all_mask_preds)
            period_results['embeddings_norm'].append(np.linalg.norm(avg_all_embedding))
            period_results['is_correct'].append(is_correct)
            period_results['mask_count'].append(len(all_mask_preds))
            
            if diagnostic_mode and len(diagnostic_samples) < 3:
                diagnostic_samples.append({
                    'text': text[:100],
                    'predictions': all_mask_preds,
                    'projection': final_projection,
                    'mask_count': len(all_mask_preds)
                })
        
        results[period] = period_results
        
        if len(period_results['projections']) > 0:
            projections = np.array(period_results['projections'])
            print(f"  Projection - Mean: {projections.mean():.4f} ± {projections.std():.4f}")
    
    return {
        'target_word': target_word,
        'results': results,
        'word_distributions': period_word_counters,
        'sample_count': len(sample_texts)
    }


def visualize_single_word(
    word_data: Dict,
    model_configs: Dict,
    save_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """Generate visualization for a single word"""
    
    target_word = word_data['target_word']
    results = word_data['results']
    
    all_projections = []
    for period in results.values():
        all_projections.extend(period['projections'])
    
    if len(all_projections) == 0:
        return None, None
    
    all_projections = np.array(all_projections)
    global_min, global_max = all_projections.min(), all_projections.max()
    
    max_abs = max(abs(global_min), abs(global_max))
    symmetric_min = -max_abs
    symmetric_max = max_abs
    global_L = symmetric_max - symmetric_min
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # First row: All data
    ax = axes[0]
    ax.barh(0, width=global_L*1.2, height=1, left=symmetric_min-global_L*0.1,
            edgecolor='k', fc=(1, 1, 1, 0), zorder=5)
    
    for period, config in model_configs.items():
        projections = results[period]['projections']
        color = config['color']
        for score in projections:
            ax.vlines(score, ymin=-0.5, ymax=0.5,
                        linestyle='-', color=color, linewidth=0.5, alpha=0.4)
    
    ax.vlines(0, ymin=-0.5, ymax=0.5, linestyle='--',
            color='black', linewidth=2, alpha=0.7, zorder=10)
    
    mean_score = all_projections.mean()
    ax.vlines(mean_score, ymin=-0.5, ymax=0.5, linestyle='-',
            color='red', linewidth=1.5, zorder=9)
    ax.scatter(mean_score, 0, marker='D', color='red', s=50, zorder=10)
    
    ax.set_xlim(symmetric_min-global_L*0.15, symmetric_max+global_L*0.15)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([0])
    ax.set_yticklabels(['Overall'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Cosine Similarity (Micro -  /  Macro +)', fontsize=11, fontweight='bold')
    
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Rows 2-4: Each model
    for ax_idx, (period, config) in enumerate(model_configs.items(), start=1):
        ax = axes[ax_idx]
        
        if len(results[period]['projections']) == 0:
            ax.text(0.5, 0, f"{config['label']}: No data",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xlim(symmetric_min-global_L*0.15, symmetric_max+global_L*0.15)
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([])
            if ax_idx == len(model_configs):
                ax.set_xlabel('Cosine Similarity (Micro -  /  Macro +)', fontsize=11, fontweight='bold')
            else:
                ax.set_xticks([])
            for spine in ['top', 'left', 'right', 'bottom']:
                ax.spines[spine].set_visible(False)
            continue
        
        projections = np.array(results[period]['projections'])
        
        ax.barh(0, width=global_L*1.2, height=1, left=symmetric_min-global_L*0.1,
                edgecolor='k', fc=(1, 1, 1, 0), zorder=5)
        
        for score in projections:
            ax.vlines(score, ymin=-0.5, ymax=0.5,
                        linestyle='-', color=config['color'], linewidth=1, alpha=0.6)
        
        ax.vlines(0, ymin=-0.5, ymax=0.5, linestyle='--',
                color='black', linewidth=2, alpha=0.7, zorder=10)
        
        mean_score_period = projections.mean()
        ax.vlines(mean_score_period, ymin=-0.5, ymax=0.5, linestyle='-',
                color='red', linewidth=1.5, zorder=9)
        ax.scatter(mean_score_period, 0, marker='D', color='red', s=50, zorder=10)
        
        ax.set_xlim(symmetric_min - global_L * 0.15, symmetric_max + global_L * 0.15)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([0])
        
        label_text = f'{config["label"]}\n(Mean: {mean_score_period:.4f})'
        ax.set_yticklabels([label_text], fontsize=10, fontweight='bold')
        
        if ax_idx == len(model_configs):
            ax.set_xlabel('Cosine Similarity (Micro -  /  Macro +)', fontsize=11, fontweight='bold')
        else:
            ax.set_xticks([])
        
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if ax_idx != len(model_configs):
            ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, f'spectrum_{target_word}_top2to6.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    return png_path, pdf_path


def save_single_word_results(
    word_data: Dict,
    model_configs: Dict,
    save_dir: str,
    axis_path: str,
    embedding_model_path: str
) -> Tuple[str, str]:
    """Save single word results as CSV and PKL"""
    
    target_word = word_data['target_word']
    results = word_data['results']
    
    # Save CSV
    all_results = []
    for period, config in model_configs.items():
        period_data = results[period]
        for i in range(len(period_data['projections'])):
            pred_words_str = ' | '.join([', '.join(mask_preds) for mask_preds in period_data['predicted_words'][i]])
            
            result_dict = {
                'target_word': target_word,
                'period': config['label'],
                'text': period_data['texts'][i],
                'projection_score': period_data['projections'][i],
                'predicted_words': pred_words_str,
                'mask_count': period_data['mask_count'][i],
                'embedding_norm': period_data['embeddings_norm'][i],
                'is_correct': period_data['is_correct'][i]
            }
            all_results.append(result_dict)
    
    csv_path = None
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(save_dir, f'projections_{target_word}_top2to6.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save PKL
    all_projections = []
    for period in results.values():
        all_projections.extend(period['projections'])
    
    all_projections = np.array(all_projections)
    global_min, global_max = all_projections.min(), all_projections.max()
    max_abs = max(abs(global_min), abs(global_max))
    
    pkl_data = {
        'target_word': target_word,
        'sample_size': word_data['sample_count'],
        'top_k_range': f'{TOP_K_START}-{TOP_K_END}',
        'embedding_model': embedding_model_path,
        'axis_path': axis_path,
        'method': 'top2to6_average_per_mask_micro_macro',
        'axis_direction': 'micro(-) to macro(+)',
        'symmetric_range': {
            'original_min': float(global_min),
            'original_max': float(global_max),
            'symmetric_min': float(-max_abs),
            'symmetric_max': float(max_abs)
        },
        'results': results,
        'word_distributions': {
            period: dict(counter.most_common(100))
            for period, counter in word_data['word_distributions'].items()
        },
        'statistics': {
            period: {
                'mean_projection': np.mean(results[period]['projections']) if len(results[period]['projections']) > 0 else None,
                'std_projection': np.std(results[period]['projections']) if len(results[period]['projections']) > 0 else None,
                'mean_mask_count': np.mean(results[period]['mask_count']) if len(results[period]['mask_count']) > 0 else None,
                'count': len(results[period]['projections']),
                'correct_count': sum(results[period]['is_correct']) if len(results[period]['is_correct']) > 0 else 0,
                'accuracy': sum(results[period]['is_correct']) / len(results[period]['is_correct']) if len(results[period]['is_correct']) > 0 else None
            }
            for period in model_configs.keys()
        }
    }
    
    pkl_path = os.path.join(save_dir, f'analysis_{target_word}_top2to6.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    
    return csv_path, pkl_path


def main():
    """Main execution function"""
    
    # Load axis data
    print("=== Loading Micro-Macro Axis ===")
    axis_path = os.path.join(SAVE_DIR, "micro_macro_axis_chunks_balanced.pkl")
    
    with open(axis_path, 'rb') as f:
        axis_data = pickle.load(f)
        axis_vector = axis_data['axis_vector']
        micro_mean = axis_data['micro_mean']
        macro_mean = axis_data['macro_mean']
    
    axis_norm = axis_vector / np.linalg.norm(axis_vector)
    micro_mean_norm = micro_mean / np.linalg.norm(micro_mean)
    macro_mean_norm = macro_mean / np.linalg.norm(macro_mean)
    
    micro_projection = np.dot(micro_mean_norm, axis_norm)
    macro_projection = np.dot(macro_mean_norm, axis_norm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load Birth-Klue* model
    print(f"\n=== Loading Birth-Klue* Model ===")
    k_star_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    k_star_model = BertModel.from_pretrained(EMBEDDING_MODEL_PATH)
    k_star_model.to(device)
    k_star_model.eval()
    
    # Period-specific model configuration
    model_base_path = ".path/"
    model_configs = {
        'range1': {
            'path': os.path.join(model_base_path, 'range1-20251213_154928/best_model'),
            'label': 'Pre-COVID (Range1)',
            'color': '#FFB04A'
        },
        'range2': {
            'path': os.path.join(model_base_path, 'range2-20251213_165302/best_model'),
            'label': 'During-COVID (Range2)',
            'color': '#0097A2'
        },
        'range3': {
            'path': os.path.join(model_base_path, 'range3-20251213_180228/best_model'),
            'label': 'Post-COVID (Range3)',
            'color': '#446081'
        }
    }
    
    # Load test data
    data_path = ".path/semantic-framing/data/all_ranges_test.csv"
    test_df = pd.read_csv(data_path)
    
    batch_summary = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, target_word in enumerate(TARGET_WORDS, 1):
        print(f"\n\n{'#'*80}")
        print(f"[{idx}/{len(TARGET_WORDS)}] Processing word: '{target_word}'")
        print(f"{'#'*80}")
        
        try:
            word_data = analyze_single_word(
                target_word=target_word,
                test_df=test_df,
                model_configs=model_configs,
                axis_vector=axis_vector,
                k_star_tokenizer=k_star_tokenizer,
                k_star_model=k_star_model,
                device=device,
                sample_size=SAMPLE_SIZE,
                diagnostic_mode=DIAGNOSTIC_MODE
            )
            
            if word_data is None:
                batch_summary.append({
                    'word': target_word,
                    'status': 'skipped',
                    'reason': 'no_data'
                })
                continue

            csv_path, pkl_path = save_single_word_results(
                word_data=word_data,
                model_configs=model_configs,
                save_dir=SAVE_DIR_OUTPUT,
                axis_path=axis_path,
                embedding_model_path=EMBEDDING_MODEL_PATH
            )
            
            summary_entry = {
                'word': target_word,
                'status': 'success',
                'sample_count': word_data['sample_count']
            }
            
            for period in model_configs.keys():
                if len(word_data['results'][period]['projections']) > 0:
                    projections = np.array(word_data['results'][period]['projections'])
                    summary_entry[f'{period}_mean'] = projections.mean()
                    summary_entry[f'{period}_std'] = projections.std()
                    summary_entry[f'{period}_count'] = len(projections)
                else:
                    summary_entry[f'{period}_mean'] = None
                    summary_entry[f'{period}_std'] = None
                    summary_entry[f'{period}_count'] = 0
            
            batch_summary.append(summary_entry)
             
        except Exception as e:
            import traceback
            traceback.print_exc()
            batch_summary.append({
                'word': target_word,
                'status': 'error',
                'error_message': str(e)
            })
            continue
    
    summary_df = pd.DataFrame(batch_summary)
    summary_csv_path = os.path.join(SAVE_DIR_OUTPUT, f'batch_summary_{timestamp}.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()