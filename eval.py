# %%
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from bert_score import score
import warnings

warnings.filterwarnings('ignore')

# %%
# 讀取 CSV 文件
csv_file = "path"  # 請將此處替換為您的 CSV 文件路徑
data = pd.read_csv(csv_file)

# %%
# 初始化模型和分數計算器
model_name = 'gpt2'  # 這裡使用 GPT-2 作為例子，可以根據需要更換
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 計算 PPL 的函數
def calculate_ppl(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    lls = []

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

# %%
def calculate_bertscore(reference_text, candidate_text, lang='zh'):
    # Calculate precision, recall, and F1 scores using BERTScore
    P, R, F1 = score([candidate_text], [reference_text], lang=lang)

    return {
        'BERTScore': {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }
    }

# %%
def calculate_rouge(reference_text, candidate_text):
    # Tokenize the reference and candidate texts
    reference_unigrams = split_into_unigrams(reference_text)
    candidate_unigrams = split_into_unigrams(candidate_text)
    
    reference_bigrams = split_into_bigrams(reference_text)
    candidate_bigrams = split_into_bigrams(candidate_text)

    def precision_recall_f1(reference_ngrams, candidate_ngrams):
        # Count n-grams
        reference_counts = {}
        candidate_counts = {}
        common_counts = {}

        for ngram in candidate_ngrams:
            candidate_counts[ngram] = candidate_counts.get(ngram, 0) + 1
        
        for ngram in reference_ngrams:
            reference_counts[ngram] = reference_counts.get(ngram, 0) + 1
        
        for ngram in candidate_counts:
            if ngram in reference_counts:
                common_counts[ngram] = min(reference_counts[ngram], candidate_counts[ngram])
        
        # Calculate precision, recall, and F1 score
        precision = sum(common_counts.values()) / max(len(candidate_ngrams), 1)
        recall = sum(common_counts.values()) / max(len(reference_ngrams), 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

        return precision, recall, f1

    # Calculate ROUGE-1 scores (unigram)
    precision_1, recall_1, f1_1 = precision_recall_f1(reference_unigrams, candidate_unigrams)
    
    # Calculate ROUGE-2 scores (bigram)
    precision_2, recall_2, f1_2 = precision_recall_f1(reference_bigrams, candidate_bigrams)

    return {
        'ROUGE-1': {'precision': precision_1, 'recall': recall_1, 'f1': f1_1},
        'ROUGE-2': {'precision': precision_2, 'recall': recall_2, 'f1': f1_2}
    }

# %%
# 初始化累積計數器
total_bleu_1 = 0
total_bleu_2 = 0
total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
total_ppl = 0
total_bert = 0

gtotal_bleu_1 = 0
gtotal_bleu_2 = 0
gtotal_rouge_1 = 0
gtotal_rouge_2 = 0
gtotal_rouge_l = 0
gtotal_ppl = 0
gtotal_bert = 0

num_rows = len(data)

def split_into_unigrams(text):
    return [char for char in text]

def split_into_bigrams(text):
    return [text[i:i+2] for i in range(len(text) - 1)]

for index, row in data.iterrows():
    reference_text = row['Counselor']
    candidate_text = row['NYCUKA']
    gpt4o_text = row['GPT-4o']

    # Split the texts into unigrams and bigrams
    reference_unigrams = split_into_unigrams(reference_text)
    candidate_unigrams = split_into_unigrams(candidate_text)
    gpt4o_unigrams = split_into_unigrams(gpt4o_text)

    reference_bigrams = split_into_bigrams(reference_text)
    candidate_bigrams = split_into_bigrams(candidate_text)
    gpt4o_bigrams = split_into_bigrams(gpt4o_text)

    # Calculate BLEU-1 score (unigram precision)
    bleu_1 = sentence_bleu([reference_unigrams], candidate_unigrams, weights=(1, 0, 0, 0), smoothing_function=smooth)
    gpt_1 = sentence_bleu([reference_unigrams], gpt4o_unigrams, weights=(1, 0, 0, 0), smoothing_function=smooth)

    # Calculate BLEU-2 score (bigram precision)
    bleu_2 = sentence_bleu([reference_bigrams], candidate_bigrams, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    gpt_2 = sentence_bleu([reference_bigrams], gpt4o_bigrams, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    
    rouge_scores = calculate_rouge(reference_text, candidate_text)
    gpt_rouge_scores = calculate_rouge(reference_text, gpt4o_text)
    
    # 計算 PPL
    ppl = calculate_ppl(model, tokenizer, candidate_text)
    gpt_ppl = calculate_ppl(model, tokenizer, gpt4o_text)

    bert_scores = calculate_bertscore(reference_text, candidate_text)
    gpt_bert_scores = calculate_bertscore(reference_text, gpt4o_text)
    
    
    # 累加分數
    total_bleu_1 += bleu_1
    total_bleu_2 += bleu_2
    total_rouge_1 += rouge_scores['ROUGE-1']['f1']
    total_rouge_2 += rouge_scores['ROUGE-2']['f1']
    total_ppl += ppl
    total_bert += bert_scores['BERTScore']['f1']

    gtotal_bleu_1 += gpt_1
    gtotal_bleu_2 += gpt_2
    gtotal_rouge_1 += gpt_rouge_scores['ROUGE-1']['f1']
    gtotal_rouge_2 += gpt_rouge_scores['ROUGE-2']['f1']
    gtotal_ppl += gpt_ppl
    gtotal_bert += gpt_bert_scores['BERTScore']['f1']

# %%
# 計算平均分數
average_bleu_1 = total_bleu_1 / num_rows
average_bleu_2 = total_bleu_2 / num_rows
average_rouge_1 = total_rouge_1 / num_rows
average_rouge_2 = total_rouge_2 / num_rows
average_ppl = total_ppl / num_rows
average_bert = total_bert / num_rows

gaverage_bleu_1 = gtotal_bleu_1 / num_rows
gaverage_bleu_2 = gtotal_bleu_2 / num_rows
gaverage_rouge_1 = gtotal_rouge_1 / num_rows
gaverage_rouge_2 = gtotal_rouge_2 / num_rows
gaverage_ppl = gtotal_ppl / num_rows
gaverage_bert = gtotal_bert / num_rows

# 印出結果
print(f"nycuka average BLEU-1: {average_bleu_1:.4f}")
print(f"nycuka average BLEU-2: {average_bleu_2:.4f}")
print(f"nycuka average ROUGE-1: {average_rouge_1:.4f}")
print(f"nycuka average ROUGE-2: {average_rouge_2:.4f}")
print(f"nycuka average PPL: {average_ppl:.4f}")
print(f"nycuka average BERT SCORE: {average_bert:.4f}")
print("")
print(f"4o average BLEU-1: {gaverage_bleu_1:.4f}")
print(f"4o average BLEU-2: {gaverage_bleu_2:.4f}")
print(f"4o average ROUGE-1: {gaverage_rouge_1:.4f}")
print(f"4o average ROUGE-2: {gaverage_rouge_2:.4f}")
print(f"4o average PPL: {gaverage_ppl:.4f}")
print(f"4o average BERT SCORE: {gaverage_bert:.4f}")


