import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import torch,os
import gc
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from pytorch_lightning import seed_everythingmodel_dirmodel_dir
from evaluate import load
bertscore = load("bertscore")



 
def generate_response( prompt, model, tokenizer):
        inputs = [prompt]
        inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt")
        
        output = model.generate(inputs.input_ids.to(map_location),
                                num_beams=3,
                                max_length=256,
                                no_repeat_ngram_size=2,
                                min_length=100,
                                top_k=50,
                                top_p=0.95,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=True)
        
        # print(tokenizer.decode(output[0], skip_special_tokens=True))
        
        # response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response[response.find('Report: ')+8:]
        return response

sample_size = '100k'
model_type = "large"
model_name = sample_size+"_gpt2_"+model_type
model_dir = "./models/"+model_name
 

Table = pd.DataFrame()
        
with torch.no_grad():

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    map_location = "cuda"
    device = torch.device(map_location)
    model.to(device)
    
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
   
    for split in range(1,3):
        if split==1:
            test_data = pd.read_csv('./data/Reports/'+sample_size+'_samples/test_split_96.csv')
            filename = 'FT_test'
        else:
            test_data = pd.read_csv('./data/Reports/real_reports.csv')
            filename = 'GT_test'
            
            
        print()
        print('Test on '+filename+' with 96 reports with gpt2-'+model_type+' model'+' trained on '+sample_size+' samples')
        # Generate responses
        generated_responses = []
        for question in test_data["Instructions"]:
            prompt = "Instruction: "+question
            response = generate_response(prompt,model, tokenizer)
            generated_responses.append(response)

        # Compute evaluation metrics
        ground_truth_answers = test_data["Report"]

        
        response = generate_response(prompt,model, tokenizer)
        
            
        
        #df.to_csv("data/Alzheimer/generated_responses.csv")
        # Accuracy
        #accuracy = accuracy_score(ground_truth_answers, generated_responses)
        #print("Accuracy:", accuracy)
        
        # ROUGE
        # Compute ROUGE evaluation for each generated response
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(gen_response, true_answer) for gen_response, true_answer in zip(generated_responses, ground_truth_answers)]
        
        # Calculate average ROUGE scores
        # avg_rouge1_precision = sum(score['rouge1'].precision for score in rouge_scores) / len(rouge_scores)
        # avg_rouge1_recall = sum(score['rouge1'].recall for score in rouge_scores) / len(rouge_scores)
        r1_f1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        # avg_rouge2_precision = sum(score['rouge2'].precision for score in rouge_scores) / len(rouge_scores)
        # avg_rouge2_recall = sum(score['rouge2'].recall for score in rouge_scores) / len(rouge_scores)
        r2_f1 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        # avg_rougeL_precision = sum(score['rougeL'].precision for score in rouge_scores) / len(rouge_scores)
        # avg_rougeL_recall = sum(score['rougeL'].recall for score in rouge_scores) / len(rouge_scores)
        rL_f1 = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        avgBLEU = []
        avgBERT = []
        for i in range(len(generated_responses)):
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([ ground_truth_answers[i]], generated_responses[i], weights = (0.5, 0.5))
            BERTscr = bertscore.compute(predictions=[generated_responses[i]], references=[ground_truth_answers[i]], lang="en")
            avgBLEU.append(BLEUscore)
            avgBERT.append(BERTscr['f1'][0])
        avgBLEU = np.mean(np.asarray(avgBLEU))
        avgBERT = np.mean(np.asarray(avgBERT))
        
        print('-')
        print('Split',split)
        # Print average ROUGE scores
        # print("ROUGE-1 Precision:", avg_rouge1_precision)
        # print("ROUGE-1 Recall:", avg_rouge1_recall)
        print("ROUGE-1 F-Measure:", r1_f1)
        
        # print("ROUGE-2 Precision:", avg_rouge2_precision)
        # print("ROUGE-2 Recall:", avg_rouge2_recall)
        print("ROUGE-2 F-Measure:", r2_f1)
        
        # print("ROUGE-L Precision:", avg_rougeL_precision)
        # print("ROUGE-L Recall:", avg_rougeL_recall)
        print("ROUGE-L F-Measure:", rL_f1)
        
        print("BLEU F-Measure:", avgBLEU)
        
        print("BERT F-Measure:", avgBERT)
        print()
        print('*'*50)
        print(model_name)
        
        df = np.hstack([filename,r1_f1,r2_f1,rL_f1,avgBLEU,avgBERT])
        Table = pd.concat([pd.DataFrame(df).T,Table])
        
        # for i in generated_responses:
        #     print(i)
        #     print()
        #     print('*'*50)
        #     print()
    
    cols = {0:'Splits',
            1:'ROGUE-1',
            2:'ROGUE-2',
            3:'ROGUE-L',
            4:'BLEU',
            5:'BERTScore'
            }
    #
    Table = Table.rename(columns=cols)
    
    path_save = './results_LLM'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    Table.to_excel(path_save+'/'+sample_size+'_gpt2_'+model_type+'.xlsx',index=False)
       

    df= pd.DataFrame(columns = ["Instructions", "generated_answer", "true_answer"])
    
    
    # # Generate responses
    # generated_responses = []
    # for question in test_data["Instructions"]:
    #     prompt = "Instruction: "+question
    #     response = generate_response(prompt,model, tokenizer)
    #     generated_responses.append(response)
    
    # Compute evaluation metrics
    ground_truth_answers = test_data["Report"]
    df["Instructions"] = test_data["Instructions"]
    df["generated_answer"] = generated_responses
    df["true_answer"] = ground_truth_answers
    
    path_save = './results_generated'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    df.to_excel(path_save+'/'+sample_size+'_gpt2_'+model_type+'.xlsx',index=False)
    