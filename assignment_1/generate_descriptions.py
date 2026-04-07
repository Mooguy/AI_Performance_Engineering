import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

from config import client, GEN_CONFIG
from judge import run_judge
from judge_sep import run_judge_isolated

# Token encoder (approximate for Llama)
enc = tiktoken.get_encoding("cl100k_base")

def build_system_prompt():
    return """You write concise product descriptions for an online catalog.

Write a clear, natural-sounding description based only on the provided product information.

Rules:
- Use only the provided product details
- Be friendly, credible, and positive
- Do not invent features, benefits, dimensions, or use cases
- Focus on what the product is and its relevant attributes
- Keep the description between 50 and 90 words
- Output only the final description
- Make it compelling for a customer ready to buy

Write ONLY the description. No extra text."""

system_prompt = build_system_prompt()

def calculate_cost(input_tokens, output_tokens):
    """Calculate total cost per call in USD (Meta-Llama-3.1-8B-Instruct eu-north1 pricing)."""
    input_cost_per_1k = 0.02 / 1000  # $0.02 per 1M = $0.00002 per 1K
    output_cost_per_1k = 0.06 / 1000  # $0.06 per 1M = $0.00006 per 1K
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    
    return total_cost

def generate_description(row, config, system_prompt):
    name = row["product_name"]
    attrs = row["Product_attribute_list"]
    material = row["material"]
    warranty = row["warranty"]

    user_prompt = f"""Product name: {name}
        Attributes: {attrs}
        Material: {material}
        Warranty: {warranty}"""

    start_time = time.time()

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        extra_body={"top_k": config["top_k"]}
    )

    latency_ms = (time.time() - start_time) * 1000
    description = response.choices[0].message.content.strip()

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_cost = calculate_cost(input_tokens, output_tokens)

    return {
        "generated_description": description,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        'cost_usd': total_cost,
        'length_words': len(description.split())
    }


def process_all_products(
    csv_file='Assignment_01_product_dataset.csv',
    output_file='assignment_01_task_5_03.xlsx',
    config=GEN_CONFIG
):
    """Process entire dataset and save Excel with blank rubric columns."""
    
    df = pd.read_csv(csv_file)
    system_prompt = build_system_prompt()
    
    results = []
    for idx, row in df.iterrows():
        print(f"Processing row {idx+1}/{len(df)}: {row['product_name']}")
        result = generate_description(row, config, system_prompt)
        result['index'] = idx  # Preserve original row index
        results.append(result)
    
    # Create full DataFrame with blank columns
    results_df = pd.DataFrame(results)
    full_df = df.merge(results_df, left_index=True, right_on='index').drop('index', axis=1)

    rubric_cols = ['fluency', 'grammar', 'tone', 'grounding',
                'fluency_explanation', 'grammar_explanation',
                'tone_explanation', 'grounding_explanation', 'length', 'latency', 'cost']
    for col in rubric_cols:
        full_df[col] = ''

    full_df['length'] = full_df['length_words'].apply(lambda x: 'good' if 50 <= x <= 90 else ('ok' if (40 <= x < 50) or (90 < x <= 110) else 'bad'))
    full_df['latency'] = full_df['latency_ms'].apply(lambda x: 'good' if x <= 5000 else ('ok' if 5000 < x <= 9000 else 'bad'))
    full_df['cost'] = full_df['cost_usd'].apply(lambda x: 'good' if x <= 0.0001 else ('ok' if 0.0001 < x <= 0.0005 else 'bad'))
    
    full_df.to_excel(output_file, index=False)
    print(f"\nSaved {len(full_df)} rows to {output_file}")
    return output_file, full_df

def pass_or_fail(row):
    """Calculate final score based on rubric criteria.
    No-go rules: 
        grounding ≠ good
        length = bad
        tone = bad

    Let G = # good, O = # ok, B = # bad
    Pass: G ≥ 3 ∧ O ≤ 2 ∧ B = 0
    """
    criteria = ['fluency', 'grammar', 'tone', 'grounding', 'length', 'latency', 'cost']
    verdicts = [row[crit] for crit in criteria]
    if row['grounding'] != 'good' or row['length'] == 'bad' or row['tone'] == 'bad':
        return 'fail'
    good_count = verdicts.count('good')
    ok_count = verdicts.count('ok')
    bad_count = verdicts.count('bad')
    if good_count >= 3 and ok_count <= 2 and bad_count == 0:
        return 'pass'
    else:
        return 'fail'   
    
def generate_final_score(df, final_output_file=None):
    """Apply pass/fail logic to entire DataFrame."""
    df['final_score'] = df.apply(pass_or_fail, axis=1)
    if final_output_file:
        df.to_excel(final_output_file, index=False)
    return df

def main():
    """run normal:"""
    # judge_input_file, df_all = process_all_products()
    # judge_input_file = 'assignment_01_task_5_03.xlsx'
    # judge_output_file = judge_input_file.replace('.xlsx', '_judged.xlsx')
    # df_judged = run_judge(input_file=judge_input_file, output_file=None)
    # final_output_file = judge_input_file.replace('.xlsx', '_final.xlsx')
    # df_final = generate_final_score(df_judged_isolated, final_output_file=final_output_file)

    """run isolated:"""
    # judge_input_file, df_all = process_all_products()
    judge_input_file = 'assignment_01_task_5_05.xlsx'
    # judge_output_file = judge_input_file.replace('.xlsx', '_judged.xlsx')
    df_judged_isolated = run_judge_isolated(input_file=judge_input_file, output_file=None)
    final_output_file = judge_input_file.replace('.xlsx', '_isolated_final.xlsx')
    df_final = generate_final_score(df_judged_isolated, final_output_file=final_output_file)

if __name__ == "__main__":
    main()
