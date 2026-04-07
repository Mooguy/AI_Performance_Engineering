from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

from config import client, JUDGE_CONFIG

class SingleCriterionScore(BaseModel):
    explanation: str
    verdict: Literal["good", "ok", "bad"]

# Template for all
CRITERION_PROMPT = """You are a strict evaluator of AI-generated e-commerce product descriptions.

Your job is to grade the description using the rubric below. Be conservative and evidence-based.

CRITERION: {criterion_name}
{criterion_definition}

SCORING RULES
- Be strict. Do not give "good" unless the description clearly meets the full definition of good.
- If the description is borderline between two ratings, choose the lower rating.
- Every criterion must include a specific explanation, even when the verdict is "good".
- Explanations must reference concrete wording from the description.
- Do not be polite or generous. Your role is accurate grading, not helpful writing.

PRODUCT INFORMATION
Product name: {product_name}
Attributes: {attributes}
Material: {material}
Warranty: {warranty}

DESCRIPTION TO EVALUATE
{generated_description}

- Explanations: MAXIMUM 25 WORDS, NO MORE."

Return valid JSON: {{"explanation": "...", "verdict": "good|ok|bad"}}
"""

# Specific definitions
CRITERIA = {
    "fluency": "good: Reads smoothly and naturally; no awkward phrasing. ok: Mostly clear with minor awkward phrasing. bad: Choppy or confusing; hard to follow. FLUENCY RULES: Ignore whether claims are factual. Only judge sentence flow and readability.",
    "grammar": "good: No noticeable spelling, grammar, or punctuation errors. ok: 1-2 minor errors that do not affect understanding. bad: Multiple errors that distract or cause confusion. GRAMMER RULES: Ignore whether claims are factual. Only judge grammar, punctuation, and spelling.",
    "tone": "good: Friendly, credible sales voice; positive and persuasive. ok: Generally appropriate but slightly too dry, casual, or overhyped. bad: Clearly inappropriate, such as negative, rude, sarcastic, robotic, or strongly unnatural. TONE RULES: Ignore whether claims are factual. Only judge the style and attitude of the writing.",
    "grounding": "good: All factual claims match the provided product information; only light generic persuasion is allowed. ok: One minor plausible inference not explicitly stated, but not contradictory. bad: Invents or contradicts key facts such as material, warranty, or features. GROUNDING RULES (be literal): Features listed as 'key: value' pairs are explicit facts. 'battery: long-lasting' = battery has long-lasting feature. 'energy efficient' MEANS the product is energy efficient. Paraphrasing provided features = grounded. Quote the exact matching text in your explanation."
}

def judge_single_criterion(row, criterion):
    prompt = CRITERION_PROMPT.format(
        criterion_name=criterion,
        criterion_definition=CRITERIA[criterion],
        product_name=row['product_name'],
        attributes=row['Product_attribute_list'],
        material=row['material'],
        warranty=row['warranty'],
        generated_description=row['generated_description']
    )
    
    response = client.chat.completions.create(
        model=JUDGE_CONFIG["model"],
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=JUDGE_CONFIG["max_tokens"],  # Much shorter now
        response_format={"type": "json_object"}  # Simpler schema
    )
    
    result = SingleCriterionScore.model_validate_json(response.choices[0].message.content)
    return {
            "verdict": result.verdict,
            "explanation": result.explanation
        }

def run_judge_isolated(input_file, output_file):
    df = pd.read_excel(input_file)

    criteria = ['fluency', 'grammar', 'tone', 'grounding']

    for criterion in criteria:
        verdict_col = f"{criterion}"
        explanation_col = f"{criterion}_explanation"

        if verdict_col not in df.columns:
            df[verdict_col] = ''
        if explanation_col not in df.columns:
            df[explanation_col] = ''

        df[verdict_col] = df[verdict_col].astype('object')
        df[explanation_col] = df[explanation_col].astype('object')

    print(f"Found {len(df)} rows, judging isolated criteria...")

    for idx, row in df.iterrows():
        if pd.notna(row.get('generated_description', '')):
            print(f"Judging row {idx+1}: {row['product_name']}")

            for criterion in criteria:
                try:
                    result = judge_single_criterion(row, criterion)

                    df.loc[idx, f"{criterion}"] = result["verdict"]
                    df.loc[idx, f"{criterion}_explanation"] = result["explanation"]

                except Exception as e:
                    print(f"Error on row {idx}, criterion {criterion}: {e}")
                    df.loc[idx, f"{criterion}"] = 'error'
                    df.loc[idx, f"{criterion}_explanation"] = str(e)

    if output_file:
        df.to_excel(output_file, index=False)

    print(f"Saved isolated judged results to {output_file}")
    return df
