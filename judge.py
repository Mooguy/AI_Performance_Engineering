# ========== TASK 5: JUDGE ==========
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

from config import client, JUDGE_CONFIG
# from utils import normalize_features

# Pydantic models
class CriterionScore(BaseModel):
    explanation: str = Field(description="Brief reasoning for the verdict")
    verdict: Literal["good", "ok", "bad"] = Field(description="One of: good, ok, bad")

class JudgeOutput(BaseModel):
    fluency: CriterionScore
    grammar: CriterionScore
    tone: CriterionScore
    grounding: CriterionScore

JUDGE_PROMPT = """You are a strict evaluator of AI-generated e-commerce product descriptions.

Your job is to grade the description using the rubric below. Be conservative and evidence-based.

RUBRIC

Fluency
- good: Reads smoothly and naturally; no awkward phrasing.
- ok: Mostly clear with minor awkward phrasing.
- bad: Choppy or confusing; hard to follow.

Grammar
- good: No noticeable spelling, grammar, or punctuation errors.
- ok: 1-2 minor errors that do not affect understanding.
- bad: Multiple errors that distract or cause confusion.

Tone
- good: Friendly, credible sales voice; positive and persuasive.
- ok: Generally appropriate but slightly too dry, casual, or overhyped.
- bad: Clearly inappropriate, such as negative, rude, sarcastic, robotic, or strongly unnatural.

Grounding
- good: All factual claims match the provided product information; only light generic persuasion is allowed.
- ok: One minor plausible inference not explicitly stated, but not contradictory.
- bad: Invents or contradicts key facts such as material, warranty, or features.

SCORING RULES
- Be strict. Do not give "good" unless the description clearly meets the full definition of good.
- If the description is borderline between two ratings, choose the lower rating.
- Every criterion must include a specific explanation, even when the verdict is "good".
- Explanations must reference concrete wording from the description.
- For grounding, compare factual claims in the description against the product information.
- If unsupported factual details are added, grounding cannot be "good".
- Do not be polite or generous. Your role is accurate grading, not helpful writing.

GROUNDING RULES (be literal):
- Features are listed as "key: value" pairs. "battery: long-lasting" MEANS the battery is long-lasting.
- "energy efficient" MEANS the product is energy efficient.
- If description uses exact phrasing from ANY product field, it is "good".
- Quote the exact matching text in your explanation.

PRODUCT INFORMATION
Product name: {product_name}
Attributes: {attributes}
Material: {material}
Warranty: {warranty}

DESCRIPTION TO EVALUATE
{generated_description}

- Explanations: 1 sentence MAX, 25 words MAX. Quote 1 example.

Return valid JSON that matches the required schema exactly.
"""

def judge_description(row, config=JUDGE_CONFIG):
    prompt = JUDGE_PROMPT.format(
        product_name=row['product_name'],
        attributes=row['Product_attribute_list'],
        material=row['material'],
        warranty=row['warranty'],
        generated_description=row['generated_description']
    )
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "system", "content": prompt}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "schema": JudgeOutput.model_json_schema()
            }
        }
    )
    
    result = JudgeOutput.model_validate_json(response.choices[0].message.content)
    
    # Flatten verdicts for Excel columns
    return {
        "fluency": result.fluency.verdict,
        "fluency_explanation": result.fluency.explanation,
        "grammar": result.grammar.verdict,
        "grammar_explanation": result.grammar.explanation,
        "tone": result.tone.verdict,
        "tone_explanation": result.tone.explanation,
        "grounding": result.grounding.verdict,
        "grounding_explanation": result.grounding.explanation
    }

def run_judge(input_file, output_file):
    """Ensure input and output file paths are provided"""
    if not input_file:        
        raise ValueError("input_file must be provided.")

    """Run judge on all rows with descriptions."""
    print("Loading Excel...")
    df = pd.read_excel(input_file)

    judge_cols = ['fluency', 'grammar', 'tone', 'grounding',
                    'fluency_explanation', 'grammar_explanation',
                    'tone_explanation', 'grounding_explanation']

    for col in judge_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].astype('object')
    
    print(f"Found {len(df)} rows, judging descriptions...")
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('generated_description', '')):
            print(f"Judging row {idx+1}: {row['product_name']}...")
            try:
                scores = judge_description(row)
                df.loc[idx, 'fluency'] = scores['fluency']
                df.loc[idx, 'grammar'] = scores['grammar']
                df.loc[idx, 'tone'] = scores['tone']
                df.loc[idx, 'grounding'] = scores['grounding']
                df.loc[idx, 'fluency_explanation'] = scores['fluency_explanation']
                df.loc[idx, 'grammar_explanation'] = scores['grammar_explanation']
                df.loc[idx, 'tone_explanation'] = scores['tone_explanation']
                df.loc[idx, 'grounding_explanation'] = scores['grounding_explanation']
            except Exception as e:
                print(f"Error judging row {idx}: {e}")
                df.loc[idx, 'fluency'] = 'error'
    if output_file:
        df.to_excel(output_file, index=False)
    print(f"Saved judged results to {output_file}")
    return df
