import json
from openai import OpenAI

def get_ai_macro_view(config, macro_description, asset_classes, log_emitter):
    """
    Uses an AI agent to analyze the macro environment and return a risk budget.
    """
    log_emitter(f"\n--- Using {config['api_provider'].upper()} Agent to analyze macro environment... ---")
    log_emitter(f"Input Description: {macro_description}")
    
    try:
        client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
        prompt = f"""You are a top-tier macro hedge fund strategist formulating top-level asset allocation for an all-weather strategy. Your task is to analyze the following macroeconomic description and assign risk budgets for different major asset classes. Macroeconomic Description: "{macro_description}" Please assign your risk budget percentages for the following asset classes. The sum must be 100. Asset Classes: {', '.join(asset_classes)}. Your output must be in strict JSON format, with keys being the asset class names and values being the corresponding risk budget percentages (integers). Do not include any explanations, code blocks, or extra text. Example output: {{"股票": 40, "债券": 40, "黄金": 20}}"""
        
        response = client.chat.completions.create(
            model=config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.1
        )
        ai_response_text = response.choices[0].message.content
        risk_budget = json.loads(ai_response_text)
        
        if not isinstance(risk_budget, dict) or sum(risk_budget.values()) != 100 or set(risk_budget.keys()) != set(asset_classes):
            raise ValueError("AI response format or content is non-compliant.")
            
        log_emitter(f"AI Agent analysis complete. Risk Budget: {risk_budget}")
        return risk_budget
        
    except Exception as e:
        log_emitter(f"Failed to call AI Agent: {e}")
        return None # Return None on failure