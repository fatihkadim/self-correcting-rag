from app.utils.prompts import PromptLoader
from app.generation.llm import LLMClient
import json
class LLMJudge():
    def __init__(self,llm_client:LLMClient):
        self.llm = llm_client

    def judge(self,claim:str,evidence: list[str]):
        template = PromptLoader.load("verify_user.txt")
        system = PromptLoader.load("verify_system.txt")
        evidence_text = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(evidence)])
        prompt = PromptLoader.format(template,claim=claim,evidence=evidence_text)
        response = self.llm.generate(prompt,system)        
        clean_response = self._clean_json(response)
    
        try:
            parsed_data = json.loads(clean_response)  
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"json ayrıstırma hatası: {e}")
            return None
        except Exception as e:
            print(f"beklenmeyen hata {e}")
            return None
 
    def _clean_json(self,text):
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        return cleaned_text
