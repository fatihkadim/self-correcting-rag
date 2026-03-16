from app.generation.llm import LLMClient
from app.schemas.claims import Claim, ClaimExtractionResult
from app.utils.prompts import PromptLoader
import json
class ClaimExtractor():
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def extract(self,answer):
        
        template = PromptLoader.load("claim_extract.txt")
        system_prompt = PromptLoader.load("claim_extract_system.txt")
        user_prompt = PromptLoader.format(template,answer=answer)
        response = self.llm.generate(user_prompt,system_prompt=system_prompt)        
        clean_response = self._clean_json(response)

        try:
            parsed_data = json.loads(clean_response)  
            claims = [Claim.model_validate(item) for item in parsed_data] 
            result = ClaimExtractionResult(claims = claims,original_answer=answer,claim_count=len(claims))
            return result
        except json.JSONDecodeError as e:
            print(f"json ayrıstırma hatası: {e}")
            return None
        except Exception as e:
            print(f"beklenmeyen hata {e}")
            return None
 
    def _clean_json(self,text):
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        return cleaned_text


