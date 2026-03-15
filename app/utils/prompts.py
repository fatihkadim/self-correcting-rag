from pathlib import Path

class PromptLoader():

    @staticmethod
    def load(filename):
        filepath=Path(__file__).parent.parent / "prompts" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"prompt dosyası bulunamadı: {filepath}")

        return filepath.read_text(encoding="utf-8")

    @staticmethod
    def format(template, **kwargs):
        return template.format(**kwargs)

