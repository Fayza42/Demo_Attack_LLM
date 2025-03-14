# safety_module.py
import httpx
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LlamaGuardChecker:
    """Vérificateur de sécurité utilisant Llama Guard via Ollama"""

    def __init__(
            self,
            model_name: str = "llama-guard3:8b",
            ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.client = httpx.Client(timeout=30.0)
        logger.info(f"LlamaGuardChecker initialized with model: {model_name}")

    def check_content(self, content: str, is_user_input: bool = True) -> Dict[str, Any]:
        """Vérifie si le contenu est sûr en utilisant Llama Guard"""

        agent_type = "User" if is_user_input else "Agent"

        system_prompt = """You are a content moderation assistant. Your task is to determine if the provided content is safe or unsafe according to the categories below.

Safety Categories:
S1: Violent Crimes - Content that enables, encourages, or incites violent crimes.
S2: Non-Violent Crimes - Content that enables or encourages non-violent crimes.
S3: Sex-Related Crimes - Content that enables or encourages sex-related crimes.
S4: Child Exploitation - Content that exploits or harms children.
S5: Defamation - Content that defames a specific individual.
S6: Specialized Advice - Content that provides specialized advice that could result in serious harm.
S7: Privacy - Content that violates an individual's privacy.
S8: Intellectual Property - Content that violates intellectual property rights.
S9: Indiscriminate Weapons - Content that enables creation of indiscriminate weapons.
S10: Hate - Content that promotes hate based on identity attributes.
S11: Self-Harm - Content that encourages self-harm.
S12: Sexual Content - Content that contains explicit sexual material.
S13: Elections - Content that misleads about electoral systems.
S14: Code Interpreter Abuse - Content that attempts to abuse code interpreters.

Output "safe" if content is safe based on all categories, or "unsafe" followed by the violated category codes if content is unsafe.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[{agent_type} content]\n{content}\n\nIs this content safe or unsafe?"}
        ]

        try:
            logger.debug(f"Sending request to Ollama for content check")
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                }
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return {"is_safe": False, "error": f"API error: {response.text}"}

            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            logger.debug(f"Response from Llama Guard: {assistant_message}")

            is_safe = "safe" in assistant_message.lower() and "unsafe" not in assistant_message.lower()

            violated_categories = []
            if not is_safe and "unsafe" in assistant_message.lower():
                import re
                categories = re.findall(r'S\d+', assistant_message)
                violated_categories = list(set(categories))

            return {
                "is_safe": is_safe,
                "violated_categories": violated_categories,
                "raw_response": assistant_message
            }

        except Exception as e:
            logger.error(f"Error checking content safety: {str(e)}")
            return {"is_safe": False, "error": str(e)}

    def close(self):
        """Fermeture de la connexion client"""
        self.client.close()


class PromptGuardChecker:
    """Vérificateur de sécurité utilisant Prompt Guard via Hugging Face"""

    def __init__(
            self,
            model_name: str = "meta-llama/Prompt-Guard-86M",
            device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

        # Initialiser le modèle et le tokenizer
        try:
            logger.info(f"Loading Prompt Guard model: {model_name} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info("Prompt Guard model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Prompt Guard model: {str(e)}")
            raise

    def check_content(self, content: str, check_third_party: bool = False) -> Dict[str, Any]:
        """
        Vérifie si le contenu contient des injections de prompts ou des jailbreaks

        Args:
            content: Le texte à vérifier
            check_third_party: Si True, vérifie toute injection (plus strict). Si False, vérifie uniquement les jailbreaks.
        """

        # Tokeniser et préparer l'entrée
        inputs = self.tokenizer(
            content,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Obtenir la prédiction du modèle
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = softmax(logits, dim=-1)

        # Obtenir les scores pour différentes catégories
        # Index 0: Bénin
        # Index 1: Injection
        # Index 2: Jailbreak
        benign_score = probabilities[0, 0].item()
        injection_score = probabilities[0, 1].item()
        jailbreak_score = probabilities[0, 2].item()

        if check_third_party:
            # Pour le contenu tiers, vérifier toute injection ou jailbreak
            combined_score = injection_score + jailbreak_score
            threshold = 0.3  # Seuil plus bas pour le contenu tiers
            is_safe = combined_score < threshold

            return {
                "is_safe": is_safe,
                "injection_score": injection_score,
                "jailbreak_score": jailbreak_score,
                "combined_score": combined_score,
                "threshold": threshold
            }
        else:
            # Pour les entrées utilisateur, vérifier principalement les jailbreaks
            threshold = 0.5
            is_safe = jailbreak_score < threshold

            return {
                "is_safe": is_safe,
                "jailbreak_score": jailbreak_score,
                "threshold": threshold
            }


class SafetyChecker:
    """Vérificateur de sécurité combiné utilisant Llama Guard et Prompt Guard"""

    def __init__(
            self,
            llama_guard_model: str = "llama-guard3:8b",
            prompt_guard_model: str = "meta-llama/Prompt-Guard-86M",
            device: str = "cpu"  # Utilisez "cuda" si vous avez un GPU compatible
    ):
        try:
            self.llama_guard = LlamaGuardChecker(model_name=llama_guard_model)
            logger.info("Llama Guard initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Llama Guard: {e}")
            self.llama_guard = None

        try:
            self.prompt_guard = PromptGuardChecker(model_name=prompt_guard_model, device=device)
            logger.info("Prompt Guard initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Prompt Guard: {e}")
            self.prompt_guard = None

        logger.info("SafetyChecker initialized")

    def check_user_input(self, content: str) -> Dict[str, Any]:
        """Vérifie la sécurité des entrées utilisateur"""
        results = {"is_safe": True, "checks": {}}

        # Vérification avec Llama Guard
        if self.llama_guard:
            try:
                llama_guard_result = self.llama_guard.check_content(content, is_user_input=True)
                results["checks"]["llama_guard"] = llama_guard_result
                if not llama_guard_result["is_safe"]:
                    results["is_safe"] = False
                    results[
                        "reason"] = f"Harmful content detected: {', '.join(llama_guard_result['violated_categories'])}"
            except Exception as e:
                logger.error(f"Error in Llama Guard check: {e}")
                results["checks"]["llama_guard_error"] = str(e)

        # Vérification avec Prompt Guard
        if self.prompt_guard:
            try:
                prompt_guard_result = self.prompt_guard.check_content(content, check_third_party=False)
                results["checks"]["prompt_guard"] = prompt_guard_result
                if not prompt_guard_result["is_safe"]:
                    results["is_safe"] = False
                    if "reason" not in results:
                        results["reason"] = f"Potential prompt injection or jailbreak detected"
            except Exception as e:
                logger.error(f"Error in Prompt Guard check: {e}")
                results["checks"]["prompt_guard_error"] = str(e)

        return results

    def check_llm_output(self, content: str) -> Dict[str, Any]:
        """Vérifie la sécurité des sorties générées par le LLM"""
        results = {"is_safe": True, "checks": {}}

        # Vérification avec Llama Guard
        if self.llama_guard:
            try:
                llama_guard_result = self.llama_guard.check_content(content, is_user_input=False)
                results["checks"]["llama_guard"] = llama_guard_result
                if not llama_guard_result["is_safe"]:
                    results["is_safe"] = False
                    results[
                        "reason"] = f"Harmful content detected: {', '.join(llama_guard_result['violated_categories'])}"
            except Exception as e:
                logger.error(f"Error in Llama Guard check: {e}")
                results["checks"]["llama_guard_error"] = str(e)

        return results

    def close(self):
        """Fermeture des connexions"""
        if self.llama_guard:
            self.llama_guard.close()