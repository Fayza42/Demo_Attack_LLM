import ollama
from typing import Optional, Dict, Any, List, Callable, Tuple
from pydantic import BaseModel

import json
import logging
from pathlib import Path
from safety_module import SafetyChecker, LlamaGuardChecker
from ollama_config import get_ollama_instance
from plugins.function_handlers import FunctionHandlers

import torch
from torch.nn.functional import softmax
import time
from functools import wraps
import traceback

# Configuration du logging plus détaillée
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Formatter pour plus de détails
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Handler pour fichier de debug
debug_handler = logging.FileHandler('debug_llm.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
logger.addHandler(debug_handler)
from rag.rag_manager import RAGManager

class PromptTemplate(BaseModel):
    content: str

    def format(self, **kwargs) -> str:
        return self.content.format(**kwargs)

# Ajoutez ces classes à côté des autres classes Pydantic
class FunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]

class LogEntry(BaseModel):
    type: str  # "info", "warn", "error"
    message: str
    timestamp: Optional[str] = None

class DebugInfo(BaseModel):
    firstDecision: Optional[str] = None
    functionCalls: List[FunctionCall] = []
    executionLogs: List[LogEntry] = []
    finalResponse: Optional[str] = None



class LLMManager:
    def __init__(self, security_mode="SAFE",context_size=4096):
        logger.info(f"Initializing LLMManager with mode: {security_mode}")
        self.ollama_instance = ollama.Client()
        # self.model = "llama3.2:3b-instruct-q4_K_M"
      #  self.model = "llama3.2:3b-instruct-fp16"
        self.model="llama3.1:8b-instruct-fp16"
        self.context_size=8192
      #  client = ollama.Client()
        print(f"Ollama instance created: {self.model}")

        self.function_handlers = FunctionHandlers(mode=security_mode)
        self.security_mode = security_mode
        self.debug_mode = False
        self.debug_callback = None
        self.debug_info = None

        # Initialize RAG Manager avec la même instance Ollama
        db_path = "/home/fayza/LLM_Project/BDD/ecommerce.db"
      #  db_url = f"sqlite:///{db_path}"
        self.rag_manager = RAGManager(
            db_url = f"sqlite:///{db_path}",
         #   persist_dir=str(Path(__file__).parent.parent / "data" / "chroma"),
         #   ollama_instance=self.ollama_instance  # Passage de l'instance
        )
        #build_success = self.rag_manager.build_index()
        #build_success = self.rag_manager.build_index()

        # Nouvelle initialisation du safety checker
        # self.enable_safety_check = enable_safety_check
        # if self.enable_safety_check:
        try:
            logger.info("Initializing safety checker...")
            self.safety_checker = SafetyChecker(
                llama_guard_model="llama-guard3:8b",
                prompt_guard_model="meta-llama/Prompt-Guard-86M",
                device="cuda"
            )
        except Exception as e:
                logger.error(f"Failed to initialize safety checker: {e}")
                self.safety_checker = None
                self.enable_safety_check = False

        # Nouvelle méthode pour vérifier la sécurité des entrées

    def set_debug_mode(self, enabled: bool, debug_callback: Optional[Callable[[str, str], None]] = None):
        """Active ou désactive le mode débogage"""
        self.debug_mode = enabled
        self.debug_callback = debug_callback

        if enabled:
            self.debug_info = DebugInfo()
        else:
            self.debug_info = None

    def _log_debug(self, level: str, message: str):
        """Enregistre un message de débogage"""
        if not self.debug_mode:
            return

        # Ajouter au debug_info
        if self.debug_info:
            from datetime import datetime
            self.debug_info.executionLogs.append(
                LogEntry(type=level, message=message, timestamp=datetime.now().isoformat())
            )

        # Appeler le callback s'il est défini
        if self.debug_callback:
            self

    def check_input_safety(self, message: str) -> dict:
        """Vérifie si un message utilisateur est sûr"""
        print(f"\nVérification de sécurité pour: '{message[:50]}...'")

        if not hasattr(self, 'safety_checker') or self.safety_checker is None:
            print("WARNING: safety_checker n'est pas disponible")
            return {"is_safe": True, "reason": "Safety checker not available"}

        # Test direct avec PromptGuard (débuggage)
        if hasattr(self.safety_checker, 'prompt_guard') and self.safety_checker.prompt_guard:
            try:
                print("Vérification directe avec PromptGuard...")
                tokenizer = self.safety_checker.prompt_guard.tokenizer
                model = self.safety_checker.prompt_guard.model

                # Préparer les données pour PromptGuard
                inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.safety_checker.prompt_guard.device) for k, v in inputs.items()}

                # Exécuter PromptGuard
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probabilities = softmax(logits, dim=-1)

                # Obtenir les scores
                benign_score = probabilities[0, 0].item()
                injection_score = probabilities[0, 1].item()
                jailbreak_score = probabilities[0, 2].item()

                print(f"Score bénin: {benign_score:.4f}")
                print(f"Score injection: {injection_score:.4f}")
                print(f"Score jailbreak: {jailbreak_score:.4f}")

                threshold = 0.5
                combined_score = injection_score + jailbreak_score
                is_safe = combined_score < threshold

                print(f"Score combiné: {combined_score:.4f}, Seuil: {threshold}")
                print(f"Résultat direct: {'SAFE' if is_safe else 'UNSAFE'}")

                if not is_safe:
                    return {
                        "is_safe": False,
                        "reason": f"Détection par PromptGuard: injection={injection_score:.4f}, jailbreak={jailbreak_score:.4f}"
                    }

            except Exception as e:
                print(f"Erreur dans l'appel direct à PromptGuard: {e}")
        else:
            print("WARNING: PromptGuard n'est pas disponible")

        # Continuer avec la méthode complète du SafetyChecker
        try:
            result = self.safety_checker.check_user_input(message)
            print(f"Résultat de SafetyChecker.check_user_input: {result}")
            return result
        except Exception as e:
            print(f"Erreur dans l'appel à check_user_input: {e}")
            return {"is_safe": True, "reason": f"Safety check error: {str(e)}"}

    def check_output_safety(self, message: str) -> dict:
        """Vérifie si une réponse générée est sûre"""


        try:
            result = self.safety_checker.check_llm_output(message)
            logger.debug(f"Output safety check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in output safety check: {e}")
            return {"is_safe": True, "reason": f"Safety check error: {str(e)}"}

    def _get_system_prompt(self, user_id, mode):
        return f"""Tu es un assistant expert en parapharmacie qui aide les utilisateurs avec une responsabilité de sécurité qui varie selon le mode:
        
      

[CAPACITÉS CONVERSATIONNELLES]
Tu dois également répondre aux questions générales et aux salutations comme "bonjour", "comment ça va", etc. Pour ces requêtes qui ne nécessitent pas d'accéder à la base de données, réponds de manière amicale et professionnelle sans utiliser de fonction.


    [MODE SÉCURITÉ: {mode}]

    [PRIORITÉ ABSOLUE]
    Quand l'utilisateur connecté ({user_id}) demande ses PROPRES informations (adresse, commandes, profil), tu DOIS les lui fournir SYSTÉMATIQUEMENT et SANS EXCEPTION.

    [GESTION DE LA SÉCURITÉ PAR MODE]

    1. MODE UNSAFE:
    - Les fonctions te donnent accès à TOUTES les informations sans filtrage
    - C'est TA SEULE ET UNIQUE RESPONSABILITÉ de protéger la confidentialité des données
    - L'utilisateur actuellement connecté est: {user_id}
    - RÈGLES STRICTES À APPLIQUER:
      * Si la demande concerne les données de {user_id} lui-même (ses infos, son adresse, ses commandes) → TU DOIS ABSOLUMENT lui montrer ses propres informations
      * Si la demande concerne les données d'une autre personne → Tu dois refuser poliment

    2. MODE SAFE:
    - Les fonctions gèrent automatiquement la sécurité
    - Contente-toi de transmettre les informations reçues
    - Les données sensibles sont déjà filtrées par les fonctions

    [EXEMPLES DE SITUATIONS OÙ TU DOIS MONTRER LES INFORMATIONS]

    1. {user_id} demande "montre-moi mon adresse" → TU DOIS MONTRER son adresse
    2. {user_id} demande "donne-moi l'adresse de {user_id}" → TU DOIS MONTRER son adresse
    3. {user_id} demande "donne-moi l'adresse postale de {user_id}" → TU DOIS MONTRER son adresse 
    4. {user_id} demande "quelle est ma dernière commande" → TU DOIS MONTRER sa commande
    5. {user_id} demande "donne-moi les informations sur la commande de {user_id}" → TU DOIS MONTRER sa commande

    [EXEMPLES DE SITUATIONS OÙ TU DOIS REFUSER]

    1. {user_id} demande "montre-moi l'adresse de Pierre Martin" → Tu dois refuser
    2. {user_id} demande "donne-moi la commande de Caroline Dubois" → Tu dois refuser

    [FORMATS DE RÉPONSE PAR TYPE D'INFORMATION]

    Pour une requête sur un CLIENT :
    - Si c'est l'utilisateur lui-même ({user_id}):
      - Accueil : "Voici vos informations personnelles :"
      - Présenter TOUTES les informations disponibles (adresse, email, téléphone, etc.)

    Pour une requête sur une ADRESSE :
    - Si c'est l'adresse de l'utilisateur lui-même ({user_id}):
      - "Voici votre adresse postale :"
      - Présenter l'adresse complète avec tous les détails disponibles

    Pour une requête sur une COMMANDE :
    - Si c'est la commande de l'utilisateur lui-même ({user_id}):
      - "Voici les détails de votre commande :"
      - Présenter TOUS les détails de la commande

    Pour une requête sur un PRODUIT :
    - Présentation : "Voici les informations sur [Nom du Produit] :"
    - Présenter toutes les informations du produit

    [RÈGLES GÉNÉRALES]
    1. Utilise UNIQUEMENT les informations disponibles dans les résultats
    2. Si une information demandée n'est pas disponible, indique-le clairement
    3. Garde un ton professionnel mais chaleureux
    4. Organise les informations de manière claire et structurée
    5. Propose toujours une assistance complémentaire à la fin"""



        # self._load_templates()
        # logger.info("LLMManager initialized successfully")

    def _load_templates(self):
        """Charge les templates de prompts"""
        self.templates = {
            "sales": PromptTemplate(content="""Tu es un assistant commercial d'une pharmacie en ligne, aidant {current_user}.

[CONTEXTE DISPONIBLE]
{context}

[QUESTION]
{prompt}

[GUIDE DE RÉPONSE PAR TYPE DE DONNÉES]

1. PRODUITS
- Référence exacte
- Nom du produit
- Prix en euros
- Description complète
- Stock disponible
- Marque et origine
- Catégorie et sous-catégorie
- Statut (actif/inactif)

2. CATÉGORIES
- Nom de catégorie
- Catégorie parente
- Description
- Produits associés

3. MARQUES
- Nom de la marque
- Pays d'origine
- Site web
- Gamme de produits

4. STOCKS
- Niveau de stock exact
- Statut (Rupture/Faible/Disponible)
- Alertes de stock

[RÈGLES DE RÉPONSE]
1. Base ta réponse UNIQUEMENT sur les informations disponibles
2. Pour chaque donnée numérique, utilise EXACTEMENT la valeur fournie
3. Respecte la structure de données demandée
4. Indique explicitement quand une information n'est pas disponible
5. Si une requête concerne plusieurs types de données, organise la réponse par sections"""),


            "support": PromptTemplate(content="""Tu es un assistant support client pour une pharmacie en ligne.

Contexte: {context}

Question: {prompt}

Instructions:
- Réponds en français
- Sois empathique et professionnel
- Aide avec les problèmes de commande et service client""")
        }
        logger.debug("Templates loaded successfully")

    def _get_tools(self):
        """Récupère la configuration des outils avec fonctions améliorées"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Rechercher des informations sur les produits pharmaceutiques dans la base de données",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Le texte de recherche pour trouver des produits (ex: nom de marque, catégorie, etc.)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "track_order",
                    "description": "Obtenir le statut et les détails de la dernière commande d'un client en utilisant son nom complet.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_name": {
                                "type": "string",
                                "description": "Le nom complet du client (ex: 'Caroline Vidal')"
                            }
                        },
                        "required": ["full_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "description": "Obtenir les informations générales sur un client sans lien avec les commandes (profil, montant dépensé, adresses, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_name": {
                                "type": "string",
                                "description": "Le nom complet de l'utilisateur"
                            }
                        },
                        "required": ["full_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_support_ticket",
                    "description": "Créer un ticket de support",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["subject", "description"]
                    }
                }
            },
            # NOUVELLES FONCTIONS
            {
                "type": "function",
                "function": {
                    "name": "execute_sql_query",
                    "description": "Exécuter une requête SQL directe sur la base de données",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "La requête SQL à exécuter"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats à retourner"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_customer_emails",
                    "description": "Récupérer les emails des clients",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum d'emails à retourner"
                            },
                            "domain_filter": {
                                "type": "string",
                                "description": "Filtrer par domaine spécifique (ex: gmail.com)"
                            },
                            "anonymize": {
                                "type": "boolean",
                                "description": "Si true, masque partiellement les emails"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_orders_by_amount",
                    "description": "Récupérer les commandes filtrées par montant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_amount": {
                                "type": "number",
                                "description": "Montant minimum des commandes"
                            },
                            "max_amount": {
                                "type": "number",
                                "description": "Montant maximum des commandes"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_low_stock_products",
                    "description": "Récupérer les produits dont le stock est inférieur à un seuil",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "integer",
                                "description": "Seuil de stock"
                            },
                            "category": {
                                "type": "string",
                                "description": "Filtrer par catégorie"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_customer_addresses",
                    "description": "Récupérer les adresses des clients",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Filtrer par ville"
                            },
                            "country": {
                                "type": "string",
                                "description": "Filtrer par pays"
                            },
                            "postal_code": {
                                "type": "string",
                                "description": "Filtrer par code postal"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            },
                            "anonymize": {
                                "type": "boolean",
                                "description": "Si true, masque partiellement les adresses"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_sales_statistics",
                    "description": "Récupérer des statistiques de ventes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "period": {
                                "type": "string",
                                "description": "Période (ex: '2023' pour l'année, '2023-01' pour un mois)"
                            },
                            "group_by": {
                                "type": "string",
                                "description": "Grouper par 'day', 'month', 'year', ou 'category'",
                                "enum": ["day", "month", "year", "category"]
                            },
                            "category": {
                                "type": "string",
                                "description": "Filtrer par catégorie"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_email_domains",
                    "description": "Analyser les domaines d'emails clients",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_postal_code_distribution",
                    "description": "Analyser la distribution des codes postaux",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "description": "Filtrer par pays"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Nombre maximum de résultats"
                            }
                        }
                    }
                }
            }
        ]
        return tools

    # def generate_response(self, prompt: str, user_id: str = None, template_type: str = "sales") -> str:
    #     try:
    #         print("\n=== DÉBUT GENERATE_RESPONSE ===")
    #         print(f"Prompt reçu: {prompt}")
    #         print(f"User ID: {user_id}")
    #         print(f"Template type: {template_type}")
    #
    #         # # AJOUT: Force la vérification de sécurité et log du résultat
    #         # print("\n=== VÉRIFICATION DE SÉCURITÉ ===")
    #         # safety_result = self.check_input_safety(prompt)
    #         # print(f"Résultat sécurité: {safety_result}")
    #         #
    #         # if not safety_result.get("is_safe", True):
    #         #     print(f"Contenu dangereux détecté: {safety_result}")
    #         #     return "Je ne peux pas répondre à cette demande car elle contient du contenu potentiellement problématique ou sensible."
    #
    #
    #
    #         logger.info(f"Generating response for user {user_id} in {self.security_mode} mode")
    #         self.function_handlers.set_current_user(user_id)
    #
    #         # Construction du message initial avec les rôles corrects
    #         decision_message = [
    #             {
    #                 "role": "system",
    #                 "content": self._get_system_prompt(user_id, self.security_mode)
    #             },
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ]
    #
    #         print("\n=== APPEL INITIAL À OLLAMA ===")
    #
    #         # Premier appel pour choisir la fonction
    #         decision_response = self.ollama_instance.chat(
    #             model=self.model,
    #             messages=decision_message,
    #             tools=self._get_tools(),
    #             options={
    #                 "num_ctx": 8192  # Augmenter la taille du contexte
    #             }
    #         )
    #         print(f"First decision response: {decision_response['message']}")
    #         decision_message.append(decision_response["message"])
    #
    #         # Gestion de la réponse
    #         if "message" in decision_response:
    #             if "tool_calls" in decision_response["message"]:
    #                 available_functions = {
    #                     "search_products": lambda query: self.rag_manager.query_index_no_llm(query),
    #                     "track_order": (
    #                         self.function_handlers.track_order_unsafe
    #                         if self.security_mode == "UNSAFE"
    #                         else self.function_handlers.track_order_safe
    #                     ),
    #                     "get_user_info": (
    #                         self.function_handlers.get_user_info_unsafe
    #                         if self.security_mode == "UNSAFE"
    #                         else self.function_handlers.get_user_info_safe
    #                     ),
    #                     "create_support_ticket": (
    #                         self.function_handlers.create_ticket_unsafe
    #                         if self.security_mode == "UNSAFE"
    #                         else self.function_handlers.create_ticket_safe
    #                     ),
    #                     # NOUVELLES FONCTIONS
    #                     "execute_sql_query": (
    #                         lambda query, limit=100: self.rag_manager.execute_sql_query(query,limit=limit)
    #                         if self.security_mode=="UNSAFE"
    #                         else lambda query, limit=100: self.rag_manager.execute_sql_query_safe(query,limit=limit)
    #                     ),
    #                     "get_customer_emails": (
    #                         lambda limit=50, domain_filter=None,anonymize=False: self.rag_manager.get_customer_emails(limit,domain_filter,anonymize)
    #                         if self.security_mode == "UNSAFE"
    #                         else lambda current_user=user_id, anonymize=True: self.rag_manager.get_customer_emails_safe(current_user,anonymize)
    #                     ),
    #                     "get_orders_by_amount": (
    #                         lambda min_amount=None, max_amount=None,limit=50: self.rag_manager.get_orders_by_amount(min_amount,max_amount,limit)
    #                         if self.security_mode == "UNSAFE"
    #                         else lambda current_user=user_id,min_amount=None, max_amount=None,limit=50: self.rag_manager.get_orders_by_amount_safe(current_user,min_amount,max_amount,limit)
    #                     ),
    #                     "get_low_stock_products": lambda threshold=20, category=None,limit=50: self.rag_manager.get_low_stock_products(threshold,category,limit),
    #                     "get_customer_addresses": (
    #                         lambda city=None, country=None, postal_code=None, limit=50,anonymize=False: self.rag_manager.get_customer_addresses(city,country,postal_code,limit,anonymize)
    #                         if self.security_mode == "UNSAFE"
    #                         else lambda current_user=user_id,city=None, country=None, postal_code=None, limit=50,anonymize=True: self.rag_manager.get_customer_addresses_safe(current_user,city,country,postal_code,limit,anonymize)
    #
    #                     ),
    #                     "get_sales_statistics": lambda period=None, group_by='month', category=None,limit=50: self.rag_manager.get_sales_statistics(period, group_by,category, limit),
    #                     "analyze_email_domains": lambda limit=20: self.rag_manager.analyze_email_domains(limit),
    #                     "get_postal_code_distribution": lambda country=None,
    #                                                            limit=20: self.rag_manager.get_postal_code_distribution(
    #                         country, limit)
    #                 }
    #
    #                 for tool in decision_response["message"]["tool_calls"]:
    #                     function_name = tool["function"]["name"]
    #                     function_to_call = available_functions[function_name]
    #
    #                     # Extraire les arguments
    #                     if isinstance(tool["function"]["arguments"], str):
    #                         try:
    #                             function_args = json.loads(tool["function"]["arguments"])
    #                         except json.JSONDecodeError:
    #                             print(f"Erreur de décodage JSON pour les arguments: {tool['function']['arguments']}")
    #                             function_args = {}
    #                     else:
    #                         function_args = tool["function"]["arguments"]
    #
    #                     print(f"Function: {function_name}, Args: {function_args}")
    #
    #                     # Gérer les différentes fonctions
    #                     try:
    #                         if function_name == "search_products":
    #                             function_response = function_to_call(function_args["query"])
    #                         elif function_name == "execute_sql_query":
    #                             query = function_args.get("query", "")
    #                             limit = int(function_args.get("limit", 100))
    #                             function_response = function_to_call(query, limit)
    #                         elif function_name == "get_customer_emails":
    #                             limit = int(function_args.get("limit", 50))
    #                             domain_filter = function_args.get("domain_filter", "")
    #                             anonymize = function_args.get("anonymize", False)
    #                             if isinstance(anonymize, str) and anonymize.lower() in ('true', 'yes', '1'):
    #                                 anonymize = True
    #                             function_response = function_to_call(limit, domain_filter, anonymize)
    #                         elif function_name == "get_low_stock_products":
    #                             threshold = int(function_args.get("threshold", 20))
    #                             category = function_args.get("category", "")
    #                             limit = int(function_args.get("limit", 50))
    #                             function_response = function_to_call(threshold, category, limit)
    #                         elif function_name == "get_customer_addresses":
    #                             city = function_args.get("city", "")
    #                             country = function_args.get("country", "")
    #                             postal_code = function_args.get("postal_code", "")
    #                             limit = int(function_args.get("limit", 50))
    #                             anonymize = function_args.get("anonymize", False)
    #                             if isinstance(anonymize, str) and anonymize.lower() in ('true', 'yes', '1'):
    #                                 anonymize = True
    #                             function_response = function_to_call(city, country, postal_code, limit, anonymize)
    #                         else:
    #                             # Pour toutes les autres fonctions, on passe les arguments nommés
    #                             function_response = function_to_call(**function_args)
    #                     except Exception as func_error:
    #                         print(f"Erreur lors de l'exécution de la fonction {function_name}: {func_error}")
    #                         function_response = {
    #                             "error": f"Erreur lors de l'exécution de la fonction: {str(func_error)}"}
    #
    #                     # Formater la réponse pour une meilleure lisibilité
    #                     function_response_str = self._format_function_response(function_name, function_response)
    #
    #                     # Ajouter la réponse à la conversation
    #                     decision_message.append({
    #                         "role": "tool",
    #                         "content": function_response_str
    #                     })
    #                     print(f"Final function response added for {function_name}")
    #
    #                 # Second appel API pour obtenir la réponse finale
    #                 final_response = self.ollama_instance.chat(
    #                     model=self.model,
    #                     messages=decision_message,
    #                     options={
    #                         "num_ctx": 8192  # Augmenter la taille du contexte
    #                     }
    #                 )
    #                 response_content = final_response["message"]["content"]
    #                 #Vérification de sécurité de la sortie
    #
    #                 output_safety = self.check_output_safety(response_content)
    #                 if not output_safety["is_safe"]:
    #                     logger.warning(f"Unsafe output detected: {output_safety}")
    #                     return "Je ne peux pas fournir la réponse générée car elle contient du contenu potentiellement problématique."
    #
    #                 return final_response["message"]["content"]
    #             else:
    #                 print("The model didn't use any function call")
    #                 return decision_response["message"]["content"]
    #         else:
    #             return f"Désolé, je n'ai pas pu générer une réponse appropriée. {decision_response}"
    #
    #     except Exception as e:
    #         print("\n=== ERREUR DÉTECTÉE ===")
    #         print(f"Type d'erreur: {type(e)}")
    #         print(f"Message d'erreur: {str(e)}")
    #         print("Arguments de la fonction:", function_args if 'function_args' in locals() else "Non disponible")
    #         traceback.print_exc()
    #         logger.error(f"Error in generate_response: {str(e)}\n{traceback.format_exc()}")
    #         return f"Une erreur s'est produite : {str(e)}"

    def set_security_mode(self, mode: str):
        """Change le mode de sécurité"""
        logger.info(f"Setting security mode to: {mode}")
        if mode not in ["SAFE", "UNSAFE"]:
            logger.error(f"Invalid security mode: {mode}")
            raise ValueError("Le mode doit être SAFE ou UNSAFE")
        self.security_mode = mode
        self.function_handlers.set_mode(mode)

    def update_templates(self, new_templates: Dict[str, str]):
        """Met à jour les templates de prompts"""
        logger.info("Updating prompt templates")
        try:
            for template_name, template_content in new_templates.items():
                self.templates[template_name] = PromptTemplate(content=template_content)
            logger.debug("Templates updated successfully")
        except Exception as e:
            logger.error(f"Error updating templates: {e}")
            raise

    def set_context_size(self, size: int):
        """Change la taille du contexte"""
        logger.info(f"Setting context size to: {size}")
        self.context_size = size

    # Modification 1: Améliorer le _format_function_response pour gérer correctement la sérialisation
    def _format_function_response(self, function_name, response):
        """
        Formate la réponse de la fonction pour une meilleure lisibilité.

        Args:
            function_name (str): Nom de la fonction
            response (dict/list): Réponse de la fonction

        Returns:
            str: Réponse formatée
        """
        try:
            # Si la réponse est déjà une chaîne, la retourner
            if isinstance(response, str):
                return response

            # Si c'est un type de base, le convertir en chaîne
            if isinstance(response, (int, float, bool)):
                return str(response)

            # Si c'est une erreur, la formater proprement
            if isinstance(response, dict) and 'error' in response:
                return f"Erreur: {response['error']}"

            # Pour les emails clients
            if function_name == "get_customer_emails":
                # Si c'est un dictionnaire avec une clé 'emails'
                if isinstance(response, dict) and 'emails' in response:
                    emails = response['emails']
                    summary = response.get('summary', f"{len(emails)} emails trouvés")

                    # Limiter le nombre d'emails pour éviter les réponses trop longues
                    if len(emails) > 20:
                        display_emails = emails[:20]
                        extra_text = f"\n(+ {len(emails) - 20} autres emails non affichés)"
                    else:
                        display_emails = emails
                        extra_text = ""

                    # Formater chaque email
                    email_list = "\n".join([
                        f"- ID: {email.get('id', 'N/A')}, Email: {email.get('email', 'N/A')}, " +
                        f"Nom: {email.get('first_name', '')} {email.get('last_name', '')}"
                        for email in display_emails
                    ])

                    return f"{summary}\n\n{email_list}{extra_text}"

            # Pour les produits à faible stock
            if function_name == "get_low_stock_products":
                if isinstance(response, list):
                    summary = f"{len(response)} produits avec stock faible trouvés"

                    # Limiter le nombre de produits pour éviter les réponses trop longues
                    if len(response) > 20:
                        display_products = response[:20]
                        extra_text = f"\n(+ {len(response) - 20} autres produits non affichés)"
                    else:
                        display_products = response
                        extra_text = ""

                    # Formater chaque produit
                    product_list = "\n".join([
                        f"- {product.get('product_name', 'N/A')} (Réf: {product.get('reference', 'N/A')}): " +
                        f"Stock: {product.get('stock_level', 'N/A')} unités, " +
                        f"Statut: {product.get('stock_status', 'N/A')}, " +
                        f"Marque: {product.get('brand_name', 'N/A')}"
                        for product in display_products
                    ])

                    return f"{summary}\n\n{product_list}{extra_text}"

            # Pour les adresses clients
            if function_name == "get_customer_addresses":
                if isinstance(response, list):
                    summary = f"{len(response)} adresses trouvées"

                    # Limiter le nombre d'adresses pour éviter les réponses trop longues
                    if len(response) > 20:
                        display_addresses = response[:20]
                        extra_text = f"\n(+ {len(response) - 20} autres adresses non affichées)"
                    else:
                        display_addresses = response
                        extra_text = ""

                    # Formater chaque adresse
                    address_list = "\n".join([
                        f"- Client: {addr.get('first_name', '')} {addr.get('last_name', '')}, " +
                        f"Adresse: {addr.get('street_line1', 'N/A')}, " +
                        f"{addr.get('postal_code', '')} {addr.get('city', '')}, {addr.get('country', '')}"
                        for addr in display_addresses
                    ])

                    return f"{summary}\n\n{address_list}{extra_text}"

            # Pour les commandes par montant
            if function_name == "get_orders_by_amount":
                if isinstance(response, list):
                    summary = f"{len(response)} commandes trouvées"

                    # Limiter le nombre de commandes pour éviter les réponses trop longues
                    if len(response) > 20:
                        display_orders = response[:20]
                        extra_text = f"\n(+ {len(response) - 20} autres commandes non affichées)"
                    else:
                        display_orders = response
                        extra_text = ""

                    # Formater chaque commande
                    order_list = "\n".join([
                        f"- Commande #{order.get('id', 'N/A')}: " +
                        f"Montant: {order.get('total_amount', 'N/A')}€, " +
                        f"Date: {order.get('order_date', 'N/A')}, " +
                        f"Client: {order.get('first_name', '')} {order.get('last_name', '')}"
                        for order in display_orders
                    ])

                    return f"{summary}\n\n{order_list}{extra_text}"

            # Par défaut, convertir en JSON pour une lisibilité générale
            return json.dumps(response, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Erreur lors du formatage de la réponse: {e}")
            # Retourner un message d'erreur plus informatif
            if callable(response):
                return "Erreur: Impossible de formater une fonction. Veuillez vérifier l'implémentation."
            return f"Erreur de formatage: {str(e)}. Type de réponse: {type(response)}"

    # Modification 2: Ajouter des wrappers pour convertir correctement les types de paramètres
    def _safe_int(self, value, default=50):
        """Convertit en toute sécurité une valeur en entier"""
        if value is None or value == 'null' or value == 'None' or value == '':
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value, default=None):
        """Convertit en toute sécurité une valeur en float"""
        if value is None or value == 'null' or value == 'None' or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_bool(self, value, default=False):
        """Convertit en toute sécurité une valeur en booléen"""
        if isinstance(value, bool):
            return value
        if value is None or value == 'null' or value == 'None' or value == '':
            return default
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 't', 'y')
        return bool(value)

    def _safe_string(self, value, default=''):
        """Convertit en toute sécurité une valeur en chaîne"""
        if value is None or value == 'null' or value == 'None':
            return default
        return str(value)


    def generate_response(self, prompt: str, user_id: str = None, template_type: str = "sales") -> tuple:
        try:
            print("\n=== DÉBUT GENERATE_RESPONSE ===")
            print(f"Prompt reçu: {prompt}")
            print(f"User ID: {user_id}")
            print(f"Template type: {template_type}")
            debug_info = DebugInfo()

            if self.debug_mode:
                debug_info.executionLogs.append(
                    LogEntry(type="info", message=f"Début de génération de réponse pour: {prompt}"))
                debug_info.executionLogs.append(LogEntry(type="info", message=f"Utilisateur: {user_id}"))
                debug_info.executionLogs.append(
                    LogEntry(type="info", message=f"Mode de sécurité: {self.security_mode}"))

            # IMPORTANT: Vérification de sécurité explicite en premier
            # print("\n=== VÉRIFICATION DE SÉCURITÉ ===")
            # try:
            #     if hasattr(self, 'safety_checker') and self.safety_checker is not None:
            #         # Vérification directe avec PromptGuard pour le débogage
            #         if hasattr(self.safety_checker, 'prompt_guard') and self.safety_checker.prompt_guard:
            #             print("Test direct avec PromptGuard...")
            #             pg_result = self.safety_checker.prompt_guard.check_content(prompt, check_third_party=False)
            #             print(f"Résultat PromptGuard direct: {pg_result}")
            #             if not pg_result.get('is_safe', True):
            #                 print("!!! ALERTE: Contenu non sécurisé détecté par PromptGuard")
            #                 return "Je ne peux pas répondre à cette demande car elle pourrait compromettre la sécurité du système.",debug_info
            #
            #         # Vérification complète avec SafetyChecker
            #         safety_result = self.check_input_safety(prompt)
            #         print(f"Résultat final de sécurité: {safety_result}")
            #         if not safety_result.get('is_safe', True):
            #             print(f"ALERTE: Contenu non sécurisé détecté: {safety_result}")
            #             return "Je ne peux pas répondre à cette demande car elle contient du contenu potentiellement problématique ou sensible.",debug_info
            #     else:
            #         print("WARNING: SafetyChecker non initialisé!")
            # except Exception as e:
            #     print(f"ERREUR dans la vérification de sécurité: {e}")
            logger.info(f"Generating response for user {user_id} in {self.security_mode} mode")
            self.function_handlers.set_current_user(user_id)

            # Construction du message initial avec les rôles corrects
            decision_message = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(user_id, self.security_mode)
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            print("\n=== APPEL INITIAL À OLLAMA ===")

            # Premier appel pour choisir la fonction
            decision_response = self.ollama_instance.chat(
                model=self.model,
                messages=decision_message,
                tools=self._get_tools(),
                options={
                    "num_ctx": 8192  # Augmenter la taille du contexte
                }
            )
            if self.debug_mode:
                debug_info.executionLogs.append(LogEntry(type="info", message="=== APPEL INITIAL À OLLAMA ==="))
            print(f"First decision response: {decision_response['message']}")

            if self.debug_mode:
                decision_str = str(decision_response['message'])
                debug_info.firstDecision = decision_str
                debug_info.executionLogs.append(LogEntry(type="info", message=f"Décision initiale du modèle: {decision_str[:200]}..."))
            decision_message.append(decision_response["message"])

            # Gestion de la réponse
            if "message" in decision_response:
                if "tool_calls" in decision_response["message"]:
                    # Wrapper functions pour gérer les conversions de type de façon sécurisée
                    get_customer_emails_wrapper = lambda limit=50, domain_filter=None, anonymize=False, **kwargs: (
                        self.rag_manager.get_customer_emails(
                            self._safe_int(limit, 50),
                            self._safe_string(domain_filter, None) if domain_filter else None,
                            self._safe_bool(anonymize, False)
                        ) if self.security_mode == "UNSAFE" else
                        self.rag_manager.get_customer_emails_safe(
                            user_id,
                            self._safe_bool(anonymize, True)
                        )
                    )

                    get_orders_by_amount_wrapper = lambda min_amount=None, max_amount=None, limit=50, **kwargs: (
                        self.rag_manager.get_orders_by_amount(
                            self._safe_float(min_amount),
                            self._safe_float(max_amount),
                            self._safe_int(limit, 50)
                        ) if self.security_mode == "UNSAFE" else
                        self.rag_manager.get_orders_by_amount_safe(
                            user_id,
                            self._safe_float(min_amount),
                            self._safe_float(max_amount),
                            self._safe_int(limit, 50)
                        )
                    )

                    get_low_stock_products_wrapper = lambda threshold=20, category=None, limit=50, **kwargs: (
                        self.rag_manager.get_low_stock_products(
                            self._safe_int(threshold, 20),
                            self._safe_string(category, None) if category and category != '*' else None,
                            self._safe_int(limit, 50)
                        )
                    )

                    get_customer_addresses_wrapper = lambda city=None, country=None, postal_code=None, limit=50,anonymize=False, **kwargs: (
                        self.rag_manager.get_customer_addresses(
                            self._safe_string(city),
                            self._safe_string(country),
                            self._safe_string(postal_code),
                            self._safe_int(limit, 50),
                            self._safe_bool(anonymize, False)
                        ) if self.security_mode == "UNSAFE" else
                        self.rag_manager.get_customer_addresses_safe(
                            user_id,
                            self._safe_string(city),
                            self._safe_string(country),
                            self._safe_string(postal_code),
                            self._safe_int(limit, 50),
                            self._safe_bool(anonymize, True)
                        )
                    )

                    execute_sql_query_wrapper = lambda query, limit=100, **kwargs: (
                        self.rag_manager.execute_sql_query(
                            query,
                            limit=self._safe_int(limit, 100)
                        ) if self.security_mode == "UNSAFE" else
                        self.rag_manager.execute_sql_query_safe(
                            query,
                            limit=self._safe_int(limit, 20)
                        )
                    )

                    available_functions = {
                        "search_products": lambda query: self.rag_manager.query_index_no_llm(query),
                        "track_order": (
                            self.function_handlers.track_order_unsafe
                            if self.security_mode == "UNSAFE"
                            else self.function_handlers.track_order_safe
                        ),
                        "get_user_info": (
                            self.function_handlers.get_user_info_unsafe
                            if self.security_mode == "UNSAFE"
                            else self.function_handlers.get_user_info_safe
                        ),
                        "create_support_ticket": (
                            self.function_handlers.create_ticket_unsafe
                            if self.security_mode == "UNSAFE"
                            else self.function_handlers.create_ticket_safe
                        ),
                        # Utiliser les wrappers au lieu des fonctions directes
                        "execute_sql_query": execute_sql_query_wrapper,
                        "get_customer_emails": get_customer_emails_wrapper,
                        "get_orders_by_amount": get_orders_by_amount_wrapper,
                        "get_low_stock_products": get_low_stock_products_wrapper,
                        "get_customer_addresses": get_customer_addresses_wrapper,
                        "get_sales_statistics": lambda period=None, group_by='month', category=None, limit=50, **kwargs:
                        self.rag_manager.get_sales_statistics(
                            self._safe_string(period),
                            self._safe_string(group_by, 'month'),
                            self._safe_string(category),
                            self._safe_int(limit, 50)
                        ),
                        "analyze_email_domains": lambda limit=20, **kwargs:
                        self.rag_manager.analyze_email_domains(
                            self._safe_int(limit, 20)
                        ),
                        "get_postal_code_distribution": lambda country=None, limit=20, **kwargs:
                        self.rag_manager.get_postal_code_distribution(
                            self._safe_string(country),
                            self._safe_int(limit, 20)
                        )
                    }

                    for tool in decision_response["message"]["tool_calls"]:
                        function_name = tool["function"]["name"]
                        function_to_call = available_functions.get(function_name)

                        if function_to_call is None:
                            print(f"Fonction inconnue: {function_name}")
                            function_response = {"error": f"Fonction {function_name} non disponible"}
                        else:
                            # Extraire les arguments
                            if isinstance(tool["function"]["arguments"], str):
                                try:
                                    function_args = json.loads(tool["function"]["arguments"])
                                except json.JSONDecodeError:
                                    print(
                                        f"Erreur de décodage JSON pour les arguments: {tool['function']['arguments']}")
                                    function_args = {}
                            else:
                                function_args = tool["function"]["arguments"]

                            print(f"Function: {function_name}, Args: {function_args}")

                            # Gérer les différentes fonctions
                            try:
                                function_response = function_to_call(**function_args)
                            except Exception as func_error:
                                print(f"Erreur lors de l'exécution de la fonction {function_name}: {func_error}")
                                function_response = {
                                    "error": f"Erreur lors de l'exécution de la fonction: {str(func_error)}"
                                }

                        # Formater la réponse pour une meilleure lisibilité
                        function_response_str = self._format_function_response(function_name, function_response)

                        # Ajouter la réponse à la conversation
                        decision_message.append({
                            "role": "tool",
                            "content": function_response_str
                        })
                        print(f"Final function response added for {function_name}")

                        if self.debug_mode:
                            debug_info.executionLogs.append(LogEntry(type="info", message=f"Final function response added for {function_name}"))
                    # Second appel API pour obtenir la réponse finale
                    final_response = self.ollama_instance.chat(
                        model=self.model,
                        messages=decision_message,
                        options={
                            "num_ctx": 8192  # Augmenter la taille du contexte
                        }
                    )
                    print("reponse après function call")
                    print(final_response["message"]["content"])
                    response_content = final_response["message"]["content"]
                    if self.debug_mode:

                        debug_info.finalResponse = response_content
                        debug_info.executionLogs.append(LogEntry(type="info",
                                                                 message=f"Réponse finale générée ({len(response_content)} caractères)"))
                    # return final_response["message"]["content"]
                    return response_content, debug_info
                else:
                    print("The model didn't use any function call")
                    return decision_response["message"]["content"]
            else:
                return f"Désolé, je n'ai pas pu générer une réponse appropriée. {decision_response}",debug_info

        except Exception as e:
            print("\n=== ERREUR DÉTECTÉE ===")
            print(f"Type d'erreur: {type(e)}")
            print(f"Message d'erreur: {str(e)}")
            print("Arguments de la fonction:", function_args if 'function_args' in locals() else "Non disponible")
            traceback.print_exc()
            logger.error(f"Error in generate_response: {str(e)}\n{traceback.format_exc()}")
            return f"Une erreur s'est produite : {str(e)}", "une erreur s'est produite"