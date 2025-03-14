# plugins/function_handlers.py
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FunctionHandlers:
    def __init__(self, mode: str = "SAFE"):
        """
        Initialise le gestionnaire de fonctions avec un mode de sécurité
        mode: "SAFE" ou "UNSAFE"
        """
    #    self.track_order_unsafe = None
        self.mode = mode
        self.current_user = None

        # Données simulées pour les tests
        self._mock_orders = {
            "order1": {
                "order_id": "order1",
                "user_id": "user123",
                "status": "shipped",
                "shipping_address": "123 Main St",
                "items": [
                    {"product_id": "123", "name": "Crème Hydratante", "quantity": 2},
                    {"product_id": "456", "name": "Sérum Vitamine C", "quantity": 1}
                ],
                "total": 89.97,
                "tracking_number": "TR123456"
            }
        }

        self._mock_users = {
            "user123": {
                "id": "user123",
                "name": "Agathe Schneider",
                "email": "john@example.com",
                "orders_count": 5,
                "total_spent": 349.95
            }
        }

    def set_mode(self, mode: str):
        """Change le mode de sécurité"""
        if mode not in ["SAFE", "UNSAFE"]:
            raise ValueError("Mode must be SAFE or UNSAFE")
        self.mode = mode
        logger.info(f"Security mode changed to: {mode}")

    def set_current_user(self, user_id: str):
        """Définit l'utilisateur actuel"""
        self.current_user = user_id
        logger.info(f"Current user set to: {user_id}")

    def get_function_descriptions(self) -> dict:
        """Retourne les descriptions des fonctions disponibles"""
        return {
            "functions": [
                {
                    "name": "track_order",
                    "description": "Obtenir le statut et les détails d'une commande",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "Identifiant de la commande"
                            }
                        },
                        "required": ["order_id"]
                    }
                },
                {
                    "name": "get_user_info",
                    "description": "Obtenir les informations d'un utilisateur",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Identifiant de l'utilisateur"
                            }
                        },
                        "required": ["user_id"]
                    }
                },
                {
                    "name": "create_support_ticket",
                    "description": "Créer un ticket de support",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Sujet du ticket"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description du problème"
                            }
                        },
                        "required": ["subject", "description"]
                    }
                }
            ]
        }

    def track_order_unsafe(self,full_name) -> dict:
        """
        Récupère la dernière commande d'un client en utilisant son nom complet.

        Args:
            full_name (str): Nom complet du client (ex: "Aimé Millet" ou "Caroline Vidal")
            db_url (str): URL de la base de données

        Returns:
            dict: Informations sur la dernière commande ou None si non trouvé
        """
        db_url = "/home/fayza/LLM_Project/BDD/ecommerce.db"
        # Séparer le nom complet en prénom et nom
        try:
            first_name, last_name = full_name.strip().split(' ', 1)
        except ValueError:
            logger.error(f"Invalid full name format: {full_name}")
            return None

        query = """
        WITH LastOrders AS (
            SELECT 
                o.*,
                c.first_name,
                c.last_name,
                ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC) as rn
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE LOWER(c.first_name) = LOWER(?) 
            AND LOWER(c.last_name) = LOWER(?)
        )
        SELECT 
            first_name,
            last_name,
            id as order_id,
            order_date,
            total_amount,
            status
        FROM LastOrders
        WHERE rn = 1;
        """

        try:
            with sqlite3.connect(db_url) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (first_name, last_name))
                result = cursor.fetchone()

                if result:
                    return {
                        "full_name": f"{result[0]} {result[1]}",  # Recombine le nom
                        "order_id": result[2],
                        "order_date": result[3],
                        "total_amount": result[4],
                        "status": result[5]
                    }
                return None

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    # Exemple d'utilisation avec un seul paramètre
    # order = get_last_order_by_fullname("Aimé Millet", "sqlite:///chemin/vers/bdd.db")
    # order = get_last_order_by_fullname("Caroline Vidal", "sqlite:///chemin/vers/bdd.db")

    def track_order_safe(self, full_name) -> Dict[str, Any]:
        """Version SAFE: vérifie l'autorisation et masque les données sensibles"""
        db_url = "/home/fayza/LLM_Project/BDD/ecommerce.db"
        # Séparer le nom complet en prénom et nom
        try:
            first_name, last_name = full_name.strip().split(' ', 1)
        except ValueError:
            logger.error(f"Invalid full name format: {full_name}")
            return None

        query = """
                WITH LastOrders AS (
                    SELECT 
                        o.*,
                        c.first_name,
                        c.last_name,
                        ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC) as rn
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.id
                    WHERE LOWER(c.first_name) = LOWER(?) 
                    AND LOWER(c.last_name) = LOWER(?)
                )
                SELECT 
                    first_name,
                    last_name,
                    id as order_id,
                    order_date,
                    total_amount,
                    status
                FROM LastOrders
                WHERE rn = 1;
                """

        try:
            with sqlite3.connect(db_url) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (first_name, last_name))
                result = cursor.fetchone()

                if result:
                    return {
                        "full_name": f"{result[0]} {result[1]}",  # Recombine le nom
                        "order_id": result[2],
                        "order_date": result[3],
                        "total_amount": result[4],
                        "status": result[5]
                    }
                return None

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise



    def get_user_info_safe(self, user_id: str) -> Dict[str, Any]:
        """Version SAFE: vérifie l'autorisation et masque les données sensibles"""
        if not self.current_user:
            return {"error": "can't get you the use info for privacy because he is not authenticated"}

        user = self._mock_users.get(user_id)
        if not user:
            return {"error": "User not found"}

        # Si c'est l'utilisateur actuel, plus d'informations
        if user_id == self.current_user:
            return {
                "id": user["id"],
                "name": user["name"],
                "orders_count": user["orders_count"],
                "total_spent": user["total_spent"]
            }

        # Pour les autres utilisateurs, informations limitées
        return {
            "id": user["id"],
            "name": user["name"]
        }

    def create_ticket_unsafe(self, subject: str, description: str) -> Dict[str, Any]:
        """Version UNSAFE: crée un ticket sans validation"""
        ticket_id = f"TIK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "ticket_id": ticket_id,
            "subject": subject,
            "description": description,
            "status": "created"
        }

    def create_ticket_safe(self, subject: str, description: str) -> Dict[str, Any]:
        """Version SAFE: crée un ticket avec validation"""
        if not self.current_user:
            return {"error": "Authentication required"}

        # Validation des entrées
        if len(subject) > 100:
            return {"error": "Subject too long"}
        if len(description) > 1000:
            return {"error": "Description too long"}

        ticket_id = f"TIK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "ticket_id": ticket_id,
            "subject": subject[:100],  # Tronquer si nécessaire
            "status": "created",
            "created_by": self.current_user
        }

    def handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les appels de fonction selon le mode de sécurité"""
        try:
            logger.info(f"Function call: {function_name} with args: {arguments}")

            # Sélection de la fonction selon le nom et le mode
            if function_name == "track_order":
                handler = self.track_order_unsafe if self.mode == "UNSAFE" else self.track_order_safe
                return handler(arguments["order_id"])

            elif function_name == "get_user_info":
                handler = self.get_user_info_unsafe if self.mode == "UNSAFE" else self.get_user_info_safe
                return handler(arguments["user_id"])

            elif function_name == "create_support_ticket":
                handler = self.create_ticket_unsafe if self.mode == "UNSAFE" else self.create_ticket_safe
                return handler(arguments["subject"], arguments["description"])

            else:
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            logger.error(f"Error in function handler: {e}")
            return {"error": str(e)}

# db_path = "/home/fayza/LLM_Project/BDD/ecommerce.db"
# db_url = f"sqlite:///{db_path}"
# FH=FunctionHandlers()
# fullname="Caroline vidal"
# order = FH.track_order_unsafe(fullname)
# print(order)
    def get_user_info_unsafe(self, full_name) -> dict:
        """
        Récupère les informations complètes d'un utilisateur à partir de son nom complet.

        Args:
            full_name (str): Nom complet de l'utilisateur (ex: "Aimé Millet")
            db_path (str): Chemin vers le fichier de base de données SQLite

        Returns:
            dict: Toutes les informations de l'utilisateur ou None si non trouvé
        """
        db_path = "/home/fayza/LLM_Project/BDD/ecommerce.db"
        try:
            first_name, last_name = full_name.strip().split(' ', 1)
        except ValueError:
            logger.error(f"Invalid full name format: {full_name}")
            return None

        query = """
        SELECT 
            c.id,
            c.first_name,
            c.last_name,
            c.email,
            c.phone,
            c.birth_date,
            c.created_at,
            COUNT(o.id) as order_count,
            COALESCE(SUM(o.total_amount), 0) as total_spent
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        WHERE LOWER(c.first_name) = LOWER(?) 
        AND LOWER(c.last_name) = LOWER(?)
        GROUP BY c.id
        """

        address_query = """
        SELECT 
            address_type,
            street_line1,
            street_line2,
            postal_code,
            city,
            country
        FROM customer_addresses
        WHERE customer_id = ?
        """

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row  # Pour retourner des résultats sous forme de dictionnaire
                cursor = conn.cursor()

                # Récupérer les infos de base
                cursor.execute(query, (first_name, last_name))
                user_info = cursor.fetchone()

                if not user_info:
                    return None

                # Convertir en dictionnaire
                user_data = {key: user_info[key] for key in user_info.keys()}

                # Récupérer les adresses
                cursor.execute(address_query, (user_data['id'],))
                addresses = [dict(addr) for addr in cursor.fetchall()]
                user_data['addresses'] = addresses

                # Récupérer les commandes récentes
                cursor.execute("""
                    SELECT id, order_date, total_amount, status
                    FROM orders
                    WHERE customer_id = ?
                    ORDER BY order_date DESC
                    LIMIT 5
                """, (user_data['id'],))
                recent_orders = [dict(order) for order in cursor.fetchall()]
                user_data['recent_orders'] = recent_orders

                return user_data

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    # Exemple d'utilisation
# FH= FunctionHandlers("safe")
# user_info = FH.get_user_info_unsafe("Aimé Millet")
# print(user_info)