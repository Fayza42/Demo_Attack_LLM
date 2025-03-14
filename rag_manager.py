import traceback
import sys
from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from typing import Optional
from contextlib import nullcontext
import torch
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.readers.database import DatabaseReader
from sqlalchemy import create_engine, text
import chromadb
import logging
from pathlib import Path
from datetime import datetime
logger = logging.getLogger(__name__)

def setup_logger():
    # Créer le logger
    logger = logging.getLogger('RAGManager')
    logger.setLevel(logging.DEBUG)

    # Créer un handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Créer un handler pour le fichier
    file_handler = logging.FileHandler('rag_search.log')
    file_handler.setLevel(logging.DEBUG)

    # Définir le format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Ajouter les handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
i=0
def get_ollama_instance():
    return Ollama(
        model="llama3.1:8b-instruct-fp16",
        temperature=0.1,
        context_window=4096,
        num_ctx=4096
    )
class RAGManager:
    def __init__(self, db_url: str, persist_dir: str = "/home/fayza/LLM_Project/backend/data/chroma"):
        try:
            # Configuration GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.empty_cache()

            # Création du dossier persist_dir
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            # Initialisation ChromaDB avec retry
            for attempt in range(3):
                try:
                  self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                  # self.chroma_client=chromadb.Client(
                  #     Settings(chroma_db_impl="duckdb+parquet"),
                  #     persist_directory=persist_dir
                  # )

                  self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name="pharma_products",
                        metadata={"updated_at": datetime.now().isoformat()}
                    )

                  break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")

            # Initialisation DB
            self.engine = create_engine(db_url)
            self.db_reader = DatabaseReader(engine=self.engine)
            # Configuration du LLM

            # Settings.llm = Ollama(
            #         model="llama3.1:8b-instruct-fp16",
            #         temperature=0.1,
            #         context_window=4096,
            #         num_ctx=4096
            #     )
            # Requête optimisée basée sur la structure Faker
            # Requêtes SQL pour chaque type d'information
            self.product_query = """
            WITH RECURSIVE CategoryPath AS (
                SELECT id, name as category_name, name as full_path, parent_id
                FROM categories WHERE parent_id IS NULL
                UNION ALL
                SELECT c.id, c.name, cp.full_path || ' > ' || c.name, c.parent_id
                FROM categories c JOIN CategoryPath cp ON c.parent_id = cp.id
            )
            SELECT 
                p.id,
                p.reference,
                p.name as product_name,
                p.description,
                p.price,
                p.weight_grams,
                p.active,
                cp.full_path as category_path,
                b.name as brand_name,      -- Renommez explicitement les colonnes de la table brands
                b.country as brand_origin,
                b.website as brand_website,
                i.quantity as stock_level,
                CASE 
                    WHEN i.quantity = 0 THEN 'Rupture de stock'
                    WHEN i.quantity < 10 THEN 'Stock faible'
                    ELSE 'En stock'
                END as stock_status
            FROM products p
            JOIN CategoryPath cp ON p.category_id = cp.id
            JOIN brands b ON p.brand_id = b.id
            LEFT JOIN inventory i ON p.id = i.product_id
            """

            self.customer_query = """
                    SELECT 
                        c.*,
                        ca.id as address_id,
                        ca.address_type,
                        ca.street_line1,
                        ca.street_line2,
                        ca.postal_code,
                        ca.city,
                        ca.country,
                        COUNT(o.id) as total_orders,
                        SUM(o.total_amount) as total_spent
                    FROM customers c
                    LEFT JOIN customer_addresses ca ON c.id = ca.customer_id
                    LEFT JOIN orders o ON c.id = o.customer_id
                    GROUP BY c.id, ca.id
                    """

            self.order_query = """
                    SELECT 
                        o.*,
                        c.first_name, c.last_name, c.email,
                        ca_ship.street_line1 as shipping_address,
                        ca_ship.city as shipping_city,
                        ca_bill.street_line1 as billing_address,
                        ca_bill.city as billing_city,
                        GROUP_CONCAT(p.name) as products_list
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.id
                    JOIN customer_addresses ca_ship ON o.shipping_address_id = ca_ship.id
                    JOIN customer_addresses ca_bill ON o.billing_address_id = ca_bill.id
                    JOIN order_items oi ON o.id = oi.order_id
                    JOIN products p ON oi.product_id = p.id
                    GROUP BY o.id
                    """
            self.query = """
            WITH RECURSIVE CategoryPath AS (
                SELECT 
                    c.id,
                    c.name as category_name,
                    c.name as full_path,
                    c.parent_id
                FROM categories c
                WHERE c.parent_id IS NULL

                UNION ALL

                SELECT
                    c.id,
                    c.name,
                    cp.full_path || ' > ' || c.name,
                    c.parent_id
                FROM categories c
                JOIN CategoryPath cp ON c.parent_id = cp.id
            )
            SELECT 
                p.id,
                p.reference,
                p.name as product_name,
                p.description as product_description,
                p.price,
                p.weight_grams,
                p.active,
                cp.full_path as category_path,
                b.name as brand_name,
                b.country as brand_origin,
                b.website as brand_website,
                i.quantity as stock_level,
                CASE 
                    WHEN i.quantity = 0 THEN 'Rupture de stock'
                    WHEN i.quantity < 10 THEN 'Stock faible'
                    ELSE 'En stock'
                END as stock_status,
                CASE 
                    WHEN p.price < 10 THEN 'Entrée de gamme'
                    WHEN p.price < 30 THEN 'Milieu de gamme'
                    ELSE 'Premium'
                END as price_category
            FROM products p
            JOIN CategoryPath cp ON p.category_id = cp.id
            JOIN brands b ON p.brand_id = b.id
            LEFT JOIN inventory i ON p.id = i.product_id
            WHERE p.active = 1
            """

            # Requête pour les catégories
            self.category_query = """
                    WITH RECURSIVE CategoryPath AS (
                        SELECT 
                            c.id, 
                            c.name as category_name,
                            c.name as full_path,
                            c.description,
                            c.parent_id,
                            NULL as parent_name
                        FROM categories c
                        WHERE c.parent_id IS NULL
                        UNION ALL
                        SELECT 
                            c.id,
                            c.name,
                            cp.full_path || ' > ' || c.name,
                            c.description,
                            c.parent_id,
                            cp.category_name as parent_name
                        FROM categories c
                        JOIN CategoryPath cp ON c.parent_id = cp.id
                    )
                    SELECT 
                        cp.*,
                        COUNT(p.id) as product_count,
                        GROUP_CONCAT(p.name) as product_examples
                    FROM CategoryPath cp
                    LEFT JOIN products p ON p.category_id = cp.id
                    GROUP BY cp.id
                    """

            # Requête pour les marques
            self.brand_query = """
                    SELECT 
                        b.*,
                        COUNT(p.id) as total_products,
                        AVG(p.price) as avg_price,
                        GROUP_CONCAT(DISTINCT c.name) as categories_served
                    FROM brands b
                    LEFT JOIN products p ON b.id = p.brand_id
                    LEFT JOIN categories c ON p.category_id = c.id
                    GROUP BY b.id
                    """

            # Requête pour l'inventaire
            self.inventory_query = """
                    SELECT 
                        i.*,
                        p.name as product_name,
                        p.reference,
                        b.name as brand_name
                    FROM inventory i
                    JOIN products p ON i.product_id = p.id
                    JOIN brands b ON p.brand_id = b.id
                    """

            # Initialisation vector store
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Modèle d'embedding optimisé
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device=self.device,
                model_kwargs={
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
                }
            )

            # Configuration Settings
            Settings.embed_model = self.embed_model
            Settings.llm = Ollama(
                model="llama3.1:8b-instruct-fp16",
                temperature=0.1,  # Réduit pour plus de précision
                context_window=4096,
                num_ctx=4096,
            )
            Settings.chunk_size = 1024  # Augmenté pour meilleur contexte
            Settings.chunk_overlap = 128

            self.index = None
            logger.info("RAG Manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG Manager: {e}")
            raise

    def format_document_text(self, row) -> str:
        """Format optimisé pour les données Faker avec emphase sur les prix exacts"""
        try:
            return f"""
    [PRODUIT {row['reference']}]
    NOM EXACT: {row['product_name']}
    PRIX EXACT EN EUROS: {row['price']}
    MARQUE: {row['brand_name']} ({row['brand_origin']})
    CATÉGORIE: {row['category_path']}
    QUANTITÉ EN STOCK EXACTE: {row['stock_level']} unités
    STATUT STOCK: {row['stock_status']}

    [DESCRIPTION PRODUIT]
    {row['product_description']}

    [CLASSIFICATION]
    Gamme de prix: {row['price_category']}
    Type de produit: {row['category_path'].split(' > ')[-1] if ' > ' in row['category_path'] else row['category_path']}

    [MOTS-CLÉS DE RECHERCHE]
    {row['brand_name'].lower()}, {row['category_path'].lower().replace(' > ', ', ')}, {row['price_category'].lower()}, {row['stock_status'].lower()}

    [FIN FICHE PRODUIT]
    """
        except Exception as e:
            logger.error(f"Error formatting document for {row.get('product_name', 'Unknown')}: {e}")
            return ""

    def format_category_document(self, row) -> str:
            """Format pour les documents de type catégorie"""
            try:
                return f"""
    [TYPE: CATÉGORIE]
    ID: {row['id']}
    Nom: {row['category_name']}
    Chemin complet: {row['full_path']}
    Description: {row['description']}

    Structure:
    - Catégorie parente: {row['parent_name'] if row['parent_name'] else 'Catégorie principale'}
    - ID parent: {row['parent_id'] if row['parent_id'] else 'Aucun'}

    Statistiques:
    - Nombre de produits: {row['product_count']}
    - Exemples de produits: {row['product_examples']}

    Mots-clés de recherche: {row['category_name'].lower()}, {row['full_path'].lower().replace(' > ', ' ')}
    """
            except Exception as e:
                logger.error(f"Error formatting category document: {e}")
                return ""

    def format_brand_document(self, row) -> str:
            """Format pour les documents de type marque"""
            try:
                return f"""
    [TYPE: MARQUE]
    ID: {row['id']}
    Nom: {row['name']}
    Pays d'origine: {row['country']}
    Site web: {row['website']}

    Statistiques:
    - Nombre total de produits: {row['total_products']}
    - Prix moyen des produits: {row['avg_price']}€
    - Catégories couvertes: {row['categories_served']}

    Mots-clés de recherche: {row['name'].lower()}, {row['country'].lower()}
    """
            except Exception as e:
                logger.error(f"Error formatting brand document: {e}")
                return ""

    def format_inventory_document(self, row) -> str:
            """Format pour les documents de type inventaire"""
            try:
                return f"""
    [TYPE: INVENTAIRE]
    ID Produit: {row['product_id']}
    Référence produit: {row['reference']}
    Nom produit: {row['product_name']}
    Marque: {row['brand_name']}
    Quantité en stock: {row['quantity']}

    Statut stock: {'Rupture' if row['quantity'] == 0 else 'Faible' if row['quantity'] < 10 else 'Normal'}

    Mots-clés de recherche: {row['product_name'].lower()}, {row['brand_name'].lower()}, stock, inventaire
    """
            except Exception as e:
                logger.error(f"Error formatting inventory document: {e}")
                return ""

    def format_product_document(self, row) -> str:
        """Format pour les documents de type produit"""
        try:
            return f"""
    [TYPE: PRODUIT]
    ID: {row['id']}
    Référence: {row['reference']}
    Nom: {row['product_name']}      # Changé de 'name' à 'product_name'
    Description: {row['description']}

    Prix et Stock:
    - Prix: {row['price']}€
    - Niveau de stock: {row['stock_level']} unités
    - Statut stock: {row['stock_status']}
    - Poids: {row['weight_grams']}g

    Marque et Catégorie:
    - Marque: {row['brand_name']}
    - Origine: {row['brand_origin']}
    - Site web: {row['brand_website']}
    - Catégorie: {row['category_path']}

    Mots-clés de recherche: {row['product_name'].lower()}, {row['brand_name'].lower()}, {row['category_path'].lower().replace(' > ', ' ')}
    """
        except Exception as e:
            logger.error(f"Error formatting product document: {e}, available keys: {row.keys()}")
            return ""

    def format_customer_document(self, row) -> str:
        """Format pour les documents de type client"""
        try:
            address = f"""
    Adresse ({row['address_type']}):
    {row['street_line1']}
    {row['street_line2'] if row['street_line2'] else ''}
    {row['postal_code']} {row['city']}
    {row['country']}"""

            orders_info = f"""
    Historique commandes:
    - Nombre total de commandes: {row['total_orders']}
    - Montant total dépensé: {row['total_spent']}€"""

            return f"""
    [TYPE: CLIENT]
    ID: {row['id']}
    Nom complet: {row['first_name']} {row['last_name']}
    Email: {row['email']}
    Téléphone: {row['phone']}
    Date de naissance: {row['birth_date']}
    Date d'inscription: {row['created_at']}

    {address}

    {orders_info}

    Mots-clés de recherche: {row['email'].lower()}, {row['first_name'].lower()}, {row['last_name'].lower()}, {row['postal_code']}, {row['city'].lower()}
    """
        except Exception as e:
            logger.error(f"Error formatting customer document: {e}")
            return ""

    def format_order_document(self, row) -> str:
        """Format pour les documents de type commande"""
        try:
            return f"""
    [TYPE: COMMANDE]
    ID Commande: {row['id']}
    Date: {row['order_date']}
    Statut: {row['status']}

    Client:
    - Nom: {row['first_name']} {row['last_name']}
    - Email: {row['email']}

    Livraison:
    Adresse: {row['shipping_address']}
    Ville: {row['shipping_city']}

    Facturation:
    Adresse: {row['billing_address']}
    Ville: {row['billing_city']}

    Détails commande:
    - Montant total: {row['total_amount']}€
    - Frais de livraison: {row['shipping_fee']}€
    - Produits: {row['products_list']}

    Mots-clés de recherche: {row['email'].lower()}, commande_{row['id']}, {row['status'].lower()}, {row['shipping_city'].lower()}
    """
        except Exception as e:
            logger.error(f"Error formatting order document: {e}")
            return ""


    def build_index(self) -> bool:
        """Construction de l'index avec vérification des données"""
        try:
            with torch.cuda.device(0) if self.device == "cuda" else nullcontext():
                logger.info("Starting index building...")

                # Exécution directe de la requête avec SQLAlchemy
                documents = []

                # Documents produits
                with self.engine.connect() as conn:
                    products = conn.execute(text(self.product_query)).fetchall()
                    for row in products:
                        doc = Document(
                            text=self.format_product_document(dict(row._mapping)),
                            metadata={"type": "product"}
                        )
                        documents.append(doc)

                    # Documents clients
                    customers = conn.execute(text(self.customer_query)).fetchall()
                    for row in customers:
                        doc = Document(
                            text=self.format_customer_document(dict(row._mapping)),
                            metadata={"type": "customer"}
                        )
                        documents.append(doc)

                    # Documents commandes
                    orders = conn.execute(text(self.order_query)).fetchall()
                    for row in orders:
                        doc = Document(
                            text=self.format_order_document(dict(row._mapping)),
                            metadata={"type": "order"}
                        )
                        documents.append(doc)
                        # Documents catégories
                        categories = conn.execute(text(self.category_query)).fetchall()
                        for row in categories:
                            doc = Document(
                                text=self.format_category_document(dict(row._mapping)),
                                metadata={"type": "category"}
                            )
                            documents.append(doc)

                        # Documents marques
                        brands = conn.execute(text(self.brand_query)).fetchall()
                        for row in brands:
                            doc = Document(
                                text=self.format_brand_document(dict(row._mapping)),
                                metadata={"type": "brand"}
                            )
                            documents.append(doc)

                        # Documents inventaire
                        inventory = conn.execute(text(self.inventory_query)).fetchall()
                        for row in inventory:
                            doc = Document(
                                text=self.format_inventory_document(dict(row._mapping)),
                                metadata={"type": "inventory"}
                            )
                            documents.append(doc)

                    self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
                    return True



        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False

    def query_index(self, query_text: str) -> Optional[str]:
        """Version simplifiée de query_index sans timeout"""
        try:
            print(f"\n=== DÉBUT QUERY_INDEX ===")
            print(f"Query reçue: {query_text}")

            if not self.get_or_create_index():
                print("❌ Échec get_or_create_index")
                return None

            print("✅ Index obtenu avec succès")

            query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize",
                streaming=False,
            )
            print("✅ Query engine configuré")

            print(f"\nExécution de la requête...")
            response = query_engine.query(query_text)
            print(f"✅ Réponse obtenue")

            return str(response)

        except Exception as e:
            print(f"\n=== ERREUR DANS QUERY_INDEX ===")
            print(f"Type: {type(e)}")
            print(f"Message: {str(e)}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            return "Une erreur est survenue lors de la recherche"

    def get_or_create_index(self) -> bool:
        """Récupération ou création de l'index avec vérification"""
        try:
            if self.index:
                return True

            if self.load_index():
                return True

            logger.info("Creating new index...")
            return self.build_index()

        except Exception as e:
            logger.error(f"Error in get_or_create_index: {e}")
            return False

    def load_index(self) -> bool:
        """Chargement de l'index existant"""
        try:
            # Utiliser le storage_context déjà initialisé dans __init__
            self.index = load_index_from_storage(self.storage_context)
            logger.info("Successfully loaded existing index")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def check_database_connection(self):
        try:
            with self.engine.connect() as conn:
                # Utiliser text() pour convertir la requête SQL en objet exécutable
                query = text("""
                    SELECT COUNT(*) 
                    FROM products p 
                    JOIN brands b ON p.brand_id = b.id 
                    WHERE b.name = 'La Roche-Posay'
                """)

                # Exécuter la requête
                result = conn.execute(query).scalar()
                logger.info(f"Found {result} La Roche-Posay products in database")
                return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def check_chroma_status(self):
        try:
            collection_count = len(self.chroma_collection.get()['ids'])
            logger.info(f"ChromaDB collection has {collection_count} documents")
            return collection_count > 0
        except Exception as e:
            logger.error(f"ChromaDB error: {e}")
            return False

    def query_index_no_llm(self, query_text: str) -> Optional[str]:
        """Version améliorée qui gère tous les types de documents avec leurs relations"""
        logger = setup_logger()

        try:
            print(f"\n=== DÉBUT QUERY_INDEX ===")
            print(f"Query reçue: {query_text}")

            if not self.get_or_create_index():
                print("❌ Échec get_or_create_index")
                return None

            # Configuration du retriever
            retriever = self.index.as_retriever(
                similarity_top_k=10  # Plus de résultats pour couvrir tous les types
            )

            # Récupération des documents
            nodes = retriever.retrieve(query_text)
            logger.info(f"Nombre de documents récupérés: {len(nodes)}")

            # Organiser les résultats par type
            results = {
                "PRODUIT": [],
                "CLIENT": [],
                "COMMANDE": [],
                "CATÉGORIE": [],
                "MARQUE": [],
                "INVENTAIRE": []
            }

            # Filtrer et organiser les résultats
            for node in nodes:
                score = float(node.score) if hasattr(node, 'score') else 0
                i=0
                logger.debug(f"Document {i + 1} - Score: {score:.4f}")
                logger.debug(f"Contenu: {node.node.text[:100]}...")
                if score < 0.25:  # Seuil minimal de pertinence
                    continue

                content = node.node.text
                # Déterminer le type et ajouter aux résultats correspondants
                for doc_type in results.keys():
                    if f"[TYPE: {doc_type}]" in content:
                        results[doc_type].append((score, content))
                        break

            # Formater la réponse
            response_parts = []

            # Traiter chaque type de document
            for doc_type, docs in results.items():
                if docs:  # Si nous avons des résultats pour ce type
                    # Trier par score de pertinence
                    docs.sort(key=lambda x: x[0], reverse=True)

                    # Ajouter l'en-tête de section
                    response_parts.append(f"\n{'=' * 20} RÉSULTATS {doc_type} {'=' * 20}")

                    # Ajouter les 3 résultats les plus pertinents
                    for i, (score, content) in enumerate(docs[:3], 1):
                        response_parts.append(f"\nRésultat {i} (Score: {score:.2f})")
                        response_parts.append(content.strip())  # Retirer les espaces inutiles
                        response_parts.append("-" * 50)

                    # Si plus de résultats disponibles
                    if len(docs) > 3:
                        response_parts.append(f"\n... et {len(docs) - 3} autres résultats pour {doc_type}")

            if not response_parts:
                return "Aucun résultat pertinent trouvé pour votre recherche."

            # Ajouter un résumé en début de réponse
            summary_parts = []
            for doc_type, docs in results.items():
                if docs:
                    summary_parts.append(f"{doc_type}: {len(docs)} résultat(s)")

            logger.info(f"Query normalisée: {query_text.lower().strip()}")
         #   logger.info(f"Dimension du vecteur d'embedding: {len(self.embed_model.encode(query_text))}")
         #   logger.info(f"Temps de recherche: {end_time - start_time:.2f} secondes")

            if summary_parts:
                response_parts.insert(0, "=== RÉSUMÉ DE LA RECHERCHE ===")
                response_parts.insert(1, "\n".join(summary_parts))
                response_parts.insert(2, "=" * 50 + "\n")

            return "\n".join(response_parts)

        except Exception as e:
            print(f"\n=== ERREUR DANS QUERY_INDEX ===")
            print(f"Type: {type(e)}")
            print(f"Message: {str(e)}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            return "Une erreur est survenue lors de la recherche"

    def execute_sql_query(self, query_str, params=None, limit=100):
        """
        Exécute une requête SQL directe avec quelques vérifications de base.

        Args:
            query_str (str): La requête SQL à exécuter
            params (list, tuple or dict, optional): Paramètres pour la requête SQL
            limit (int, optional): Nombre maximum de résultats à retourner

        Returns:
            list: Résultats de la requête SQL
        """
        try:
            print(f"Exécution de execute_sql_query avec query={query_str}, params={params}")
            with self.engine.connect() as conn:
                # On ajoute une limite si elle n'est pas dans la requête
                if "LIMIT" not in query_str.upper():
                    query_str = f"{query_str} LIMIT {limit}"

                # Exécution de la requête
                if params:
                    # Modification pour corriger le problème de formatage des paramètres
                    result = conn.execute(text(query_str), params).fetchall()
                else:
                    result = conn.execute(text(query_str)).fetchall()

                # Vérifier si on a des résultats
                if not result:
                    print("Aucun résultat retourné par la requête SQL")
                    return []

                # Conversion en liste de dictionnaires
                formatted_result = []
                for row in result:
                    row_dict = dict(row._mapping)
                    # Conversion des types non-sérialisables (datetime, Decimal, etc.)
                    for key, value in row_dict.items():
                        if hasattr(value, 'isoformat'):  # Pour les dates et heures
                            row_dict[key] = value.isoformat()
                        elif hasattr(value, '__float__'):  # Pour les Decimal
                            row_dict[key] = float(value)
                    formatted_result.append(row_dict)

                print(f"Nombre de résultats: {len(formatted_result)}")
                if formatted_result:
                    print(f"Premier résultat (exemple): {formatted_result[0]}")

                return formatted_result

        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": error_msg}

    def get_low_stock_products(self, threshold=10, category=None, limit=50):
        """
        Récupère les produits dont le stock est inférieur à un seuil.

        Args:
            threshold (int): Seuil de stock
            category (str, optional): Filtrer par catégorie
            limit (int): Nombre maximum de résultats

        Returns:
            list: Liste des produits à faible stock
        """
        try:
            # Construction de la requête de base
            query = """
            SELECT 
                p.id,
                p.reference,
                p.name AS product_name,
                p.description,
                p.price,
                b.name AS brand_name,
                c.name AS category_name,
                i.quantity AS stock_level,
                CASE 
                    WHEN i.quantity = 0 THEN 'Rupture de stock'
                    WHEN i.quantity < 10 THEN 'Stock critique'
                    ELSE 'Stock faible'
                END AS stock_status
            FROM products p
            JOIN inventory i ON p.id = i.product_id
            JOIN brands b ON p.brand_id = b.id
            JOIN categories c ON p.category_id = c.id
            WHERE i.quantity <= :threshold
            """

            # Paramètres sous forme de dictionnaire
            params = {"threshold": threshold}

            # Ajout du filtre par catégorie si nécessaire
            if category and category != '*':
                query += " AND c.name LIKE :category"
                params["category"] = f"%{category}%"

            # Ajout de l'ordre et de la limite
            query += " ORDER BY i.quantity ASC LIMIT :limit"
            params["limit"] = limit

            return self.execute_sql_query(query, params)

        except Exception as e:
            logger.error(f"Error in get_low_stock_products: {e}")
            return {"error": str(e)}

    def get_customer_addresses(self, city=None, country=None, postal_code=None, limit=50, anonymize=False):
        """
        Récupère les adresses clients avec diverses options de filtrage.

        Args:
            city (str, optional): Filtrer par ville
            country (str, optional): Filtrer par pays
            postal_code (str, optional): Filtrer par code postal
            limit (int): Nombre maximum de résultats
            anonymize (bool): Si True, masque partiellement les adresses

        Returns:
            list: Liste des adresses clients
        """
        try:
            # Construction de la requête de base
            query = """
            SELECT 
                ca.id,
                ca.customer_id,
                ca.address_type,
                ca.street_line1,
                ca.street_line2,
                ca.postal_code,
                ca.city,
                ca.country,
                c.first_name,
                c.last_name,
                c.email
            FROM customer_addresses ca
            JOIN customers c ON ca.customer_id = c.id
            WHERE 1=1
            """

            # Paramètres sous forme de dictionnaire
            params = {}

            # Ajout des filtres si nécessaires
            if city:
                query += " AND ca.city LIKE :city"
                params["city"] = f"%{city}%"

            if country:
                query += " AND ca.country LIKE :country"
                params["country"] = f"%{country}%"

            if postal_code:
                query += " AND ca.postal_code LIKE :postal_code"
                params["postal_code"] = f"%{postal_code}%"

            # Ajout de la limite
            if limit > 0:
                query += " LIMIT :limit"
                params["limit"] = limit

            # Exécution de la requête
            results = self.execute_sql_query(query, params)

            # Anonymisation si demandée
            if anonymize and results:
                for item in results:
                    if 'street_line1' in item and item['street_line1']:
                        parts = item['street_line1'].split(' ', 1)
                        if len(parts) > 1:
                            item['street_line1'] = parts[0] + ' ' + '***'

                    if 'email' in item and item['email']:
                        email = item['email']
                        at_pos = email.find('@')
                        if at_pos > 0:
                            username = email[:at_pos]
                            domain = email[at_pos:]
                            if len(username) > 2:
                                item['email'] = username[:2] + '***' + domain

            return results

        except Exception as e:
            logger.error(f"Error in get_customer_addresses: {e}")
            return {"error": str(e)}

    def get_orders_by_amount(self, min_amount=None, max_amount=None, limit=50):
        """
        Récupère les commandes filtrées par montant.

        Args:
            min_amount (float, optional): Montant minimum
            max_amount (float, optional): Montant maximum
            limit (int): Nombre maximum de résultats

        Returns:
            list: Liste des commandes
        """
        try:
            # Construction de la requête de base
            query = """
            SELECT 
                o.id,
                o.order_date,
                o.total_amount,
                o.status,
                c.first_name,
                c.last_name,
                c.email
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE 1=1
            """

            # Paramètres sous forme de dictionnaire
            params = {}

            # Ajout des filtres si nécessaires
            if min_amount is not None:
                query += " AND o.total_amount >= :min_amount"
                params["min_amount"] = min_amount

            if max_amount is not None:
                query += " AND o.total_amount <= :max_amount"
                params["max_amount"] = max_amount

            # Ajout de l'ordre et de la limite
            query += " ORDER BY o.total_amount DESC LIMIT :limit"
            params["limit"] = limit

            return self.execute_sql_query(query, params)

        except Exception as e:
            logger.error(f"Error in get_orders_by_amount: {e}")
            return {"error": str(e)}

    def get_orders_by_amount_safe(self, current_user, min_amount=None, max_amount=None, limit=50):
        """
        Version sécurisée qui récupère uniquement les commandes de l'utilisateur connecté filtrées par montant.

        Args:
            current_user (str): Nom complet de l'utilisateur connecté (format: "Prénom Nom")
            min_amount (float, optional): Montant minimum
            max_amount (float, optional): Montant maximum
            limit (int): Nombre maximum de résultats

        Returns:
            list: Liste des commandes de l'utilisateur
        """
        try:
            # Vérification que l'utilisateur est spécifié
            if not current_user:
                logger.error("Tentative d'accès sans identifiant utilisateur")
                return {"error": "Authentification requise pour accéder aux commandes"}

            # Extraction du prénom et du nom
            try:
                first_name, last_name = current_user.strip().split(' ', 1)
            except ValueError:
                logger.error(f"Format de nom d'utilisateur invalide: {current_user}")
                return {"error": "Format de nom d'utilisateur invalide"}

            # Construction de la requête sécurisée
            query = """
            SELECT 
                o.id,
                o.order_date,
                o.total_amount,
                o.status,
                c.first_name,
                c.last_name,
                c.email
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE c.first_name LIKE :first_name 
            AND c.last_name LIKE :last_name
            """

            # Paramètres sous forme de dictionnaire avec les informations de l'utilisateur
            params = {
                "first_name": first_name,
                "last_name": last_name
            }

            # Ajout des filtres de montant si nécessaires
            if min_amount is not None:
                query += " AND o.total_amount >= :min_amount"
                params["min_amount"] = min_amount

            if max_amount is not None:
                query += " AND o.total_amount <= :max_amount"
                params["max_amount"] = max_amount

            # Ajout de l'ordre et de la limite
            query += " ORDER BY o.order_date DESC LIMIT :limit"
            params["limit"] = limit

            # Exécution de la requête
            result = self.execute_sql_query(query, params)

            # Log de sécurité
            if result:
                logger.info(f"Utilisateur {current_user} a accédé à {len(result)} de ses commandes")

            return result

        except Exception as e:
            logger.error(f"Error in get_orders_by_amount_safe: {e}")
            return {"error": str(e)}

    def get_customer_emails(self, limit=50, domain_filter=None, anonymize=False):
        """
        Récupère les emails clients, avec option de filtrage et d'anonymisation.

        Args:
            limit (int): Nombre maximum d'emails à retourner
            domain_filter (str, optional): Filtrer par domaine spécifique
            anonymize (bool): Si True, masque partiellement les emails

        Returns:
            dict: Résultats formatés avec résumé et données
        """
        try:
            print(
                f"Exécution de get_customer_emails avec limit={limit}, domain_filter={domain_filter}, anonymize={anonymize}")

            # Construction de la requête de base
            query = """
            SELECT 
                id,
                email,
                first_name,
                last_name
            FROM customers
            WHERE 1=1
            """

            # Paramètres sous forme de dictionnaire
            params = {}

            # Ajout du filtre par domaine si nécessaire
            if domain_filter:
                query += " AND email LIKE :domain"
                params["domain"] = f"%{domain_filter}%"

            # Ajout de la limite
            query += " LIMIT :limit"
            params["limit"] = limit

            # Exécution de la requête
            print(f"Exécution de la requête SQL: {query} avec params={params}")
            results = self.execute_sql_query(query, params)
            print(f"Résultats bruts: {results[:2] if results else 'Aucun résultat'}")

            # Vérifier si on a des résultats
            if not results or isinstance(results, dict) and 'error' in results:
                error_msg = results.get('error', "Aucun résultat trouvé") if isinstance(results,
                                                                                        dict) else "Aucun résultat trouvé"
                print(f"Erreur ou aucun résultat: {error_msg}")
                return {
                    "summary": f"Erreur lors de la recherche d'emails: {error_msg}",
                    "emails": []
                }

            # Anonymisation si demandée
            if anonymize and results:
                for item in results:
                    email = item['email']
                    at_pos = email.find('@')
                    if at_pos > 0:
                        username = email[:at_pos]
                        domain = email[at_pos:]
                        if len(username) > 2:
                            item['email'] = username[:2] + '***' + domain

            # Formatage explicite du résultat
            formatted_result = {
                "summary": f"Total de {len(results)} adresses email trouvées" +
                           (f" avec filtre de domaine '{domain_filter}'" if domain_filter else "") +
                           (". Les emails sont partiellement masqués." if anonymize else "."),
                "emails": results
            }

            print(f"Résultat formaté: {formatted_result['summary']}")
            return formatted_result

        except Exception as e:
            error_msg = f"Error in get_customer_emails: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {
                "summary": f"Erreur lors de la recherche d'emails: {str(e)}",
                "emails": []
            }

    def analyze_email_domains(self, limit=20):
        """
        Analyse les domaines d'emails clients.

        Args:
            limit (int): Nombre maximum de résultats

        Returns:
            dict: Statistiques sur les domaines d'emails
        """
        try:
            # Requête pour compter les domaines
            query = """
            SELECT 
                SUBSTR(email, INSTR(email, '@') + 1) AS domain,
                COUNT(*) AS count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
            FROM customers
            GROUP BY domain
            ORDER BY count DESC
            LIMIT :limit
            """

            # Paramètres
            params = {"limit": limit}

            # Exécution de la requête
            domain_stats = self.execute_sql_query(query, params)

            # Récupération des statistiques générales
            general_stats = {}

            # Requête pour le total des clients
            total_query = "SELECT COUNT(*) as total FROM customers"
            total_result = self.execute_sql_query(total_query)
            if total_result and len(total_result) > 0:
                general_stats["total_customers"] = total_result[0]["total"]

            # Requête pour des exemples anonymisés
            samples_query = """
            SELECT email FROM customers
            ORDER BY RANDOM()
            LIMIT 5
            """
            samples = self.execute_sql_query(samples_query)

            # Anonymisation des exemples
            anonymized_samples = []
            if samples:
                for item in samples:
                    email = item["email"]
                    at_pos = email.find('@')
                    if at_pos > 0:
                        username = email[:at_pos]
                        domain = email[at_pos:]
                        anonymized_samples.append(username[:2] + "***" + domain)

            general_stats["anonymized_samples"] = anonymized_samples

            return {
                "domain_statistics": domain_stats,
                "general_statistics": general_stats
            }

        except Exception as e:
            logger.error(f"Error in analyze_email_domains: {e}")
            return {"error": str(e)}

    def get_postal_code_distribution(self, country=None, limit=20):
        """
        Analyse la distribution des codes postaux.

        Args:
            country (str, optional): Filtrer par pays
            limit (int): Nombre maximum de résultats

        Returns:
            list: Distribution par code postal
        """
        try:
            # Construction de la requête
            query = """
            SELECT 
                SUBSTR(postal_code, 1, 2) AS region_code,
                COUNT(*) AS customer_count,
                GROUP_CONCAT(DISTINCT city, ', ') AS cities
            FROM customer_addresses
            WHERE 1=1
            """

            # Paramètres
            params = {}

            # Ajout du filtre par pays si nécessaire
            if country:
                query += " AND country LIKE :country"
                params["country"] = f"%{country}%"

            # Ajout du groupement et de la limite
            query += """
            GROUP BY region_code
            ORDER BY customer_count DESC
            LIMIT :limit
            """
            params["limit"] = limit

            return self.execute_sql_query(query, params)

        except Exception as e:
            logger.error(f"Error in get_postal_code_distribution: {e}")
            return {"error": str(e)}

    def get_sales_statistics(self, period=None, group_by='month', category=None, limit=50):
        """
        Récupère des statistiques de ventes avec diverses options.

        Args:
            period (str, optional): Période (ex: "2023" pour l'année, "2023-01" pour un mois)
            group_by (str): Grouper par 'day', 'month', 'year', 'category'
            category (str, optional): Filtrer par catégorie
            limit (int): Nombre maximum de résultats

        Returns:
            list: Statistiques de ventes
        """
        try:
            # Base de la requête avec CTE (Common Table Expression)
            base_query = """
            WITH OrderStats AS (
                SELECT 
                    o.id AS order_id,
                    o.order_date,
                    o.total_amount,
                    p.name AS product_name,
                    p.price,
                    oi.quantity,
                    p.price * oi.quantity AS item_total,
                    c.name AS category_name
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                JOIN products p ON oi.product_id = p.id
                JOIN categories c ON p.category_id = c.id
                WHERE 1=1
            """

            # Paramètres
            params = {}

            # Ajout des filtres
            if period:
                if len(period) == 4:  # Année
                    base_query += " AND strftime('%Y', o.order_date) = :period_year"
                    params["period_year"] = period
                elif len(period) == 7:  # Année-mois
                    base_query += " AND strftime('%Y-%m', o.order_date) = :period_month"
                    params["period_month"] = period

            if category:
                base_query += " AND c.name LIKE :category"
                params["category"] = f"%{category}%"

            base_query += ")"

            # Différentes requêtes selon le type de groupement
            if group_by == 'day':
                query = base_query + """
                SELECT 
                    strftime('%Y-%m-%d', order_date) AS day,
                    COUNT(DISTINCT order_id) AS order_count,
                    SUM(item_total) AS daily_revenue,
                    ROUND(AVG(total_amount), 2) AS avg_order_value,
                    COUNT(DISTINCT product_name) AS unique_products_sold
                FROM OrderStats
                GROUP BY day
                ORDER BY day DESC
                """

            elif group_by == 'month':
                query = base_query + """
                SELECT 
                    strftime('%Y-%m', order_date) AS month,
                    COUNT(DISTINCT order_id) AS order_count,
                    SUM(item_total) AS monthly_revenue,
                    ROUND(AVG(total_amount), 2) AS avg_order_value,
                    COUNT(DISTINCT product_name) AS unique_products_sold
                FROM OrderStats
                GROUP BY month
                ORDER BY month DESC
                """

            elif group_by == 'year':
                query = base_query + """
                SELECT 
                    strftime('%Y', order_date) AS year,
                    COUNT(DISTINCT order_id) AS order_count,
                    SUM(item_total) AS yearly_revenue,
                    ROUND(AVG(total_amount), 2) AS avg_order_value,
                    COUNT(DISTINCT product_name) AS unique_products_sold
                FROM OrderStats
                GROUP BY year
                ORDER BY year DESC
                """

            elif group_by == 'category':
                query = base_query + """
                SELECT 
                    category_name,
                    COUNT(DISTINCT order_id) AS order_count,
                    SUM(item_total) AS category_revenue,
                    ROUND(AVG(price), 2) AS avg_product_price,
                    SUM(quantity) AS units_sold
                FROM OrderStats
                GROUP BY category_name
                ORDER BY category_revenue DESC
                """

            else:
                return {"error": f"Type de groupement non reconnu: {group_by}"}

            # Ajout de la limite
            query += " LIMIT :limit"
            params["limit"] = limit

            return self.execute_sql_query(query, params)

        except Exception as e:
            logger.error(f"Error in get_sales_statistics: {e}")
            return {"error": str(e)}

    def get_customer_emails_safe(self, current_user, anonymize=False):
        """
        Version sécurisée qui récupère uniquement l'email de l'utilisateur connecté.

        Args:
            current_user (str): Nom complet de l'utilisateur connecté (format: "Prénom Nom")
            anonymize (bool): Si True, masque partiellement l'email

        Returns:
            dict: Résultat formaté avec l'email de l'utilisateur
        """
        try:
            print(f"Exécution de get_customer_emails_safe pour utilisateur: {current_user}, anonymize={anonymize}")

            # Vérification que l'utilisateur est spécifié
            if not current_user:
                logger.error("Tentative d'accès aux emails sans identifiant utilisateur")
                return {
                    "summary": "Authentification requise pour accéder aux emails",
                    "emails": []
                }

            # Extraction du prénom et du nom
            try:
                first_name, last_name = current_user.strip().split(' ', 1)
            except ValueError:
                logger.error(f"Format de nom d'utilisateur invalide: {current_user}")
                return {
                    "summary": "Format de nom d'utilisateur invalide",
                    "emails": []
                }

            # Construction de la requête sécurisée - filtre pour retourner uniquement l'utilisateur connecté
            query = """
            SELECT 
                id,
                email,
                first_name,
                last_name
            FROM customers
            WHERE first_name LIKE :first_name 
            AND last_name LIKE :last_name
            LIMIT 1
            """

            # Paramètres sous forme de dictionnaire avec les informations de l'utilisateur
            params = {
                "first_name": first_name,
                "last_name": last_name
            }

            # Exécution de la requête
            print(f"Exécution de la requête SQL sécurisée: {query} avec params={params}")
            results = self.execute_sql_query(query, params)
            print(f"Résultats bruts: {results}")

            # Vérifier si on a des résultats
            if not results or isinstance(results, dict) and 'error' in results:
                error_msg = results.get('error', "Aucun résultat trouvé") if isinstance(results,
                                                                                        dict) else "Aucun résultat trouvé"
                print(f"Erreur ou aucun résultat: {error_msg}")
                return {
                    "summary": f"Aucune information trouvée pour l'utilisateur {current_user}",
                    "emails": []
                }

            # Anonymisation si demandée
            if anonymize and results:
                for item in results:
                    email = item['email']
                    at_pos = email.find('@')
                    if at_pos > 0:
                        username = email[:at_pos]
                        domain = email[at_pos:]
                        if len(username) > 2:
                            item['email'] = username[:2] + '***' + domain

            # Formatage explicite du résultat
            formatted_result = {
                "summary": f"Informations email pour {current_user}" +
                           (". L'email est partiellement masqué." if anonymize else "."),
                "emails": results
            }

            # Log de sécurité
            logger.info(f"Utilisateur {current_user} a accédé à son propre email")
            print(f"Résultat formaté: {formatted_result['summary']}")

            return formatted_result

        except Exception as e:
            error_msg = f"Error in get_customer_emails_safe: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {
                "summary": f"Erreur lors de la recherche d'emails: {str(e)}",
                "emails": []
            }

    def execute_sql_query_safe(self, query_str, params=None, limit=20, current_user=None):
        """
        Version sécurisée pour exécuter une requête SQL avec restrictions.

        Args:
            query_str (str): La requête SQL à exécuter
            params (dict, optional): Paramètres pour la requête SQL
            limit (int): Nombre maximum de résultats (plafonné à 20 en mode safe)
            current_user (str): Identifiant de l'utilisateur actuel pour les restrictions d'accès

        Returns:
            dict: Résultats filtrés et message de sécurité
        """
        try:
            # Liste de mots-clés dangereux
            dangerous_keywords = ['DELETE', 'DROP', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE', 'GRANT',
                                  'REVOKE']

            # Vérification des mots-clés dangereux
            for keyword in dangerous_keywords:
                if keyword in query_str.upper():
                    return {
                        "error": f"Requête non autorisée en mode sécurisé: contient '{keyword}'",
                        "security_message": "Les requêtes de modification sont désactivées en mode sécurisé."
                    }

            # Limiter le nombre de résultats
            # On ne prend jamais plus de 20 résultats en mode safe
            safe_limit = min(limit, 20)

            # Exécuter la requête avec précaution
            results = self.execute_sql_query(query_str, params, safe_limit)

            # Si on a une erreur, on la retourne
            if isinstance(results, dict) and 'error' in results:
                return results

            # Anonymiser les résultats
            sanitized_results = []
            for item in results:
                sanitized_item = {}
                for key, value in item.items():
                    # Anonymisation des données sensibles
                    if 'email' in key.lower() and isinstance(value, str):
                        if '@' in value:
                            username, domain = value.split('@', 1)
                            if len(username) > 2:
                                value = username[:2] + '***@' + domain

                    elif 'address' in key.lower() or 'street' in key.lower():
                        if isinstance(value, str) and len(value) > 5:
                            parts = value.split(' ', 1)
                            if len(parts) > 1:
                                value = parts[0] + ' ***'

                    sanitized_item[key] = value

                sanitized_results.append(sanitized_item)

            return {
                "data": sanitized_results,
                "security_message": "Résultats filtrés et anonymisés en mode sécurisé",
                "count": len(sanitized_results)
            }

        except Exception as e:
            logger.error(f"Error in execute_sql_query_safe: {e}")
            return {"error": f"Erreur lors de l'exécution de la requête sécurisée: {str(e)}"}

    def get_customer_addresses_safe(self, current_user, city=None, country=None, postal_code=None, limit=50,
                                    anonymize=False):
        """
        Version sécurisée qui récupère uniquement les adresses de l'utilisateur connecté.

        Args:
            current_user (str): Nom complet de l'utilisateur connecté (format: "Prénom Nom")
            city (str, optional): Filtrer par ville
            country (str, optional): Filtrer par pays
            postal_code (str, optional): Filtrer par code postal
            limit (int): Nombre maximum de résultats
            anonymize (bool): Si True, masque partiellement les adresses

        Returns:
            list: Liste des adresses de l'utilisateur connecté
        """
        try:
            # Vérification que l'utilisateur est spécifié
            if not current_user:
                logger.error("Tentative d'accès aux adresses sans identifiant utilisateur")
                return {"error": "Authentification requise pour accéder aux adresses"}

            # Extraction du prénom et du nom
            try:
                first_name, last_name = current_user.strip().split(' ', 1)
            except ValueError:
                logger.error(f"Format de nom d'utilisateur invalide: {current_user}")
                return {"error": "Format de nom d'utilisateur invalide"}

            # Construction de la requête sécurisée
            query = """
            SELECT 
                ca.id,
                ca.customer_id,
                ca.address_type,
                ca.street_line1,
                ca.street_line2,
                ca.postal_code,
                ca.city,
                ca.country,
                c.first_name,
                c.last_name,
                c.email
            FROM customer_addresses ca
            JOIN customers c ON ca.customer_id = c.id
            WHERE c.first_name LIKE :first_name 
            AND c.last_name LIKE :last_name
            """

            # Paramètres sous forme de dictionnaire avec les informations de l'utilisateur
            params = {
                "first_name": first_name,
                "last_name": last_name
            }

            # Ajout des filtres additionnels si nécessaires
            if city:
                query += " AND ca.city LIKE :city"
                params["city"] = f"%{city}%"

            if country:
                query += " AND ca.country LIKE :country"
                params["country"] = f"%{country}%"

            if postal_code:
                query += " AND ca.postal_code LIKE :postal_code"
                params["postal_code"] = f"%{postal_code}%"

            # Ajout de la limite
            if limit > 0:
                query += " LIMIT :limit"
                params["limit"] = limit

            # Exécution de la requête
            results = self.execute_sql_query(query, params)

            # Anonymisation si demandée
            if anonymize and results:
                for item in results:
                    if 'street_line1' in item and item['street_line1']:
                        parts = item['street_line1'].split(' ', 1)
                        if len(parts) > 1:
                            item['street_line1'] = parts[0] + ' ' + '***'

                    if 'email' in item and item['email']:
                        email = item['email']
                        at_pos = email.find('@')
                        if at_pos > 0:
                            username = email[:at_pos]
                            domain = email[at_pos:]
                            if len(username) > 2:
                                item['email'] = username[:2] + '***' + domain

            # Log de sécurité
            if results:
                logger.info(f"Utilisateur {current_user} a accédé à {len(results)} de ses adresses")
            else:
                logger.info(f"Utilisateur {current_user} n'a pas d'adresses ou filtre trop restrictif")

            return results

        except Exception as e:
            logger.error(f"Error in get_customer_addresses_safe: {e}")
            return {"error": str(e)}