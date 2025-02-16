import networkx as nx
import json
import os
import re

class KnowledgeGraph:
    def __init__(self, file_path="knowledge.json"):
        self.graph = nx.Graph()
        self.file_path = file_path
        self.load_facts()

    def extract_keywords(self, text):
        """Extracts important words from a sentence."""
        words = text.lower().split()
        return [word for word in words if len(word) > 3]  # Ignore short words

    def add_fact(self, fact):
        """Adds a fact and links it to relevant keywords."""
        fact_id = len(self.graph.nodes)
        self.graph.add_node(fact_id, fact=fact)
        
        # Link to keywords
        keywords = self.extract_keywords(fact)
        for keyword in keywords:
            if keyword not in self.graph:
                self.graph.add_node(keyword)
            self.graph.add_edge(keyword, fact_id)

        self.save_facts()

    def query_fact(self, query):
        """Finds a relevant fact based on the keywords in a query."""
        keywords = self.extract_keywords(query)
        best_fact = None
        best_score = 0

        for keyword in keywords:
            if keyword in self.graph:
                linked_facts = list(self.graph.neighbors(keyword))
                for fact_id in linked_facts:
                    if isinstance(fact_id, int):  # Ensure it's a fact, not a keyword
                        fact = self.graph.nodes[fact_id]['fact']
                        score = sum(1 for word in keywords if word in fact)
                        if score > best_score:
                            best_score = score
                            best_fact = fact
        
        return best_fact

    def save_facts(self):
        """Save the knowledge graph to a JSON file."""
        data = {n: self.graph.nodes[n]['fact'] for n in self.graph.nodes if isinstance(n, int)}
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_facts(self):
        """Load the knowledge graph from a JSON file if it exists."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                data = json.load(f)
                for fact_id, fact in data.items():
                    self.graph.add_node(int(fact_id), fact=fact)
