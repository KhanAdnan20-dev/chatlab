"""
Dynamic AI/ML Roadmap Generator & Curriculum DAG Engine.

Implements graph traversal, ancestor prerequisite resolution, topological sorting (Kahn's algorithm),
and deterministic Mermaid.js flowchart generation.
Produces interactive, syntactically sanitized Mermaid diagrams and downloadable Markdown roadmaps.
"""
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from backend.app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CurriculumNode:
    id: str
    label: str
    domain: str
    phase: str
    prerequisites: List[str] = field(default_factory=list)
    difficulty: str = "Intermediate"
    description: str = ""
    key_topics: List[str] = field(default_factory=list)
    project: str = ""
    paper: str = ""


@dataclass
class RoadmapOutput:
    domain: str
    title: str
    mermaid_code: str
    phases: List[str]
    nodes: List[CurriculumNode]
    markdown_content: str
    node_count: int
    edge_count: int


class RoadmapEngine:
    """
    Traverses the Master Curriculum DAG and generates validated Mermaid flowcharts and downloadable roadmaps.
    """

    DOMAIN_ALIASES = {
        "ai_soft_computing": "ai_soft_computing",
        "soft computing": "ai_soft_computing",
        "hard computing": "ai_soft_computing",
        "ai and soft computing": "ai_soft_computing",
        "fuzzy logic": "ai_soft_computing",
        "fuzzy": "ai_soft_computing",
        "mamdani": "ai_soft_computing",
        "sugeno": "ai_soft_computing",
        "defuzzification": "ai_soft_computing",
        "centroid": "ai_soft_computing",
        "zadeh": "ai_soft_computing",
        "anfis": "ai_soft_computing",
        "neuro fuzzy": "ai_soft_computing",
        "neuro-fuzzy": "ai_soft_computing",
        "perceptron": "ai_soft_computing",
        "kohonen": "ai_soft_computing",
        "self organizing map": "ai_soft_computing",
        "som": "ai_soft_computing",
        "state space": "ai_soft_computing",
        "heuristic": "ai_soft_computing",
        "admissible": "ai_soft_computing",
        "a* search": "ai_soft_computing",
        "bfs": "ai_soft_computing",
        "dfs": "ai_soft_computing",
        "first order logic": "ai_soft_computing",
        "first-order logic": "ai_soft_computing",
        "fol": "ai_soft_computing",
        "unification": "ai_soft_computing",
        "resolution": "ai_soft_computing",
        "strips": "ai_soft_computing",
        "genetic algorithm": "ai_soft_computing",
        "genetic algorithms": "ai_soft_computing",
        "evolutionary": "ai_soft_computing",
        "chromosome": "ai_soft_computing",
        "crossover": "ai_soft_computing",
        "mutation": "ai_soft_computing",
        "midsem": "ai_soft_computing",
        "syllabus": "ai_soft_computing",
        "artificial intelligence": "ai_soft_computing",
        "nlp": "nlp_llms",
        "llm": "nlp_llms",
        "llms": "nlp_llms",
        "language": "nlp_llms",
        "transformers": "nlp_llms",
        "vision": "computer_vision",
        "cv": "computer_vision",
        "image": "computer_vision",
        "cnn": "computer_vision",
        "deep learning": "deep_learning",
        "dl": "deep_learning",
        "neural": "deep_learning",
        "machine learning": "classical_ml",
        "ml": "classical_ml",
        "classical": "classical_ml",
        "math": "foundations",
        "foundations": "foundations",
        "basics": "foundations",
        "mlops": "mlops",
        "deployment": "mlops",
    }

    def __init__(self, dag_path: Optional[Path] = None) -> None:
        self.dag_path = dag_path or (settings.BASE_DIR / "data" / "curriculum_dag.json")
        self.nodes_map: Dict[str, CurriculumNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self._load_dag()

    def _load_dag(self) -> None:
        """Loads and indexes the curriculum graph."""
        if not self.dag_path.exists():
            logger.warning("[RoadmapEngine] DAG file missing at %s", self.dag_path)
            return

        try:
            with open(self.dag_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.nodes_map = {
                entry["id"]: CurriculumNode(
                    id=entry["id"],
                    label=entry["label"],
                    domain=entry["domain"],
                    phase=entry["phase"],
                    prerequisites=entry.get("prerequisites", []),
                    difficulty=entry.get("difficulty", "Intermediate"),
                    description=entry.get("description", ""),
                    key_topics=entry.get("key_topics", []),
                    project=entry.get("project", ""),
                    paper=entry.get("paper", ""),
                )
                for entry in data.get("nodes", [])
            }

            self.edges = [(u, v) for u, v in data.get("edges", [])]
            logger.info("[RoadmapEngine] Loaded %d curriculum nodes and %d dependency edges.", len(self.nodes_map), len(self.edges))
        except Exception as exc:
            logger.error("[RoadmapEngine] Failed to load curriculum DAG: %s", exc)

    def detect_domain(self, query: str) -> str:
        """Extracts the intended curriculum domain from user query text."""
        q_lower = query.lower()
        if "a*" in q_lower or re.search(r"\b(?:co[1-6]|unit\s*(?:[1-6]|i|ii|iii|iv|v|vi))\b", q_lower):
            return "ai_soft_computing"

        # Check domain aliases (longer match first e.g. 'soft computing' before 'computing')
        for alias in sorted(self.DOMAIN_ALIASES.keys(), key=len, reverse=True):
            domain = self.DOMAIN_ALIASES[alias]
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                return domain

        # Default to full core roadmap if general query
        return "all"

    def get_induced_subgraph(self, target_domain: str = "all") -> Tuple[List[CurriculumNode], List[Tuple[str, str]]]:
        """
        Extracts relevant nodes and edges including all prerequisite ancestry.
        Ensures topological validity using Kahn's algorithm.
        """
        if target_domain == "all":
            selected_node_ids = set(self.nodes_map.keys())
        else:
            # 1. Start with domain nodes
            seed_nodes = {nid for nid, node in self.nodes_map.items() if node.domain == target_domain}
            if not seed_nodes:
                selected_node_ids = set(self.nodes_map.keys())
            else:
                # 2. Traverse backward to collect all prerequisite ancestors
                selected_node_ids = set(seed_nodes)
                queue = deque(seed_nodes)
                while queue:
                    curr_id = queue.popleft()
                    curr_node = self.nodes_map.get(curr_id)
                    if curr_node:
                        for prereq_id in curr_node.prerequisites:
                            if prereq_id in self.nodes_map and prereq_id not in selected_node_ids:
                                selected_node_ids.add(prereq_id)
                                queue.append(prereq_id)

        # 3. Filter edges
        induced_edges = [
            (u, v) for u, v in self.edges
            if u in selected_node_ids and v in selected_node_ids
        ]

        # 4. Topological Sort (Kahn's Algorithm)
        in_degree = {nid: 0 for nid in selected_node_ids}
        adj = defaultdict(list)
        for u, v in induced_edges:
            adj[u].append(v)
            in_degree[v] += 1

        zero_in = deque([nid for nid in selected_node_ids if in_degree[nid] == 0])
        sorted_ids: List[str] = []

        while zero_in:
            u = zero_in.popleft()
            sorted_ids.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    zero_in.append(v)

        # If any nodes remain unvisited, a dependency cycle was encountered;
        # append remaining nodes deterministically and strip cycle-causing back-edges
        if len(sorted_ids) < len(selected_node_ids):
            logger.warning("[RoadmapEngine] Cycle detected in subgraph; breaking back-edges to preserve DAG.")
            for nid in sorted(selected_node_ids):
                if nid not in sorted_ids:
                    sorted_ids.append(nid)
            order_Rank = {nid: idx for idx, nid in enumerate(sorted_ids)}
            induced_edges = [(u, v) for u, v in induced_edges if order_Rank[u] < order_Rank[v]]

        sorted_nodes = [self.nodes_map[nid] for nid in sorted_ids if nid in self.nodes_map]
        return sorted_nodes, induced_edges

    def generate_mermaid_code(self, nodes: List[CurriculumNode], edges: List[Tuple[str, str]]) -> str:
        """
        Generates clean, syntactically safe Mermaid.js flowchart syntax.
        Ensures quotes on all labels to avoid Mermaid bracket/parenthesis parser crashes.
        """
        lines = [
            "%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#4f46e5', 'edgeLabelBackground':'#1e293b', 'tertiaryColor': '#0f172a'}}}%%",
            "graph TD",
        ]

        # Group nodes by phase
        phase_groups = defaultdict(list)
        for node in nodes:
            phase_groups[node.phase].append(node)

        # Render subgraphs for each phase
        phase_idx = 1
        for phase_name, p_nodes in phase_groups.items():
            safe_phase_id = f"Phase_{phase_idx}"
            clean_phase_label = phase_name.replace('"', "'")
            lines.append(f'  subgraph {safe_phase_id}["{clean_phase_label}"]')
            for n in p_nodes:
                clean_label = n.label.replace('"', "'")
                diff_tag = f"[{n.difficulty}]"
                lines.append(f'    {n.id}["{clean_label} <br/> <small>{diff_tag}</small>"]')
            lines.append("  end")
            phase_idx += 1

        # Render edges
        for u, v in edges:
            lines.append(f"  {u} --> {v}")

        return "\n".join(lines)

    @staticmethod
    def sanitize_mermaid_code(code: str) -> str:
        """
        Sanitizes raw AI-generated Mermaid code.
        Ensures all node and subgraph bracket labels are strictly double-quoted
        to prevent syntax crashes in the Mermaid.js parser.
        """
        if not code or not code.strip():
            return "graph TD\n  start[\"Start Learning\"]\n"

        clean = code.strip()
        # Remove markdown fences if present
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:mermaid)?\s*", "", clean)
        if clean.endswith("```"):
            clean = re.sub(r"\s*```$", "", clean)
        clean = clean.strip()

        # Ensure graph TD declaration
        if not re.search(r"^\s*(graph|flowchart)\s+", clean, re.MULTILINE):
            clean = "graph TD\n" + clean

        lines = clean.split("\n")
        sanitized_lines = []
        for line in lines:
            trimmed = line.rstrip()
            # Replace unquoted subgraph labels: subgraph sub_1[Phase 1] -> subgraph sub_1["Phase 1"]
            subgraph_match = re.match(r'^(\s*subgraph\s+[\w\-]+)\[([^"\]]+)\]', trimmed)
            if subgraph_match:
                trimmed = f'{subgraph_match.group(1)}["{subgraph_match.group(2).strip()}"]'

            # Replace unquoted node labels: id[Label Here] -> id["Label Here"]
            node_match = re.search(r'([\w\-]+)\[([^"\]\n]+)\]', trimmed)
            if node_match and not trimmed.strip().startswith("subgraph"):
                inner_text = node_match.group(2).replace('"', "'").strip()
                trimmed = trimmed[:node_match.start()] + f'{node_match.group(1)}["{inner_text}"]' + trimmed[node_match.end():]

            sanitized_lines.append(trimmed)

        res = "\n".join(sanitized_lines)
        if "classDef" not in res:
            res += "\n  classDef default fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;"
            res += "\n  classDef beginner fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5;"
            res += "\n  classDef advanced fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#faf5ff;"
        return res

    def generate_markdown_roadmap(
        self,
        domain_name: str,
        mermaid_code: str,
        nodes: List[CurriculumNode],
    ) -> str:
        """Builds a rich, downloadable Markdown document with study checklist and project guides."""
        title = f"ChatLab AI/ML Roadmap: {domain_name.replace('_', ' ').title()}"

        mermaid_code = mermaid_code or ""
        md_lines = [
            f"# {title}",
            "",
            "> Generated dynamically by ChatLab Grounded Curriculum & Graph Engine.",
            "",
            "## 🗺️ Interactive Learning Flowchart",
            "",
            "```mermaid",
            mermaid_code,
            "```",
            "",
            "---",
            "",
            "## 📚 Curriculum Breakdown & Milestones",
            "",
        ]

        # Group by phase
        phase_groups = defaultdict(list)
        for node in nodes:
            phase_groups[node.phase].append(node)

        for phase, p_nodes in phase_groups.items():
            md_lines.append(f"### {phase}")
            for n in p_nodes:
                topics_str = ", ".join(n.key_topics) if n.key_topics else "Core fundamentals"
                md_lines.append(f"- [ ] **{n.label}** `({n.difficulty})`")
                md_lines.append(f"  - **Overview**: {n.description}")
                md_lines.append(f"  - **Key Concepts**: {topics_str}")
                if n.project:
                    md_lines.append(f"  - **Hands-on Project**: 🛠️ {n.project}")
                if n.paper:
                    md_lines.append(f"  - **Seminal Literature**: 📄 *{n.paper}*")
                md_lines.append("")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("*Pro-tip: You can ask ChatLab questions about any node in this roadmap to practice with the Socratic Tutor!*")

        return "\n".join(md_lines)

    def generate_custom_topic_roadmap(self, query: str) -> RoadmapOutput:
        """Dynamically synthesizes a 4-phase structured syllabus & flowchart for any novel subject query."""
        stripped_query = re.sub(
            r"^\s*(?:can\s+you\s+|please\s+|could\s+you\s+)?(?:give\s+me\s+|generate\s+|create\s+|build\s+|show\s+me\s+|make\s+)?"
            r"(?:a\s+|an\s+|the\s+)?(?:complete\s+|detailed\s+|comprehensive\s+|interactive\s+)?"
            r"(?:learning\s+roadmap|roadmap|curriculum|learning\s+path|study\s+plan)\s*(?:for|on|about|to\s+learn|in)?\s*",
            "",
            query or "",
            flags=re.IGNORECASE,
        )
        clean_topic = re.sub(r"[^a-zA-Z0-9\s\-]", "", stripped_query).strip().title()
        if not clean_topic:
            clean_topic = re.sub(r"[^a-zA-Z0-9\s\-]", "", query or "").strip().title() or "Custom Subject"

        phases = [
            f"Phase 1: Foundations of {clean_topic}",
            f"Phase 2: Core Architecture & Algorithms",
            f"Phase 3: Advanced Theory & Optimization",
            f"Phase 4: Capstone Applications & Research",
        ]

        nodes = [
            CurriculumNode(
                id="topic_foundations",
                label=f"Mathematical & Conceptual Foundations",
                domain="custom",
                phase=phases[0],
                difficulty="Beginner",
                description=f"Core mathematical pre-requisites and fundamental terminology for {clean_topic}.",
                key_topics=[f"{clean_topic} Basics", "Formal Notation", "Core Principles"],
                project=f"Build an exploratory primer notebook on {clean_topic}.",
                paper="Seminal survey and foundations monograph",
            ),
            CurriculumNode(
                id="topic_core_algos",
                label=f"Core Algorithms & Mechanisms",
                domain="custom",
                phase=phases[1],
                difficulty="Intermediate",
                description=f"Primary algorithmic primitives, execution flows, and implementation details.",
                key_topics=["Algorithm Workflow", "State Transitions", "Complexity Bounds"],
                project=f"Implement baseline algorithmic pipeline from scratch in Python.",
                paper="Foundational algorithmic design paper",
            ),
            CurriculumNode(
                id="topic_advanced_methods",
                label=f"Advanced Optimization & Scaling",
                domain="custom",
                phase=phases[2],
                difficulty="Advanced",
                description=f"High-performance heuristics, error mitigation, and hybrid variants.",
                key_topics=["Optimization Strategies", "Error Analysis", "Benchmarking"],
                project=f"Benchmark and optimize algorithmic throughput against standard baselines.",
                paper="Advanced performance and theoretical convergence analysis",
            ),
            CurriculumNode(
                id="topic_capstone",
                label=f"Real-World Production Capstone",
                domain="custom",
                phase=phases[3],
                difficulty="Advanced",
                description=f"End-to-end applied synthesis, industrial case studies, and research frontiers.",
                key_topics=["Deployment Architecture", "Integration Testing", "Future Directions"],
                project=f"Full-stack end-to-end capstone system demonstrating {clean_topic}.",
                paper="State-of-the-art applied survey and real-world deployment review",
            ),
        ]

        edges = [
            ("topic_foundations", "topic_core_algos"),
            ("topic_core_algos", "topic_advanced_methods"),
            ("topic_advanced_methods", "topic_capstone"),
        ]

        mermaid_code = self.generate_mermaid_code(nodes, edges)
        markdown_doc = self.generate_markdown_roadmap(clean_topic, mermaid_code, nodes)

        return RoadmapOutput(
            domain="custom",
            title=f"{clean_topic} Curriculum & Roadmap",
            mermaid_code=mermaid_code,
            phases=phases,
            nodes=nodes,
            markdown_content=markdown_doc,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def generate_roadmap(self, query: str = "") -> RoadmapOutput:
        """
        End-to-end roadmap generator: detects domain, extracts subgraph,
        builds Mermaid code, and produces complete Markdown output.
        """
        domain = self.detect_domain(query)
        if domain == "all" and query.strip() and not any(k in query.lower() for k in ["all", "everything", "full", "complete"]):
            return self.generate_custom_topic_roadmap(query)

        nodes, edges = self.get_induced_subgraph(domain)
        mermaid_code = self.generate_mermaid_code(nodes, edges)
        markdown_doc = self.generate_markdown_roadmap(domain, mermaid_code, nodes)
        phases = list(dict.fromkeys(n.phase for n in nodes))

        if domain == "all":
            domain_display = "Comprehensive AI/ML"
        elif domain == "ai_soft_computing":
            domain_display = "AI & Soft Computing (Syllabus Units I-VI)"
        else:
            domain_display = domain.replace("_", " ").title()

        return RoadmapOutput(
            domain=domain,
            title=f"{domain_display} Roadmap",
            mermaid_code=mermaid_code,
            phases=phases,
            nodes=nodes,
            markdown_content=markdown_doc,
            node_count=len(nodes),
            edge_count=len(edges),
        )


# Singleton instance
roadmap_engine = RoadmapEngine()
