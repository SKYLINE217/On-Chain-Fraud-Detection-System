# scripts/validate_scores.py
"""
Post-batch validation: confirm all nodes have risk_score set.
Run after every batch job.

See person_a_stages.md §4.3 for full reference.

Usage:
    python scripts/validate_scores.py

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "changeme_in_prod"),
    )
)

with driver.session() as session:
    result = session.run("""
        MATCH (t:Transaction)
        RETURN
          count(t) AS total,
          count(t.risk_score) AS scored,
          count(CASE WHEN t.risk_score IS NULL THEN 1 END) AS unscored
    """)
    r = result.single()
    total, scored, unscored = r["total"], r["scored"], r["unscored"]

assert scored == 203769, f"Not all nodes scored: scored={scored}, total={total}"
assert unscored == 0, f"Unscored nodes: {unscored}"
print(f"✅ Validation passed: {scored}/{total} nodes scored, {unscored} unscored.")

driver.close()
