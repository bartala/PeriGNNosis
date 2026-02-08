import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import HeteroData

from neo4j import GraphDatabase

# LangChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore")


DATA_DIR = "./data"
NARRATIVE_FILE = "embedded_CBPTSD.csv"
PDI_FILE = "CBEx_pdi.csv"

# Neo4j defaults
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

LLAMA_MODEL = "llama3.1:8b"
MIN_WORDS = 30
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


df = pd.read_csv(os.path.join(DATA_DIR, NARRATIVE_FILE))
df = df[df["source"] == "CBEx"]

df_pdi = pd.read_csv(os.path.join(DATA_DIR, PDI_FILE))
df = pd.merge(df, df_pdi, on="record_id", how="left")

df["word_count"] = df["narrative"].str.split().str.len()
df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)


llm = ChatOllama(
    model=LLAMA_MODEL,
    temperature=0.0,
    num_ctx=8192
)

graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
    ],
    strict=False
)


graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# Clean database (safe for Desktop)
graph.query("MATCH (n) DETACH DELETE n")

# Index for faster merges
graph.query("""
CREATE INDEX woman_record_id IF NOT EXISTS
FOR (w:Woman) ON (w.record_id)
""")


documents = [
    Document(
        page_content=row["narrative"],
        metadata={"record_id": row["record_id"]}
    )
    for _, row in df.iterrows()
]


graph_docs = []
for doc in tqdm(documents):
    graph_docs.extend(
        graph_transformer.convert_to_graph_documents([doc])
    )

graph.add_graph_documents(graph_docs)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

PDI_COLS = [c for c in df.columns if c.startswith("pdi_")]

with driver.session(database=NEO4J_DATABASE) as session:
    for _, row in df.iterrows():
        props = {
            "record_id": row["record_id"],
            "label": int(row["CB_PTSD"]),
            "obstetric_complication": int(row["obstetric_complication"]),
        }
        for c in PDI_COLS:
            props[c] = float(row[c])

        session.run(
            """
            MERGE (w:Woman {record_id: $record_id})
            SET w += $props
            """,
            props=props
        )

driver.close()

print("Woman node features added")


# --- EXPORT GRAPH TO PYTORCH GEOMETRIC

def export_to_pyg(graph: Neo4jGraph) -> HeteroData:
    data = HeteroData()
    node_map = {}

    nodes = graph.query("""
        MATCH (n)
        RETURN id(n) AS id, labels(n) AS labels
    """)

    for n in nodes:
        node_type = n["labels"][0]
        if "x" not in data[node_type]:
            data[node_type].x = torch.empty((0, 1))
        idx = data[node_type].x.size(0)
        data[node_type].x = torch.cat(
            [data[node_type].x, torch.ones((1, 1))], dim=0
        )
        node_map[n["id"]] = (node_type, idx)

    edges = graph.query("""
        MATCH (a)-[r]->(b)
        RETURN id(a) AS src, type(r) AS rel, id(b) AS dst
    """)

    for e in edges:
        src_type, src_idx = node_map[e["src"]]
        dst_type, dst_idx = node_map[e["dst"]]
        key = (src_type, e["rel"], dst_type)

        if key not in data.edge_index_dict:
            data[key].edge_index = torch.empty((2, 0), dtype=torch.long)

        data[key].edge_index = torch.cat(
            [
                data[key].edge_index,
                torch.tensor([[src_idx], [dst_idx]])
            ],
            dim=1
        )

    return data


pyg_data = export_to_pyg(graph)

torch.save(pyg_data, "periGNNosis_graph.pt")
df.to_csv("periGNNosis_metadata.csv", index=False)
