import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import HeteroData

# Neo4j + LangChain
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_DIR = "./data"
NARRATIVE_FILE = "embedded_CBPTSD.csv"
PDI_FILE = "CBEx_pdi.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

LLAMA_MODEL = "llama3.1:8b"
MIN_WORDS = 30
SEED = 42

# Embedding
JINA_MODEL = "jinaai/jina-embeddings-v2-base-en"
NARR_BATCH = 32
ENTITY_ENCODE_BATCH = 128
DB_WRITE_CHUNK = 500
ENTITY_FETCH_LIMIT = 5000

OUT_GRAPH_PT = "periGNNosis_graph.pt"
OUT_META_CSV = "periGNNosis_metadata.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# LOAD DATA
# ==============================================================================
df = pd.read_csv(os.path.join(DATA_DIR, NARRATIVE_FILE))
df = df[df["source"] == "CBEx"].copy()

df_pdi = pd.read_csv(os.path.join(DATA_DIR, PDI_FILE))
df = pd.merge(df, df_pdi, on="record_id", how="left")

df["word_count"] = df["narrative"].fillna("").str.split().str.len()
df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)

PDI_COLS = [c for c in df.columns if c.startswith("pdi_")]
if "CB_PTSD" not in df.columns:
    raise ValueError("Expected column 'CB_PTSD' in the merged dataframe.")
if "obstetric_complication" not in df.columns:
    raise ValueError("Expected column 'obstetric_complication' in the merged dataframe.")

# Make record_id string for consistent joins
df["record_id"] = df["record_id"].astype(str)


# ==============================================================================
# CONNECT TO NEO4J
# ==============================================================================
llm = ChatOllama(model=LLAMA_MODEL, temperature=0.0, num_ctx=8192)

graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[],   # keep open; you can restrict if needed
    strict=False
)

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# Clean database (safe for Neo4j Desktop)
graph.query("MATCH (n) DETACH DELETE n")

# Helpful indexes
graph.query("""
CREATE INDEX doc_id IF NOT EXISTS
FOR (d:Document) ON (d.id)
""")
graph.query("""
CREATE INDEX woman_record_id IF NOT EXISTS
FOR (w:Woman) ON (w.record_id)
""")

# ==============================================================================
# STEP 1-4: BUILD GRAPH VIA LLMGraphTransformer
# ==============================================================================
documents = [
    Document(
        page_content=row["narrative"],
        metadata={
            # LangChain Neo4jGraph typically writes Document nodes with `id`
            "id": row["record_id"],
            "record_id": row["record_id"],
        }
    )
    for _, row in df.iterrows()
]

graph_docs = []
for doc in tqdm(documents, desc="LLM graph extraction"):
    graph_docs.extend(graph_transformer.convert_to_graph_documents([doc]))

graph.add_graph_documents(graph_docs)

# IMPORTANT:
# Treat the LangChain-created :Document nodes as the "Woman" nodes (they have the edges).
# Add :Woman label and ensure record_id is present on the SAME nodes.
graph.query("""
MATCH (d:Document)
SET d:Woman,
    d.record_id = coalesce(d.record_id, d.id)
""")

# ==============================================================================
# ADD CLINICAL PROPERTIES (PDI, label, complication) ON THE SAME Woman NODES
# ==============================================================================
rows = []
for _, row in df.iterrows():
    r = {
        "record_id": str(row["record_id"]),
        "label": int(row["CB_PTSD"]),
        "obstetric_complication": int(row["obstetric_complication"]),
    }
    for c in PDI_COLS:
        # robust cast; fill missing with 0.0
        val = row[c]
        r[c] = float(val) if pd.notna(val) else 0.0
    rows.append(r)

graph.query("""
UNWIND $rows AS row
MATCH (w:Woman {record_id: row.record_id})
SET w += row
""", params={"rows": rows})

print("Woman clinical features added on Document/Woman nodes.")


# ==============================================================================
# COMPUTE JINA EMBEDDINGS FOR ALL NODES
# ==============================================================================
print("Loading Jina embeddings model...")
model = SentenceTransformer(JINA_MODEL, trust_remote_code=True)

# Guard against silent truncation (only works if model supports it)
try:
    model.max_seq_length = 8192
except Exception:
    pass
print("Model max_seq_length:", getattr(model, "max_seq_length", "unknown"))

# --- Embed Woman nodes (narratives) ---
print("Embedding Woman (narrative) nodes...")
texts = [doc.page_content or "" for doc in documents]
rids = [str(doc.metadata["record_id"]) for doc in documents]

women_payload = []
for start in tqdm(range(0, len(texts), NARR_BATCH), desc="Encoding narratives"):
    batch_texts = texts[start:start + NARR_BATCH]
    batch_rids = rids[start:start + NARR_BATCH]

    embs = model.encode(
        batch_texts,
        normalize_embeddings=True,
        batch_size=NARR_BATCH,
        show_progress_bar=False,
    )

    for rid, emb in zip(batch_rids, embs):
        women_payload.append({"record_id": rid, "embedding": emb.tolist()})

for i in tqdm(range(0, len(women_payload), DB_WRITE_CHUNK), desc="Writing narrative embeddings"):
    chunk = women_payload[i:i + DB_WRITE_CHUNK]
    graph.query(
        """
        UNWIND $data AS row
        MATCH (w:Woman {record_id: row.record_id})
        SET w.embedding = row.embedding
        """,
        params={"data": chunk},
    )

# --- Embed entity nodes (non-Woman nodes) ---
print("Embedding entity nodes (non-Woman)...")

while True:
    entities = graph.query(
        """
        MATCH (n)
        WHERE NOT n:Woman
          AND n.embedding IS NULL
          AND exists(n.id) AND n.id IS NOT NULL AND n.id <> ""
        RETURN elementId(n) AS node_id, n.id AS text
        LIMIT $lim
        """,
        params={"lim": ENTITY_FETCH_LIMIT},
    )
    if not entities:
        break

    entity_texts = [e["text"] for e in entities]
    entity_ids = [e["node_id"] for e in entities]

    embs = model.encode(
        entity_texts,
        normalize_embeddings=True,
        batch_size=ENTITY_ENCODE_BATCH,
        show_progress_bar=False,
    )

    batch_data = [{"node_id": nid, "embedding": emb.tolist()} for nid, emb in zip(entity_ids, embs)]

    for i in tqdm(range(0, len(batch_data), 1000), desc="Writing entity embeddings"):
        chunk = batch_data[i:i + 1000]
        graph.query(
            """
            UNWIND $data AS row
            MATCH (n) WHERE elementId(n) = row.node_id
            SET n.embedding = row.embedding
            """,
            params={"data": chunk},
        )

print("Done: embeddings written for Woman and entity nodes.")


# ==============================================================================
# EXPORT GRAPH TO PYTORCH GEOMETRIC (HeteroData)
# ==============================================================================
def export_to_pyg(graph: Neo4jGraph, pdi_cols: list[str]) -> HeteroData:
    data = HeteroData()
    node_map = {}  # neo4j internal id -> (node_type, local_idx)

    # Fetch nodes with embeddings + (for Woman) clinical features
    # Note: nodes can have multiple labels; we treat 'Woman' as special.
    node_query = f"""
    MATCH (n)
    RETURN id(n) AS neo_id,
           labels(n) AS labels,
           n.embedding AS embedding,
           n.record_id AS record_id,
           n.label AS label,
           n.obstetric_complication AS obstetric_complication,
           {", ".join([f"n.{c} AS {c}" for c in pdi_cols])}
    """
    nodes = graph.query(node_query)

    # Collect features by node type
    feats = {}     # node_type -> list[list[float]]
    ys = {}        # node_type -> list[int] (only for Woman typically)

    # Determine embedding dim dynamically (fallback to 768)
    emb_dim = 768
    for n in nodes:
        emb = n.get("embedding", None)
        if isinstance(emb, list) and len(emb) > 0:
            emb_dim = len(emb)
            break

    woman_dim = emb_dim + len(pdi_cols) + 1

    for n in nodes:
        labels = n.get("labels", []) or []
        is_woman = "Woman" in labels
        node_type = "Woman" if is_woman else (labels[0] if len(labels) else "Node")

        emb = n.get("embedding", None)
        if not (isinstance(emb, list) and len(emb) == emb_dim):
            emb = [0.0] * emb_dim

        if is_woman:
            pdis = []
            for c in pdi_cols:
                v = n.get(c, 0.0)
                try:
                    pdis.append(float(v) if v is not None else 0.0)
                except Exception:
                    pdis.append(0.0)
            obst = n.get("obstetric_complication", 0)
            try:
                obst = float(obst) if obst is not None else 0.0
            except Exception:
                obst = 0.0

            x = emb + pdis + [obst]
            # Ensure fixed length
            if len(x) != woman_dim:
                x = (x + [0.0] * woman_dim)[:woman_dim]

            label = n.get("label", None)
            if label is None:
                raise ValueError("Missing label on at least one Woman node. Check clinical feature write step.")
            y = int(label)

            feats.setdefault(node_type, []).append(x)
            ys.setdefault(node_type, []).append(y)
        else:
            feats.setdefault(node_type, []).append(emb)

        local_idx = len(feats[node_type]) - 1
        node_map[n["neo_id"]] = (node_type, local_idx)

    # Assign x / y tensors
    for nt, X in feats.items():
        data[nt].x = torch.tensor(X, dtype=torch.float)

    if "Woman" in ys:
        data["Woman"].y = torch.tensor(ys["Woman"], dtype=torch.long)

    # Fetch edges
    edges = graph.query("""
        MATCH (a)-[r]->(b)
        RETURN id(a) AS src, type(r) AS rel, id(b) AS dst
    """)

    # Build edge indices per relation
    edge_lists = {}  # (src_type, rel, dst_type) -> [ [src_idx...], [dst_idx...] ]

    for e in edges:
        if e["src"] not in node_map or e["dst"] not in node_map:
            continue

        src_type, src_idx = node_map[e["src"]]
        dst_type, dst_idx = node_map[e["dst"]]
        key = (src_type, e["rel"], dst_type)

        if key not in edge_lists:
            edge_lists[key] = [[], []]
        edge_lists[key][0].append(src_idx)
        edge_lists[key][1].append(dst_idx)

    for key, (srcs, dsts) in edge_lists.items():
        data[key].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

    return data


pyg_data = export_to_pyg(graph, PDI_COLS)
torch.save(pyg_data, OUT_GRAPH_PT)

df.to_csv(OUT_META_CSV, index=False)

print(f"Saved PyG graph: {OUT_GRAPH_PT}")
print(f"Saved metadata : {OUT_META_CSV}")

# Cleanup
gc.collect()
