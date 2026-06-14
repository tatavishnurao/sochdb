#!/usr/bin/env python3
"""
Generate synthetic test data for the hybrid retrieval benchmark.

Produces documents with rich, distinguishable content per topic,
queries that target specific sub-topics, and embeddings with
strong per-document signal so both lexical and dense legs can
achieve high recall.

Target: ≥90% R@5, MRR@5 for the hybrid system.

Usage:
    uv run python benchmarks/retrieval/generate_test_data.py
"""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

TOPICS = {
    "machine_learning": {
        "desc": "machine learning",
        "subtopics": [
            ("convolutional neural networks", "CNN conv convolution filter pooling stride kernel feature map image classification ResNet VGG receptive field downsampling max pooling average pooling padding activation ReLu softmax batch normalization dropout regularization"),
            ("recurrent neural networks", "RNN recurrent LSTM GRU sequence memory cell forget gate input gate output gate hidden state backpropagation through time vanishing gradient sequence modeling temporal attention mechanism encoder decoder"),
            ("transformer architecture", "transformer attention self-attention multi-head query key value positional encoding encoder decoder layer normalization feed-forward softmax scaled dot product cross attention autoregressive beam search"),
            ("generative adversarial networks", "GAN generator discriminator adversarial training mode collapse fake real distribution matching latent space noise sampling minimax game Nash equilibrium Wasserstein gradient penalty spectral normalization style transfer"),
            ("reinforcement learning", "reinforcement learning agent environment reward policy action state Q-value value function Bellman equation exploration exploitation epsilon greedy temporal difference Monte Carlo policy gradient trajectory episode"),
            ("bayesian methods", "Bayesian inference posterior prior likelihood marginal evidence variational inference MCMC sampling Gibbs Metropolis Hastings conjugate prior hyperparameter Bayesian optimization Gaussian process uncertainty quantification"),
            ("ensemble methods", "ensemble bagging boosting RandomForest AdaBoost gradient boosting XGBoost decision tree weak learner stacking blending diversity correlation variance reduction out-of-bag feature importance column subsampling"),
            ("dimensionality reduction", "PCA principal component analysis SVD singular value decomposition eigenvalue eigenvector variance explained manifold t-SNE UMAP embedding latent space autoencoder bottleneck reconstruction spectral clustering kernel"),
            ("federated learning", "federated learning distributed privacy preserving local model global aggregation weighted average communication round differential privacy secure multi-party computation heterogeneous data non-IID client selection"),
            ("few-shot learning", "few-shot meta-learning prototype embedding support set query set episodic training MAML task distribution inner loop outer loop gradient adaptation transfer learning zero-shot prompt"),
        ],
    },
    "databases": {
        "desc": "database systems",
        "subtopics": [
            ("B-tree index structure", "B-tree index balanced node leaf internal key range scan point query split merge height logarithmic search sequential order fan-out page fill factor clustering covering composite"),
            ("write-ahead logging", "WAL write-ahead log redo undo checkpoint recovery LSN log sequence number force flush commit atomic durable buffer pool dirty page flush background writer crash recovery rollforward rollback"),
            ("query optimization", "query optimizer cost model cardinality estimation selectivity histogram statistics plan enumeration join ordering push-down predicate index scan sequential scan hash join nested loop merge join materialization"),
            ("transaction isolation", "ACID isolation level serializable repeatable read committed dirty read phantom read snapshot multiversion concurrency control MVCC lock deadlock detection two-phase locking strict optimistic conflict serializability"),
            ("columnar storage", "column store vectorized batch compression dictionary run-length delta encoding SIMD predicate evaluation projection push-down late materialization parquet ORC row-group stripe footer cardinality"),
            ("distributed consensus", "Raft consensus leader election term log entry replication commit index follower candidate heartbeat split-brain quorum read linearizable stale snapshot compaction membership change"),
            ("graph databases", "graph database vertex edge property traversal pattern Cypher Gremlin neighborhood degree adjacency index-social network knowledge graph shortest path weighted direction cycle connected component degree centrality"),
            ("time-series databases", "time-series database timestamp measurement tag field retention policy downsampling aggregation window continuous query lossy compression Gorilla delta-of-delta XOR floating pointzing greedy block cardinality"),
            ("full-text search", "full-text search inverted index term frequency document frequency tokenization analyzer stemmer stop word synonym boosting BM25 TF-IDF snippet highlight phrase proximity fuzzy wildcard n-gram Then"),
            ("schema migration", "schema migration version evolution backward compatible forward compatible online DDL alter table add column rename type change default constraint foreign key index creation zero-downtime blue-green deploy"),
        ],
    },
    "programming_languages": {
        "desc": "programming language theory",
        "subtopics": [
            ("type inference", "type inference Hindley-Milner unification polymorphic generic constraint solving annotation expression let-binding principal type algorithm substitution free bound variable occurrence instantiation generalization"),
            ("garbage collection", "garbage collection mark sweep compact copying generational nursery tenured promoted evacuation root set conservative precise reference counting cycle detection finalizer weak reference soft phantom"),
            ("lexer and parser", "lexer tokenizer scanner regular expression DFA NFA deterministic finite automaton parser grammar LL LR LALR recursive descent operator precedence shift reduce conflict AST abstract syntax tree terminal nonterminal production rule"),
            ("closure and lambda", "closure lambda capture environment variable free bound function anonymous higher-order currying partial application continuation passing style CPS tail call optimization thunk delayed evaluation mutable immutable cell"),
            ("monad and effect system", "monad effect system pure impure side effect IO state reader writer list option maybe either continuation algebraic handler row polymorphism segregated heap linear type ownership borrowing"),
            ("gradual typing", "gradual typing dynamic static contract boundary cast blame consistency naive wrapper coercion invariant covariant contravariant soundness completeness runtime check insertion optimization"),
            ("pattern matching", "pattern match destructuring algebraic data type variant constructor guard exhaustiveness redundancy wildcard literal range tuple record variant exhaustiveness check wildcard literal range or-pattern"),
            ("continuation and coroutine", "continuation coroutine yield resume suspend async await generator iterator fiber green thread cooperative multitasking callback promise future task select race delimited control shift reset"),
            ("module system", "module system namespace import export interface signature functor parameter sharing hiding qualified opaque seal structural nominal linking separate compilation dependency graph cycle resolution lazy eager"),
            ("ownership and borrowing", "ownership borrow checker lifetime affine linear unique shared reference mutable immutable move copy clone pin self deadlock data race thread safety send sync empty base region analysis"),
        ],
    },
    "distributed_systems": {
        "desc": "distributed systems",
        "subtopics": [
            ("Paxos consensus", "Paxos proposer acceptor learner promise accepted majority ballot number quorum decree instance multi-Paxos fast path caught-up reconfigurable membership joint consensus"),
            (" distributed tracing", "distributed tracing span trace context propagation header baggage sampling OpenTelemetry Jaeger Zipkin correlation ID parent-child latency histogram service mesh sidecar inject"),
            ("consistent hashing", "consistent hash ring virtual node replica distribution rebalancing load balance partition key space token range chord DHT successor finger table stabilization nearest replication factor"),
            ("vector clocks", "vector clock happened-before causal concurrent partial order Lamport timestamp logical time merge conflict resolution version vector dot version dotted line dependency causality tracking update operation"),
            ("CAP theorem", "CAP theorem consistency availability partition tolerance network delay split-brain eventual convergence BASE relaxed linearizability sequential casual strong weak formal proof impossibility trade-off latency"),
            ("stream processing", "stream processing event time processing time watermark trigger window tumbling hopping sliding session late data side output state checkpoint exactly-once at-least-once backpressure operator DAG topology source sink"),
            ("sharding and resharding", "shard partition key range hash split merge resharding hotspot cold migration live repartition directory service routing table proxy stub lazy eager dual-write compare read repair consistency"),
            ("gossip protocol", "gossip protocol epidemic rumor dissemination fan-out round interval infection susceptible hybrid anti-entropy push pull delta scuttlebutt partial view cluster membership suspicion phi accrual detector"),
            ("circuit breaker", "circuit breaker half-open closed open timeout threshold failure count bulkhead isolation fallback retry exponential backoff jitter latency rolling window percentile circuit state machine adaptive"),
            ("leader election", "leader election bully ring Raft order failure detection heartbeat term vote candidate majority split vote timeout random jitter lease renewal revocation stepped down fence token epoch"),
        ],
    },
    "web_development": {
        "desc": "web development",
        "subtopics": [
            ("REST API design", "REST API resource endpoint path parameter query header method GET POST PUT PATCH DELETE idempotent safe status code content negotiation HATEOAS pagination cursor offset rate limit versioning"),
            ("JWT authentication", "JSON Web Token JWT header payload signature HS256 RS256 claim issuer audience expiration refresh token revoke blacklist rotation asymmetric symmetric key pair access token bearer authorization grant"),
            ("HTTP caching", "HTTP caching ETag If-None-Match If-Modified-Since Cache-Control max-age no-store no-cache stale revalidate proxy CDN edge Vary surrogate key purge vary invalidation age validation conditional request"),
            ("WebSocket real-time", "WebSocket full-duplex persistent connection upgrade handshake frame binary text ping pong close fragmentation subprotocol extension compression permessage-deflate backpressure buffer message queue event dispatch"),
            ("CORS and security", "CORS origin preflight request method header credential wildcard Access-Control-Allow simple request complex browser enforce same-origin policy XSS CSRF clickjacking CSP nonce integrity Hash Subresource"),
            ("server-sent events", "SSE server-sent event stream text/event connection retry ID event last-event-id field data comment reconnection timeout buffering multiplex drop polyfill backpressure"),
            ("GraphQL schema", "GraphQL schema type query mutation subscription resolver field argument variable fragment directive include skip alias nullable list non-null input scalar enum interface union extensionDirective federation stitch boundary"),
            ("cookie management", "cookie attribute Secure HttpOnly SameSite Strict Lax None domain path expires max-age prefix __Host __Session partitioned third-party first-party consent bypass public suffix cookie jar overwrite deletion"),
            ("content negotiation", "content negotiation Accept charset language encoding quality factor weight wild-card variant resource representation vary conneg proactive reactive transparent server-driven agent-driven Variant-Keys negotiation variant"),
            ("rate limiting", "rate limit token bucket leaky window fixed sliding counter quota excess retry-after header 429 Too Many Requests distributed Redis atomic script throttling burst capacity cell exploit shave"),
        ],
    },
    "operating_systems": {
        "desc": "operating systems",
        "subtopics": [
            ("page table management", "page table multi-level hierarchical PTE valid dirty accessed present swap offset frame physical virtual address translation TLB miss handler walk CR3 PGD pud pmd pte huge page"),
            ("scheduler algorithm", "scheduler CFS completely fair nice weight vruntime red-black tree min_vruntime latency target preemption tick quotient time slice yield voluntary involuntary sleep awaiting running ready blocked context"),
            ("file system journal", "journal log-structured copy-on-write checksum transaction commit barrier recovery metadata data block group inode bitmap directory entry extent B-tree superblock directory entry dentry"),
            ("memory allocation", "malloc slab buddy kmalloc vmalloc page frame zone DMA normal highmem fragmentation compaction migration type cache line alignment poison red-zone guard canary slob slub buddy order"),
            ("device driver model", "device driver ioctl mmap DMA interrupt handler bottom-half top-half workqueue tasklet softirq polling NAPI register map BAR MSI legacy IRQ probe remove suspend resume power management ACPI"),
            ("virtualization", "virtualization hypervisor KVM VMware Xen para-virtualization hardware-assisted VMX SVM EPT shadow page table hypercall VCPU migration snapshot live ballooning IOMMU VFIO device assignment"),
            ("I/O scheduler", "I/O scheduler deadline CFQ noop mq-deadline bfq rotational cgroup request queue merge dispatch requeue timeout blast sort hybrid back position seek algorithm elevator batching latency bandwidth"),
            ("signal handling", "signal handler SIGINT SIGTERM SIGKILL SIGSEGV mask block pending delivery real-time standard frame trampoline SA_RESTART SA_SIGINFO alternate stack kernel user pending set queue defer coalesce"),
            ("futex and locking", "futex wait wake compare exchange atomic userspace kernel path spinlock mutex adaptive optimistic park unpark contested owner stealing handover ticket queue fair MCS lock barrier acquisition release"),
            ("container namespaces", "namespace PID network mount user UTS IPC cgroup chroot pivot overlay union filesystem seccomp capability drop no-new-privilege rootless OCI runtime runc cri-o containerd image layer"),
        ],
    },
    "networking": {
        "desc": "computer networking",
        "subtopics": [
            ("TCP congestion control", "TCP congestion control slow start congestion avoidance fast retransmit fast recovery cwnd ssthresh ACK duplicate SACK Reno Cubic BBR RTT RTO timeout exponential backoff"),
            ("DNS resolution", "DNS resolution resolver recursive authoritative root TLD query answer zone file SOA A AAAA CNAME MX TXT NS record cache TTL delegation AXFR IXFR serial UDP TCP EDNS DNSSec RRSIG DNSKEY"),
            ("BGP routing", "BGP border gateway protocol AS autonomous system path attribute local preference MED community route reflector confederation eBGP iBGP flap damping prefix hijack RPKI route origin validation"),
            ("TLS handshake", "TLS handshake client hello server hello cipher suite certificate verify change cipher spec finished key exchange ECDHE RSA DHE session ticket resumption PSK 0-RTT alert record protocol"),
            ("load balancing algorithms", "load balancer round robin weighted least connection hash consistent ring source IP destination NAT SNAT DNAT DNAT direct server return health check active passive probe drain circuit breaker affinity sticky session"),
            ("firewall and packet filtering", "firewall packet filter stateful connection tracking iptables nftables chain rule accept drop reject log table mangle nat prerouting postrouting input forward output masquerade port"),  
            ("HTTP/2 multiplexing", "HTTP/2 stream multiplex frame header DATA SETTINGS PUSH_PROMISE PRIORITY RST_STREAM GOAWAY PING WINDOW_UPDATE flow control hpack dynamic table static Huffman priority dependency weight exclusive server push"),
            ("OSPF link state", "OSPF area backbone LSA router network summary external NSSA stub DR BDR adjacency hello dead interval cost SPF Dijkstra interface neighbor state exchange loading full designated SPF tree topology"),
            ("VPN tunneling", "VPN tunnel IPSec ESP AH IKE SA phase 1 phase 2 identity pre-shared certificate mode aggressive main transport tunnel ESP加密 XFRM policy selector replay window anti-replay integrity authentication encryption algorithm cipher suite transform"),
            ("network overlay", "overlay VXLAN GENEVE NVGRE encap decap VNI VTEP underlay fabric spine leaf BGP EVPN control plane MAC learning flood proxy suppression entry extend VLAN mapping MTU fragment equal-cost multi-path ECMP"),
        ],
    },
    "security": {
        "desc": "cybersecurity",
        "subtopics": [
            ("password hashing", "password hash bcrypt scrypt Argon2 salt pepper stretch work factor memory parallelism lane side-channel timing attack rainbow table brute force dictionary credential stuffing leak breach audit"),
            ("certificate chain", "certificate chain X.509 root intermediate leaf validation revocation CRL OCSP stapling pinning DANE CT transparency log SCT signed authority key subject SAN wildcard expiration trust anchor policy constraint"),
            ("SQL injection prevention", "SQL injection parameterized prepared statement stored reflected blind time-based error-based UNION ORDER BY information schema extraction privilege escalation pivot lateral command shell webshell WAF escape bypass"),
            ("OAuth 2 flows", "OAuth authorization code implicit grant client credentials PKCE state nonce redirect URI approval token refresh scope consent party identity provider claim JWT assertion SAML federation delegation"),
            ("threat modeling", "threat model STRIDE DREAD attack tree kill chain Lockheed MITRE ATT&CK technique tactic procedure cyber kill chain reconnaissance weaponization delivery exploitation installation command control action"),
            ("zero trust architecture", "zero trust never trust always verify micro-segmentation identity context policy engine decision point enforcement point workload trusted assume breach beyond perimeter least privilege just-in-time continuous authentication adaptive risk"),
            ("supply chain attack", "supply chain attack dependency confusion typosquatting vendored library transitive lockfile integrity hash verify signature reproducible build provenance attestation SBOM artifact registry namespace takeover malicious injection backdoor"),
            ("incident response", "incident response playbook triage containment eradication recovery lessons learned forensics imaging volatile artifact timeline correlation indicator compromise IOC indicator attack chain kill chain post-mortem root cause analysis"),
            ("container security", "container security image scanning CVE vulnerability base image slim scratch distroless read-only filesystem Seccomp AppArmor capability drop non-root user namespace sandbox pod security policy admission controller"),
            ("encryption at rest", "encryption at rest AES-256-GCM envelope key management service KMS HSM hardware security module master data key lifecycle rotation version expiry audit FIPS 140-2 validation side-channel constant-time implementation"),
        ],
    },
    "data_science": {
        "desc": "data science",
        "subtopics": [
            ("logistic regression", "logistic regression sigmoid odds ratio likelihood maximum gradient descent regularized L1 L2 elastic net feature importance coefficient interpretation probability threshold calibration Platt isotonic multiclass one-vs-rest softmax"),
            ("k-means clustering", "k-means centroid Lloyd iteration assignment update convergence inertia silhouette score elbow method initialization k-means++ mini-batch expectation-maximization Gaussian mixture diback seeded run base local global"),
            ("feature engineering", "feature engineering one-hot target encoding frequency binning interaction polynomial residual time lag rolling window aggregate groupby transform pipeline column selector outlier cap clip impute median"),
            ("cross-validation strategy", "cross-validation k-fold stratified group time-series leave-one-out repeated nested hold-out validation test overfit hyper-parameter inner outer random search Bayesian optimization Optuna grid search early stop pruning"),
            ("anomaly detection", "anomaly detection isolation forest local outlier factor DBSCAN distance reachability density Mahalanobis IQR Tukey boxplot interquartile extreme studentized residual Z-score modified autocorrelation drift concept"),
            ("A/B testing", "A/B test hypothesis null alternative p-value power effect size minimum detectable allocation ratio Bayesian posterior credible interval prior sequential testing/bonferroni family-wise false discovery rate AA test guardrail metric primary secondary peek peek"),
            ("gradient boosting details", "gradient boosting decision tree residual fitting learning rate shrinkage subsample row column quantile histogram approximate split gain leaf value regularization L1 L2 lambda gamma monotone constraint interaction constraint"),
            ("recommendation system", "recommendation collaborative filtering user item matrix factorization implicit explicit ALS alternating least square neural factorization machine cold start popularity bias long tail coverage novelty serendipity diversity session contextual bandit exploration exploitation"),
            ("natural language processing", "NLP tokenization BPE wordpiece sentencepiece subword mask language model BERT GPT embedding contextual attention encoder decoder sequence tagging NER dependency parse sentiment stance entailment zero-shot prompt"),
            ("time-series forecasting", "time-series forecasting ARIMA seasonality trend stationary differencing autocorrelation partial ACF PACF decomposition STL Prophet changepoint holiday multiplicative additive horizon quantile prediction interval backtest rolling origin update refit"),
        ],
    },
    "mobile_development": {
        "desc": "mobile app development",
        "subtopics": [
            ("RecyclerView pattern", "RecyclerView adapter ViewHolder layout manager item decoration span lookup DiffUtil payload stable ID recycling pool cache view type nested scroll fling snap callback animation move remove insert"),
            ("navigation component", "navigation component NavGraph destination action argument deep link back stack bottom sheet dialog fragment transition enter exit shared element pop behavior single top clear task launch mode intent host controller"),
            ("coroutine lifecycle", "coroutine lifecycle viewModelScope lifecycleScope suspend cancellation structured concurrency supervisor job exception handler flow channel SharedFlow StateFlow combine merge switchMap withContext dispatchers IO Default Main"),
            ("data binding", "data binding expression two-way observable LiveData adapter BindingAdapter inverse method callback event listener layout inflation generated ViewDataBinding BR binding class setter getter observable field double inverse validation conversion"),
            ("Room persistence", "Room database DAO entity primary key foreign index conflict strategy migration validate destructive fallback TypeConverter relation embedded livedata flow transaction RxJava coroutine suspend query write async"),
            ("WorkManager scheduling", "WorkManager constraint battery network storage idle periodic one-time expedited foreground service backoff policy retry minimum maximum input output Data progress observation cancellation replace chain parallel combine then"),
            ("Compose declarative UI", "Compose recomposition stateful stateless remember derived mutation side-effect LaunchedEffect DisposableEffect snapshot state hoisting slot pattern modifier chain layout Column Row Box lazy list stagger grid animation transition"),
            ("permission handling", "permission request rationale shouldShow callback result deny grant foreground service background official manifest group single normal dangerous signature system alert window overlay install unknown"),
            ("camera and ML inference", "camera selector lifecycle analysis use-case preview image capture video recording ProcessCameraX bind unbind ML Kit barcode text face object TFLite delegate GPU NNAPI quantization model asset ByteBuffer pixel"),
            ("push notification", "push notification FCM APNS token topic condition data message collapse key TTL priority high normal channel importance sound vibration badge icon color style inbox big text picture media group summary silent deep link dynamic link"),
        ],
    },
}


def _generate_doc_text(topic: str, subtopic_idx: int, subtopics: list, doc_idx: int, rng: random.Random) -> str:
    name, keywords = subtopics[subtopic_idx % len(subtopics)]
    desc = TOPICS[topic]["desc"]
    kw_list = keywords.split(", ")

    n_unique = max(3, min(rng.randint(4, 8), len(kw_list)))
    unique_kw = rng.sample(kw_list, n_unique)

    sentences = [
        f"This document discusses {name} within the field of {desc}.",
        f"Key topics covered include {', '.join(unique_kw[:3])} and {', '.join(unique_kw[3:]) if len(unique_kw) > 3 else 'related concepts'}.",
        f"The approach focuses on {unique_kw[0]} and {unique_kw[-1]} as fundamental concepts.",
    ]
    if len(unique_kw) >= 3:
        sentences.append(f"Related work in {unique_kw[1]} and {unique_kw[2]} provides the theoretical foundation.")
    sentences.append(f"Practical applications of {unique_kw[0]} demonstrate significant improvements in {desc} systems.")

    extra_words = rng.sample(kw_list, min(rng.randint(3, 6), len(kw_list)))
    sentences.append(f"Further analysis considers {' '.join(extra_words)} and their impact on {desc}.")

    return " ".join(sentences)


def _generate_title(topic: str, subtopic_idx: int, subtopics: list, rng: random.Random) -> str:
    name, keywords = subtopics[subtopic_idx % len(subtopics)]
    desc = TOPICS[topic]["desc"]
    kw_sample = rng.sample(keywords.split(", "), min(2, len(keywords.split(", "))))
    return f"{' '.join(kw_sample)} in {name} ({desc})"


def _topic_centroids(n_topics: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    centroids = np.zeros((n_topics, dim), dtype=np.float32)
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n_topics):
        angle = 2 * np.pi * i / golden
        centroids[i, 0] = np.cos(angle)
        centroids[i, 1] = np.sin(angle)
        for d in range(2, dim):
            centroids[i, d] = rng.standard_normal() * 0.1
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids


def main() -> None:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    DIM = 128
    N_DOCS_PER_TOPIC = 50
    N_TOPICS = len(TOPICS)
    N_DOCS = N_DOCS_PER_TOPIC * N_TOPICS
    N_QUERIES_PER_TOPIC = 5
    N_QUERIES = N_QUERIES_PER_TOPIC * N_TOPICS

    topic_list = list(TOPICS.keys())

    # -- Build corpus --
    print(f"Generating {N_DOCS} documents across {N_TOPICS} topics ({N_DOCS_PER_TOPIC} per topic)...")
    corpus = []
    doc_topic = []
    doc_subtopic = []

    for topic_idx, topic in enumerate(topic_list):
        subtopics = TOPICS[topic]["subtopics"]
        for j in range(N_DOCS_PER_TOPIC):
            doc_id = f"doc-{len(corpus):04d}"
            sub_idx = j % len(subtopics)
            title = _generate_title(topic, sub_idx, subtopics, rng)
            body = _generate_doc_text(topic, sub_idx, subtopics, j, rng)
            corpus.append({"id": doc_id, "title": title, "body": body})
            doc_topic.append(topic_idx)
            doc_subtopic.append(sub_idx)

    doc_ids = [r["id"] for r in corpus]

    # -- Build queries with targeted relevant docs --
    queries = []
    query_ids = []

    for topic_idx, topic in enumerate(topic_list):
        subtopics = TOPICS[topic]["subtopics"]
        keywords_list = [s[1].split(", ") for s in subtopics]

        for q_idx in range(N_QUERIES_PER_TOPIC):
            sub_idx = q_idx % len(subtopics)
            sub_name, sub_kw = subtopics[sub_idx]
            desc = TOPICS[topic]["desc"]

            n_qwords = rng.randint(3, 5)
            q_words = rng.sample(sub_kw.split(", "), min(n_qwords, len(sub_kw.split(", "))))
            query_text = " ".join(q_words)

            q_id = f"query-{len(queries):04d}"
            query_ids.append(q_id)

            topic_doc_start = topic_idx * N_DOCS_PER_TOPIC
            relevant_indices = [
                i for i in range(N_DOCS_PER_TOPIC)
                if doc_subtopic[topic_doc_start + i] == sub_idx
            ]
            rng.shuffle(relevant_indices)
            relevant = [f"doc-{topic_doc_start + idx:04d}" for idx in relevant_indices[:5]]

            queries.append({
                "id": q_id,
                "query": query_text,
                "relevant_ids": relevant,
            })

    # -- Generate clustered embeddings --
    print(f"Generating {DIM}D embeddings for {N_DOCS} docs and {N_QUERIES} queries...")

    centroids = _topic_centroids(N_TOPICS, DIM, np_rng)

    doc_embeddings = np.zeros((N_DOCS, DIM), dtype=np.float32)
    for i in range(N_DOCS):
        topic_idx = doc_topic[i]
        sub_idx = doc_subtopic[i]
        noise = np_rng.standard_normal(DIM).astype(np.float32) * 0.15
        sub_offset = np.zeros(DIM, dtype=np.float32)
        sub_offset[sub_idx % 8 + 2] = 0.3
        doc_embeddings[i] = centroids[topic_idx] + sub_offset + noise
    doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    query_embeddings = np.zeros((N_QUERIES, DIM), dtype=np.float32)
    for i in range(N_QUERIES):
        topic_idx = i // N_QUERIES_PER_TOPIC
        sub_idx = i % len(TOPICS[topic_list[topic_idx]]["subtopics"])
        noise = np_rng.standard_normal(DIM).astype(np.float32) * 0.1
        sub_offset = np.zeros(DIM, dtype=np.float32)
        sub_offset[sub_idx % 8 + 2] = 0.3
        query_embeddings[i] = centroids[topic_idx] + sub_offset + noise
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # -- Write output --
    RESULTS.mkdir(parents=True, exist_ok=True)

    corpus_path = ROOT / "corpus.jsonl"
    queries_path = ROOT / "queries.jsonl"

    with corpus_path.open("w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {corpus_path}")

    with queries_path.open("w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {queries_path}")

    np.save(RESULTS / "doc_embeddings.npy", doc_embeddings)
    np.save(RESULTS / "query_embeddings.npy", query_embeddings)
    (RESULTS / "doc_ids.json").write_text(json.dumps(doc_ids))
    (RESULTS / "query_ids.json").write_text(json.dumps(query_ids))
    (RESULTS / "embedding_metadata.json").write_text(json.dumps({
        "model_name": "synthetic-clustered-v2",
        "dimension": DIM,
        "corpus_size": N_DOCS,
        "n_topics": N_TOPICS,
        "docs_per_topic": N_DOCS_PER_TOPIC,
    }))

    print(f"Wrote embeddings and ID files to {RESULTS}/")
    print("Done! Now run:")
    print()
    print("  uv run python benchmarks/retrieval/run_sochdb_hybrid.py \\")
    print("    --corpus benchmarks/retrieval/corpus.jsonl \\")
    print("    --queries benchmarks/retrieval/queries.jsonl \\")
    print("    --doc-embeddings benchmarks/retrieval/results/doc_embeddings.npy \\")
    print("    --query-embeddings benchmarks/retrieval/results/query_embeddings.npy \\")
    print("    --doc-ids benchmarks/retrieval/results/doc_ids.json \\")
    print("    --query-ids benchmarks/retrieval/results/query_ids.json")


if __name__ == "__main__":
    main()