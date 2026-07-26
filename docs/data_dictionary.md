# docs/data_dictionary.md (skeleton — finalize in Stage 6)

> **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.

## Node: Transaction

| Property      | Type    | Source       | Description                          |
|---------------|---------|--------------|--------------------------------------|
| txId          | string  | Elliptic     | Anonymized transaction identifier    |
| timeStep      | int     | Elliptic     | Temporal snapshot (1–49)             |
| class         | string  | Elliptic     | "1"=illicit, "2"=licit, "unknown"    |
| f1..f94       | float32 | Elliptic     | Local transaction features (anon.)   |
| f95..f166     | float32 | Elliptic     | 1-hop aggregated features (anon.)    |
| tx_freq       | float32 | Engineered   | In+out degree per node/timestep      |
| amount_mean   | float32 | Engineered   | Mean BTC amount (Etherscan or proxy) |
| amount_skew   | float32 | Engineered   | Skewness of amounts                  |
| address_age   | float32 | Engineered   | Timestep of first appearance         |
| clustering_coeff | float32 | GDS       | Local clustering coefficient         |
| burst_score   | float32 | Engineered   | Z-score tx count vs trailing avg     |
| pageRank      | float32 | GDS          | PageRank centrality                  |
| communityId   | int     | GDS          | Louvain community ID                 |
| risk_score    | float32 | Model (batch)| Illicit probability [0.0, 1.0]       |
| predicted_label | string | Model (batch)| "illicit"/"licit"/"unknown"         |
| confidence    | float32 | Model (batch)| max(softmax probs)                   |
| embedding     | float[] | Model (batch)| Final hidden layer (dim=128)         |

## Edge: FLOWS_TO

| Property | Type | Source   | Description              |
|----------|------|----------|--------------------------|
| (none)   | —    | Elliptic | Directed BTC transaction |
