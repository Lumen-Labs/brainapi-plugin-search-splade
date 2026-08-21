# search-splade

Learned-sparse **first-stage** retriever for BrainAPI `POST /retrieve/search`. It keeps its own inverted index (plugin-local, in memory) and registers channel `plugin:splade`. It does **not** run on `/retrieve/context`. Core hybrid (passages / BM25 / dense) works if this plugin is absent.

Unknown or missing `plugin:splade` is **400**, never a silent BM25 fallback.

| | |
|---|---|
| Registry name | `search-splade` |
| Version | `0.1.0` |
| BrainAPI | `>=2.17.0` |
| Channel | `plugin:splade` |
| Default model | `naver/splade-cocondenser-ensembledistil` |
| Index | `POST /search-splade/index` |
| Health | `GET /search-splade/health` |

## Install

```bash
git clone https://github.com/Lumen-Labs/brainapi-plugin-search-splade.git plugins/search-splade
```

Or:

```bash
./bin/brainapi install search-splade
```

Restart the API. Encoding needs `torch` and `transformers` in the BrainAPI environment. The model is lazy-loaded on first encode.

## Quick start

Index text chunks already stored in the brain, then search:

```bash
curl -X POST "$BRAINAPI_URL/search-splade/index" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -d '{"brain_id": "searchbenchsmoke", "limit": 1000}'

curl -X POST "$BRAINAPI_URL/retrieve/search" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -H "X-Brain-ID: searchbenchsmoke" \
  -d '{
    "query": "navy wool coat",
    "k": 50,
    "channels": ["plugin:splade"]
  }'
```

Fuse with core passages:

```json
{ "channels": ["passages", "plugin:splade"] }
```

Benchmark harness: `--channels plugin:splade` after indexing.

## How it retrieves

1. `POST /search-splade/index` pages `get_text_chunks` (up to `limit`, max 20 000) and SPLADE-encodes each doc to a sparse `{token: weight}` vector.
2. An inverted index maps term → `[(chunk_id, weight), …]`.
3. A query is encoded the same way. Score is the dot product over overlapping terms.
4. Top `k` ids are returned to `/retrieve/search` as channel `plugin:splade`.

The index lives **in process**. Restarting the API clears it — you must re-index. `index_chunks` replaces the brain’s index (`reset` then rebuild).

## Configuration

| Env | Default |
|---|---|
| `SEARCH_SPLADE_MODEL` | `naver/splade-cocondenser-ensembledistil` |

Encoding: MLM logits → `log1p(relu)` → max-pool over sequence → nonzero vocab weights. Special tokens (`cls` / `sep` / `pad` / `unk`) are dropped. Max length 256.

Tests can inject `set_encoder(fn)`.

## API

### `GET /search-splade/health?brain_id=`

```json
{
  "plugin": "search-splade",
  "channel": "plugin:splade",
  "model": "naver/splade-cocondenser-ensembledistil",
  "loaded": false,
  "error": null,
  "index": { "brain_id": "searchbenchsmoke", "n_docs": 2043, "n_terms": 12000 }
}
```

`index` is included only when `brain_id` is passed.

### `POST /search-splade/index`

```json
{ "brain_id": "searchbenchsmoke", "limit": 1000 }
```

`limit` is `1…20000` (default 1000). Returns `{ brain_id, n_docs, n_terms }`.

## Layout

```text
search-splade/
  plugin.yaml
  main.py       # register_search_retriever("splade", …)
  encode.py     # SPLADE sparse encoder
  index.py      # inverted index + retrieve
  routes.py     # health + index
```

## Publishing

Pushes to `main` publish to the BrainAPI registry via GitHub Actions.

## License

Business Source License 1.1. See [LICENSE](LICENSE).

## Related

- [search-colbert](https://github.com/Lumen-Labs/brainapi-plugin-search-colbert)
- [search-rerank](https://github.com/Lumen-Labs/brainapi-plugin-search-rerank)
- [BrainAPI](https://github.com/Lumen-Labs/brainapi2)
- `docs/research/18-search-eval-protocol.md` on brainapi2
