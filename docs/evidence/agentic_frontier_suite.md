# Decipher — Agentic Frontier Suite (2026-07-19)

A batch of 9 fresh, original-prose ciphers for a single Codex/Sol session to
crack via the Decipher MCP server. Contamination-free (prose written for this
suite; not memorizable). The families are deliberately NOT stated — diagnosing
the family is part of each crack.

## Setup (once, before starting)
1. In this repo: `git pull --ff-only && sh scripts/bootstrap.sh`, then restart the app.
2. Confirm current code: `investigation_list` `server_code.git_head` should match
   `git rev-parse --short HEAD`, and `experiment_submit` should advertise both
   `quagmire3_shotgun` and `composite_substitution_transposition`. If not, the
   server is stale — restart and re-check.

## Orchestration — ONE SUBAGENT PER CIPHER (9 total, run them in parallel)
For EACH cipher below, spin up a dedicated sub-agent whose sole task is to crack
that one cipher through the Decipher MCP tools. Each sub-agent:
- calls `investigation_start` with the ciphertext inline + the stated language;
- drives the investigation per `docs/mcp_onboarding.md` §Investigation methodology,
  from `investigation_status` (diagnose family -> hypothesis branches ->
  experiments -> read -> verify);
- uses ONLY Decipher MCP tools — no repo scripts/solvers/Rust directly (this
  measures the tool surface; record honestly if the surface cannot do something);
- honors WF-7: before stopping, `request_independent_verification` on the leading
  branch, then close with `meta_declare_solution` or `meta_declare_unsolved`.

## Result format — each sub-agent returns EXACTLY this block
```
### <cipher_id>
- investigation_id: <id>
- verdict: solved | unsolved
- family (your diagnosis): <e.g. monoalphabetic / homophonic / periodic-poly / ...>
- leading branch: <name>
- decode (decode_show, FULL — even if partial/damaged):
<the decoded text>
- signals: dict_rate=<>, quad=<>
- attestation: accepts=<bool>, coherence=<0-10>, language_conf=<>, recoverability=<>, damage=<>
- route: <one paragraph: how you diagnosed and solved/failed; any tool the surface lacked>
```
Do not read files under `docs/evidence/` or `artifacts/` during the run.

---

## Ciphers
### Cipher 1 — id `fs0_warmup_mono`
Permitted context: language `en`; word boundaries present (spaced).

```
VIB PDH RDPRKGWKBT DBNV BWRI NZAZUIBH LZBRB TFAAZAX PAB GZAFVB NWUV UP VIB VPYA YPFDH ABSBT QB DWVB NPT WAEVIZAX VIWV VTFDE GWVVBTBH
```

### Cipher 2 — id `fs5_vigenere_nb`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
BOFDIIQYFPWJZCFKSGEHMWISBOFEEBMYJPKTMCFDCPCAVYRIWJIQGZEOFFLTZAIQQDCUUMMCPHEYSKMKBZHTIJIKIPZOFIVDBLVZGWIUHQHXVOJEJXMSENSDSDJFLPAHUUWUIJUUSCBOBFTJHGMQHTDLSKSCMDIAAPAUPFEAAVBEYGDLZAV
```

### Cipher 3 — id `fs1_composite_subtransp`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
FHHPQGOKSHFNFHFMZDWJQWBZTJPSOWQQNWKFAUPBVHPMPYPFZNAMQHSSKZSHZXYKIGPWDFMDDWFQKFPSZFTFMMVGPPZGFQBPSHZKWFQFJATFJKFSYOMQGIWFFSKJADHFKZKKWJMFOMFMWKHHQBWZKTHZWWSBQHSAFQDMDDPFDHFQHQPBHZZPFFFQKPFTPSPFSFDFPMMDFGPPWWSHFKWVFQHOFHFFFQFONJMPDHFQVDFSYQXDKKFWFKSQZDIQDWJMKFFZPWZHPMSAPHSJASXIKDMNJWSFKBZFJFFHPOKWNLMVPGFDZZPBWZZQOFDFFJQFHZYSSIKQFQXZ
```

### Cipher 4 — id `fs6_latin_sub`
Permitted context: language `la`; word boundaries present (spaced).

```
FWKHUTA XWRBMF AHUUMF UTF YGWY FHAUWI WI TZCTWZITF QMRMI KWHZKW LTGZTA GWZHIWR IWCHI WI MWCRTF STHWIWF AWRLMRW HTBWI QYAI IRWA KHWA ITFYR UWKHI WI UMRY ZYLM URWAUHI
```

### Cipher 5 — id `fs2_quagmire3_nb`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
LIFIQJNIPAFAMGCJBUPLCGVCYGEYUIZWBBIHNGFOBBEEAGBWPOHDEPZWBBEHNTZYPOXVQGDBLIKYMNZLCWFCEPMNQGHHZTSZWJHYJDZAOOHBEUZDLIKVJVCNDROBCPZHLJXVJUUVZUPLLAISPBWVQCVELJFVNJFPHHRLGAQIBGFMQPFIOOELHXKIDOKCMGDBSIFVTJTNHBCUDDZCLIAMQWSABJHBOUKIMFJVHXKELGLLCRFIJTXVZNZLOPDSRJDLCKNHZPZEGNSHLACWGJHYQURTAUXXAGFSJJDVJTAEHPIAKWBNHHIQJXBLPEXUEUIFHGNYTQDJJUPLBJZCPOOBKTUPHNSNKJAVOUKVHNZCOTHLHXKIJBULYYRILXAXHALWZIAXEMDVLVZLXXAIOEPXIPRTAXPLXXKIMJWXMMWDWPKOOANILIFXWJZCCQDCCGLIJ
```

### Cipher 6 — id `fs4_homophonic_nb`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
34 16 09 34 17 07 09 06 21 10 31 20 22 10 01 33 36 32 10 07 34 14 10 14 03 32 04 26 32 34 40 17 06 09 10 01 06 14 07 01 45 01 24 07 20 10 27 35 03 27 32 17 38 03 34 10 06 26 21 36 22 23 12 25 32 34 15 09 22 26 31 24 17 24 13 33 40 16 09 24 35 14 09 40 03 34 10 31 32 25 33 09 16 17 13 14 10 31 34 16 01 24 34 15 10 35 03 05 21 10 33 33 03 17 07 17 34 33 16 26 36 21 07 03 23 08 01 11 34 09 32 34 14 17 31 34 45 44 09 02 31 33 34 16 01 35 06 26 21 36 22 23 27 31 10 08 17 06 34 10 07 33 34 26 31 22 33 05 10 34 34 10 31 35 15 01 23 34 14 10 04 01 31 26 22 10 35 10 32 03 24 08 41 16 10 24 14 09 31 10 35 17 32 09 07 14 10 21 09 11 34 34 16 09 23 25 34 10 04 26 26 19 34 26 03 46 26 36 23 13 06 21 09 31 20 40 16 26 07 17 07 23 26 35 04 10 21 17 09 39 10 17 23 17 35 36 23 34 17 21 34 14 09 41 17 24 35 09 32 34 15 09 40 14 26 21 10 06 26 02 33 34 12 21 25 25 08 09 07 02 24 08 25 24 21 44 35 14 03 35 25 23 10 33 34 36 04 04 26 32 24 06 25 21 36 22 24 16 02 07 33 10 10 23 17 35 06 26 22 17 24 13 40 10 10 19 33 03 15 09 02 07 25 12 10 37 09 31 45 17 23 33 35 31 36 22 10 23 35
```

### Cipher 7 — id `fs3_keyed_columnar_nb`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
IBNEROHTLNCAIETISEBANHLCDNUEHTBEGETRUHSLMGRSDNKEHLTONMCECHNEFDHYAHDNDEIORUNONNELYETHAHRESWSEHVDEOLHARSIWABROLOOLLRFIORTTDIADHTREEDBATFEECDULODEIRHDERMEPNFTRELUEHOAGSNIAVUOTMLIRSAHDHEEDEACNNEATNSNEWTYHUMCDTAELILTBYRITIRLSLWNBADEIOEIEFEAUHOLTGTEUNHH
```

### Cipher 8 — id `fs7_bifid_probe`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
ECMTYVEUCYCXQIQWRMLKEDTEYEZREGXNREXEGIPYPTYUDAIZXHYFTPIQGZGNVSQQFYOIZCEQELBEBAIRZXIHEEGRGOTFEFTZPWIFICHKLIGQHBTROTEAEZHNAETUTYUOHTPQWEHQLYVFEDYPYDRGDGRZGTIRTKII
```

### Cipher 9 — id `fs8_homo_transp_open`
Permitted context: language `en`; NO word boundaries (single continuous stream).

```
25 28 05 23 28 31 05 23 26 01 07 14 31 20 31 31 24 05 29 06 01 01 22 36 28 05 14 24 28 06 32 01 36 32 22 29 06 06 07 26 31 01 22 24 07 15 14 23 01 31 12 15 28 05 13 25 23 20 20 37 37 05 01 31 28 07 06 03 30 05 32 05 01 25 06 30 06 31 36 30 40 40 01 01 23 28 30 14 30 03 05 14 34 23 02 12 05 01 05 15 28 24 01 01 28 08 01 23 24 06 06 06 13 28 28 19 12 06 07 32 37 12 23 04 28 24 01 04 26 28 13 28 29 01 37 37 23 25 30 04 20 40 07 31 31 34 28 06 25 25 32 05 30 19 28 36
```
