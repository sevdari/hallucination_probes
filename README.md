# Multitoken Hallucination Probes for Apertus

This is the codebase for the project on multitoken hallucination prediction with probes for the Apertus LLM. The project is done as a part of Large Scale AI Engineering course at ETHZ by Klejdi Sevdari, Michal Korniak and Tymoteusz Kwiecinski and supervised by Anna Hedström and Imanol Schlag.

In scope of the project we:
1. reproduced the project, along with the annotation pipeline for generating the dataset with annotated hallucination spans
2. implemented and evaluated multitoken probes concatenating the tokens
3. implemented and evaluated attention probes


In the `./clariden/` directory there are some files and scripts that help with working on the cluster, including sbatch scripts and environment files.

`./generation_pipeline` contains a script which can be used to create a dataset with model outputs using a dataset with prompts (e.g. longfact or longfact++). Such dataset with generations is a later input for an annotation pipeline (which is in `./annotation_pipeline` directory), that fact-checks the generations using an advanced model with web-search functionality. In the original paper, they used Sonnet 4.5, but we used GPT4o, because we had no access to Anthropic API.


Readme of the original repository can be found [here](README.old.md).


## Setup
We moved from `uv` to simple `pip` setup because of problems with torch cuda dependencies.
Maybe it will be good to switch back to uv, but I found out this setup to be working just fine.
