# Generation pipeline

this is the pipeline for generting the input for annotation pipeline

Firstly, I tried running it with vllm, but finally used native transformers. 
The script uploads the dataset to huggingface that is compliant with the input dataset for annotation pipeline.


This script just requires transformers>=0.46.0 with apertus model.

## TODO
- create better config