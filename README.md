# Dataset Pruning with H-Score

The official code for the paper [Automatic Pruning of Fine-tuning Datasets for Transformer-based Language Models](https://arxiv.org/abs/2407.08887) at [CoLLAs 2024](https://lifelong-ml.cc/Conferences/2024).

This repository enables users to create the H-Score of their datasets and use it for dataset pruning.

As demonstrated by our [results](#experimental-results), users can take advantage of our proposed dataset pruning method to achieve faster fine-tuning runs with similar evaluation performance.
This is particularly useful for cases with multiple fine-tuning runs like NAS.

## Installation

Simply run:
`pip install -r requirements.txt`
to install the required packages.

Note that we use a specific version of the [HuggingFace Transformers](https://github.com/huggingface/transformers/) library.

## Usage

The dataset pruning functionality in this repository is tested on two models ([Roberta](https://arxiv.org/abs/1907.11692) and [OPT](https://arxiv.org/abs/2205.01068)) and five tasks ([MNLI, SST-2](https://gluebenchmark.com/), [SNLI](https://nlp.stanford.edu/projects/snli/), [SQuAD v2](https://rajpurkar.github.io/SQuAD-explorer/), and [RACE](https://www.cs.cmu.edu/~glai1/data/race/))

### Phase 1

In the first phase of dataset pruning algorithms, the dataset is analyzed to create respective dataset pruning scores.
In this phase, we first fine-tune the model with 6 different seeds and then use the model's outputs during these runs to create the scores.

The [phase1.sh](./scripts/phase1.sh) script performs these steps for all the tasks.
It defaults to Roberta Large but you can easily change the model.

At the end of this step a folder named `scores` will be created which will contain the rewards and log-probabilities for these runs.
The rewards are used to compute the H-Scores at runtime and the log probabilites are used for the *ambiguous* setup.

### Phase 2

In this phase the models are fine-tuned on the pruned datasets.
Dataset pruning is done during runtime using the files from the first phase.

The [phase2.sh](./scripts/phase2.sh) script runs the pruned fine-tuning runs.
Using the `removed_scores` array in this script you can control which pruned subsets to use for the runs.
By default, we use subsets that are reported in our results.
Note that `REMOVE_STR=06` is equivalent to our **Winning Ticket Subset**.

For pruning methods other than `hscore`, the subset size is chosen to be the same as our subset, but the data points in the pruned subsets are created using each respective method.
For example, `PRUNING_METHOD=ambiguous` and `REMOVE_STR=06` fine-tunes the model on a subset with the same size as our winning ticket subset, but the datapoints in the subset are chosen based on the value of their *variability* score.

## Experimental Results

The following table includes a comparison of the evaluation performance of fine-tuning Roberta Large on our proposed Winning Ticket Subsets against the entire fine-tuning dataset (baseline).

For more results, refer to Sections 4.1 and A.4 of the paper.

|Task|Baseline|| Winning Ticket Subset||
|-----------|------------------|-------|------------|-------|
||**Subset Size**|**Evaluation Performance**|**Subset Size**|**Evaluation Performance**|
|MNLI M.|100%|90.04|27%|90.57|
|MNLI MisM.|100%|89.99|27%|90.22|
|SNLI|100%|92.25|21%|92.50|
|SST-2|100%|95.79|24%|95.80|
|RACE|100%|84.67|62%|82.95|
|SQuAD v2 (F1)|100%|86.92|47%|87.70|

## Support

Please contact [mohammadreza.tayaranian@mail.mcgill.ca](mailto:mohammadreza.tayaranian@mail.mcgill.ca) for questions or issues.
