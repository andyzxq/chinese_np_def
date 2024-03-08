# Computational Modelling of Plurality and Definiteness in Chinese Noun Phrases

## The work are presented in our LERC-Coling 2024 paper [[arxiv]](http://arxiv.org/abs/2403.04376).

Theoretical linguists have suggested that some languages (e.g., Chinese and Japanese) are “cooler” than other languages based on the observation that the intended meaning of phrases in these languages depends more on their contexts. As a result, many expressions in these languages are shortened, and their meaning is inferred from the context. In this paper, we focus on the omission of the plurality and definiteness markers in Chinese noun phrases (NPs) to investigate the predictability of their intended meaning given the contexts. To this end, we built a corpus of Chinese NPs, each of which is accompanied by its corresponding context, and by labels indicating its singularity/plurality and definiteness/indefiniteness. We carried out corpus assessments and analyses. The results suggest that Chinese speakers indeed drop plurality and definiteness markers very frequently. Building on the corpus, we train a bank of computational models using both classic machine learning models and state-of-the-art pre-trained language models to predict the plurality and definiteness of each NP. We report on the performance of these models and analyse their behaviours.

## Dataset

We automatility constructed a large-scale dataset using parallel Chinese-English corpus in which each NP is annotated with its plurality and definiteness. The dataset contains more than 2M annotated NPs. We used 124K of them in this paper. The ratio of singular to plural is 3:1, and the ratio of indefinite and definite is about 1:1.

## Getting Started

Clone the repo to get the dataset.

```
https://github.com/andyzxq/Computational-Modelling-of-Plurality-and-Definiteness-in-Chinese-NPs.git
```

The folder stucture is as follows:

```
++ (root)
++++ code
++++++ dataset_process: Code related to dataset construction, including word alignment, NP identification, matching, post processing, and annotation
++++++ dataset_analysis: Some code for analyzing datasets (not all uploaded)
++++++ models: Code for models
++++++ model_analysis: Some code for model analysis (not all uploaded)
++++ dataset
++++++ all_data: All annotated data, totaling approximately 2M
++++++ used_data_for_models: training, develop and test dataset
++++++++ train.csv
++++++++ dev.csv
++++++++ test.csv
++++++ dataset_assessment: Dataset assessment results (including annotation results for each human annotator)
++++ results
++++++ all_models_result.xls: Summary of performance results for all models (including selection of hyperparameters for each model)
++++++ confusion_matrix_for_models
++++++ model_predict_details
```

## Authors

* [**Chen Guanyi**](https://a-quei.github.io)
* **Liu Yuqi**

## Publications


If you use the data, please cite the following paper:

```
Todo.
```

## License

**This data is only used for research purpose**. 

todo.

## Acknowledgments

todo.
