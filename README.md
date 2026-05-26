# Dermatology AI Fairness Optimization

This is my research project focused on measuring and fixing performance gaps in skin condition AI models across different demographic groups. 

The project uses the dataset from the 2022 Stanford DDI paper (*"Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set"*) as a baseline. From there, I built an automated pipeline to see if we can use smart hyperparameter tuning and optimization to actively close the accuracy gap on darker skin tones.

## What I Actually Built & Explored
Instead of just running the models as-is, I wanted to see if altering the training process could make the outcomes more equitable. 

Here is what I engineered for this repo:
* **`fairness_search_pipeline.py`**: The orchestration pipeline I engineered to automate the evaluation loops and manage model configurations.
* **`bayes_opt_efficiency.py` & `bayes_opt_high_improvement.py`**: My implementation of Bayesian optimization to systematically hunt for training parameters that balance overall accuracy with demographic fairness.
* **`evaluate_fairness_paired.py`**: The statistical evaluation framework I wrote to run paired testing and prove the fairness improvements were not just random noise.

## What I Learned / Results
* I found that optimizing specifically for minority class fairness helped close the accuracy gap by about 97.7%, though it required a slight trade-off in overall processing speed.
* All of my test runs and statistical summaries are saved under the `results/` directory.

## How to Run the Code
If you want to replicate my optimization experiments, make sure you have the DDI dataset downloaded via AzCopy, and then run:
```bash
python3 fairness_search_pipeline.py
```
