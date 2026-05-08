# MLOps Dynamic Fairness Monitoring Pipeline
prototyping real-time automated bias detection through python, a Flask-based API integration, and interpretable interactive dashboard

## Why?
For the majority of ML models, ethical alignment is not statically set from training data but rather remains in dynamic states that degrade over time, slowly shifting data & causing bias to emerge

The ethics of these models must be continuously monitored as much as their performance, which can be done by statistically tracking predefined measurable metrics

#  Key Features
## multi-metric analysis
- statistical parity difference (selection rate gaps between groups)
- equal opportunity difference (true positive rate disparities)
- predictive parity difference (comparison to false positive rates)
- accuracy gaps (performance differentials across demographics)

## controlled bias simulations
three automated bias-injection simulations modeling variation in real-world data distribution using `inject_bias()` function to flip predictions for unprivileged group
- gradual drift: introducing bias gradually
- intermittent (oscillating) drift: introducing bias periodically with differing maximum intensities
- bias-correction: beginning at peak bias and slowly returning to baseline

## seamless integration
generalized variable labeling to accomodate multiple binary classification datasets
*** baseline requirements: labeled headers, target columnm, protected_attribute column

# To Use:
