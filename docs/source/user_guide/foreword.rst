Foreword
=====================

*This foreword should include things like:*

* *[The need for causal inference](https://www.pywhy.org/dowhy/main/user_guide/intro.html#the-need-for-causal-inference) (just not focused only on effect estimation)*

The need for causal inference
----------------------------------

Causal inference is essential for informed decision-making, as it uncovers causality beyond mere associations found in predictive models. It enables us to estimate effects of interventions and counterfactual outcomes, even in the absence of data. Moving beyond correlation-based analysis is vital for generalizing insights and gaining a true understanding of real-world relationships, such as:

* Will it work?
    * Does a proposed change to a system improve people's outcomes?
* Why did it work?
    * What led to a change in a system's outcome?
* What should we do?
    * What changes to a system are likely to improve outcomes for people?
* What are the overall effects?
    * How does the system interact with human behavior?
    * What is the effect of a system's recommendations on people's activity?

Answering these questions requires causal reasoning. While many methods exist
for causal inference, it is hard to compare their assumptions and robustness of results. DoWhy makes three contributions,

1. Provides a systematic method for modeling causal relationships through graphical representations, ensuring that all underlying assumptions are clearly stated and transparent.
2. Provides a unified interface for many popular causal inference methods, combining the two major frameworks of graphical models and potential outcomes.
3. Provides capabilities to test the validity of assumptions if possible and assesses the robustness of the estimate to violations.

To see DoWhy in action, check out how it can be applied to estimate the effect
of a subscription or rewards program for customers [`Rewards notebook
<https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_example_effect_of_memberrewards_program.ipynb>`_] and for implementing and evaluating causal inference methods on benchmark datasets like the `Infant Health and Development Program (IHDP) <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_ihdp_data_example.ipynb>`_ dataset, `Infant Mortality (Twins) <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_twins_example.ipynb>`_ dataset, and the `Lalonde Jobs <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_lalonde_example.ipynb>`_ dataset.




* *Some of the [description in the documentation](https://www.pywhy.org/dowhy/main):*

>Much like machine learning libraries have done for prediction, DoWhy is a Python library that aims to spark causal thinking and analysis. DoWhy provides a wide variety of algorithms for effect estimation, quantification of causal influences, causal structure learning, diagnosis of causal structures, root cause analysis, interventions and counterfactuals.
* Some of the [GitHub description](https://github.com/py-why/dowhy):

>DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy is based on a unified language for causal inference, combining causal graphical models and potential outcomes frameworks.
* Some of its [README](https://github.com/py-why/dowhy/blob/main/README.rst):

>As computing systems are more frequently and more actively intervening in societally critical domains such as healthcare, education, and governance, it is critical to correctly predict and understand the causal effects of these interventions. Without an A/B test, conventional machine learning methods, built on pattern recognition and correlational analyses, are insufficient for decision-making.
The key feature of DoWhy is its state-of-the-art refutation API that can test causal assumptions for any estimation method, thus making inference more robust and accessible to non-experts. DoWhy supports estimation of the average causal effect for backdoor, frontdoor, instrumental variable and other identification methods, and estimation of the conditional effect (CATE) through an integration with the EconML library. Additionally, DoWhy supports answering causal questions beyond effect estimation by utilizing graphical causal models, which enable tackling problems such as root cause analysis or quantification of causal influences.


*In general, it should create some motivation and excitement for why one would use this library and why it’s needed.*
