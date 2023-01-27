# DoWhy User Guide

# Foreword

*This foreword should include things like:*

* *[The need for causal inference](https://www.pywhy.org/dowhy/main/user_guide/intro.html#the-need-for-causal-inference) (just not focused only on effect estimation)*
* *Some of the [description in the documentation](https://www.pywhy.org/dowhy/main):*

>Much like machine learning libraries have done for prediction, DoWhy is a Python library that aims to spark causal thinking and analysis. DoWhy provides a wide variety of algorithms for effect estimation, causal structure learning, diagnosis of causal structures, root cause analysis, interventions and counterfactuals.

* Some of the [GitHub description](https://github.com/py-why/dowhy):

>DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy is based on a unified language for causal inference, combining causal graphical models and potential outcomes frameworks.

* Some of its [README](https://github.com/py-why/dowhy/blob/main/README.rst):

>As computing systems are more frequently and more actively intervening in societally critical domains such as healthcare, education, and governance, it is critical to correctly predict and understand the causal effects of these interventions. Without an A/B test, conventional machine learning methods, built on pattern recognition and correlational analyses, are insufficient for decision-making.

Much like machine learning libraries have done for prediction, "DoWhy" is a Python library that aims to spark causal thinking and analysis. DoWhy provides a principled four-step interface for causal inference that focuses on explicitly modeling causal assumptions and validating them as much as possible. The key feature of DoWhy is its state-of-the-art refutation API that can automatically test causal assumptions for any estimation method, thus making inference more robust and accessible to non-experts. DoWhy supports estimation of the average causal effect for backdoor, frontdoor, instrumental variable and other identification methods, and estimation of the conditional effect (CATE) through an integration with the EconML library.


*In general, it should create some motivation and excitement for why one would use this library and why it’s needed.*

# Introduction

*[Maybe also include: “Who is this library/user guide for?”]*

DoWhy helps to perform a wide variety of causal *tasks* based on a method generally referred to as causal inference. These tasks help to give answers to questions, such as “If I change the color of my button to red, how much will this change users’ purchase decisions (effect)?”, or “Which service in my distributed system has caused the frontend to be slower than usual?”.

To give readers easy access to learn about such tasks, this user guide dedicates the entire chapter [Performing Causal Tasks](#performing-causal-tasks) to explain how DoWhy can be used to perform such tasks. We categorize tasks into [Estimating Causal Effects](#estimating-causal-effects), [Root-Causing and Explaining Observed Effects](#root-causing-and-explaining-observed-effects), and [Asking and Answering What-If Questions](#asking-and-answering-what-if-questions).

To perform tasks, DoWhy leverages two powerful frameworks, namely graphical causal models (GCM) and potential outcomes (PO). Depending on the task at hand, it leverages GCM, or PO, or both, or none. What’s common to all tasks though, except for unit-change and feature attribution, is that they require a *causal graph*, which is modeled after the problem domain. For that reason, this user guide starts with [Modeling Causal Relations](#modeling-causal-relations). Readers interested in [Unit-Change Attribution](#unit-change-attribution) or [Feature Attribution](#feature-attribution) can skip right to those sections.

To aid the navigation within this user guide and the library further, the following flow chart can be used:

*[TODO (pego): insert flow chart here, which is a* *visual navigation map for the library.]*
* * *

# Modeling Causal Relations

Except for a couple of exceptions, in DoWhy, the first step to perform a causal *task* is to model causal relations in form of a causal graph. A causal graph models the causal relations, or “cause-effect-relationships” present in a system or problem domain. This serves to make each causal assumption explicit. Think e.g. altitude → temperature, i.e. higher altitude causes lower temperature. A causal graph is a directed acyclic graph (DAG) where an edge X→Y implies that X causes Y. Statistically, a causal graph encodes the conditional independence relations between variables.

In cases, where we do not know the causal graph, we can apply methods for learning causal structures from data. Section [Learning Causal Structure](#learning-causal-structure) introduces statistical methods for this.

There are other ways to construct causal graphs too. Often, we can derive it, when systems already convey this kind of information. E.g. in a distributed system of microservices, we can use request tracing frameworks to reconstruct the graph of dependencies in that system. A causal graph is the reverse of that dependency graph.

*[TODO: add image here for microservices]*

In other cases, we can consult domain experts to learn about the causal graph and construct it.

*[TODO: add image for hotel booking cancellations]*

Note that, depending on the causal task, this graph need not be complete. E.g. for [Effect Estimation Using specific Effect Estimators (for ACE, mediation effect, ...)](#effect-estimation-using-specific-effect-estimators-for-ace-mediation-effect) you can provide a partial graph, representing prior knowledge about some of the variables. DoWhy automatically considers the rest of the variables as potential confounders.

In DoWhy, we can use the [NetworkX](https://networkx.github.io/) library to create causal graphs. In the snippet below, we create a chain X→Y→Z:

```
import networkx as nx
causal_graph = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])
```

Once we have the causal graph, the next steps are defined by what we want to do:

* For effect estimation using specific effect estimators, this is all we need. The next step would be “identification” as explained in [Effect Estimation Using specific Effect Estimators (for ACE, mediation effect, ...)](#effect-estimation-using-specific-effect-estimators-for-ace-mediation-effect).
* For most other tasks, we will also have to assign so-called causal mechanisms to each node as we’ll show in the following section.

To diagnose a causal graph, check out [Diagnosing a Causal Model](#diagnosing-a-causal-model).

## Modeling cause-effect relationships with causal mechanisms

To perform causal tasks based on graphical causal models, such as root cause analysis or what-if analysis, we also have to know the nature of underlying data-generating process of variables. A causal graph by itself, being a diagram, does not have any information about the data-generating process. To introduce this data-generating process, we use an SCM that’s built on top of our causal graph:

```
>>> from dowhy import gcm
>>> causal_model = gcm.StructuralCausalModel(causal_graph)
```

At this point we would normally load our dataset. For this introduction, we generate some synthetic data instead. The API takes data in form of Pandas DataFrames:

```
>>> import numpy as np, pandas as pd
```

```
>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
>>> data.head()
 *X         Y          Z*
*0 -2.253500 -3.638579 -10.370047*
*1 -1.078337 -2.114581  -6.028030*
*2 -0.962719 -2.157896  -5.750563*
*3 -0.300316 -0.440721  -2.619954*
*4  0.127419  0.158185   1.555927*
```

Note how the columns X, Y, Z correspond to our nodes X, Y, Z in the graph constructed above. We can also see how the values of X influence the values of Y and how the values of Y influence the values of Z in that data set.
The causal model created above allows us now to assign causal mechanisms to each node in the form of functional causal models. Here, these mechanism can either be assigned manually if, for instance, prior knowledge about certain causal relationships are known or they can be assigned automatically using the [`auto`](https://www.pywhy.org/dowhy/main/dowhy.gcm.html#module-dowhy.gcm.auto) module. For the latter, we simply call:

```
>>> gcm.auto.assign_causal_mechanisms(causal_model, data)
```

In case we want to have more control over the assigned mechanisms, we can do this manually as well. For instance, we can can assign an empirical distribution to the root node X and linear additive noise models to nodes Y and Z:

```
>>> causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
```

Section [Customizing Causal Mechanism Assignment](https://www.pywhy.org/dowhy/main/user_guide/gcm_based_inference/customizing_model_assignment.html) will go into more detail on how one can even define a completely customized model or add their own implementation.
In the real world, the data comes as an opaque stream of values, where we typically don’t know how one variable influences another. The graphical causal models can help us to deconstruct these causal relationships again, even though we didn’t know them before.

## Diagnosing a Causal Model

When we modeled our problem domain as a causal model, or causal graph, a natural question that comes up: Is our causal model correct?

To answer this question, there are a number of statistical methods to verify this, which we’ll cover in the following sub-sections.

### Conditional Independence Tests

*[Pure copy of existing documentation]*

Assuming we have the following data:


```
>>> import numpy as np, pandas as pd
*>>>*
>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
```

To test whether X is conditionally independent of Z given Y using the [kernel dependence measure](https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf), all you need to do is:


```
>>> import dowhy.gcm as gcm
>>>
>>> *# Null hypothesis: x is independent of y given z*
>>> p_value = gcm.independence_test(X, Z, conditioned_on=Y)
>>> p_value
*0.48386151342564865*
```

If we define a threshold of 0.05 (as is often done as a good default), and the p-value is clearly above this, it says X and Z are indeed independent when we condition on Y. This is what we would expect, given that we generated the data using the causal graph X→Y→Z, where Z is conditionally independent of X given Y.
To test whether X is independent of Z (*without* conditioning on Y), we can use the same function without the third argument.


```
>>> *# Null hypothesis: x is independent of y*
>>> p_value = gcm.independence_test(X, Z)
>>> p_value
*0.0*
```

Again, we can define a threshold of 0.05, but this time the p-value is clearly below this threshold. This says X and Z *are* dependent on each other. Again, this is what we would expect, since Z is dependent on Y and Y is dependent on X, but we don’t condition on Y.

### Refute causal structure

*[Pure copy of existing documentation]*

Testing the causal graph against data:

```
>>> rejection_result, _ = gcm.reject_causal_structure(causal_graph, data)
>>> if rejection_result == gcm.RejectionResult.REJECTED:
>>>  print("Do not continue. We advice revising the causal graph.")
```

*[TODO (kaibud or bloebp): this chapter should provide more content]*

### Direct arrow strength

*[Pure copy of existing documentation]*

By quantifying the strength of an arrow, we answer the question:

>How strong is the causal influence from a cause to its direct effect?

#### How to use it

To see how the method works, let us generate some data.

```
>>> import numpy as np, pandas as pd, networkx as nx
>>> from dowhy import gcm
>>> np.random.seed(10)  *# to reproduce these results*
```

```
>>> Z = np.random.normal(loc=0, scale=1, size=1000)
>>> X = 2*Z + np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 3*X + 4*Z + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(dict(X=X, Y=Y, Z=Z))
```

Next, we will model cause-effect relationships as a probabilistic causal model and fit it to the data.

```
>>> causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('Z', 'Y'), ('Z', 'X'), ('X', 'Y')]))
>>> causal_model.set_causal_mechanism('Z', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('X', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> gcm.fit(causal_model, data)
```

Finally, we can estimate the strength of incoming arrows to a node of interest (e.g., Y).

```
>>> strength = gcm.arrow_strength(causal_model, 'Y')
>>> strength
 {('X', 'Y'): 41.321925893102716,
 ('Z', 'Y'): 14.736197949517237}
```

Interpreting the results: By default, the measurement unit of the scalar values for arrow strengths is variance for a continuous real-valued target, and the number of bits for a categorical target. Above, we observe that the direct influence from X to Y (~41.32) is stronger (by ~2.7 times) than the direct influence from Z to Y (~14.73). Roughly speaking, “removing” the arrow from X to Y increases the variance of Y by ~41.32 units, whereas removing Z→Y increases the variance of Y by ~14.73 units.
In the next section, we explain what “removing” an edge implies. In particular, we briefly explain the science behind our method for quantifying the strength of an arrow.

#### Understanding the method

We will use the causal graph below to illustrate the key idea behind our method.
Recall that we can obtain the joint distribution of variables P from their causal graph via a product of conditional distributions of each variable given its parents. To quantify the strength of an arrow from Z to Y, we define a new joint distribution PZ→Y, also called post-cutting distribution, obtained by removing the edge Z→Y, and then feeding Y with an i.i.d. copy of Z. The i.i.d. copy can be simulated, in practice, by applying random permutation to samples of Z. The strength of an arrow from Z to Y, denoted CZ→Y, is then the distance (e.g., KL divergence) between the post-cutting distribution PZ→Y and the original joint distribution P.
CZ→Y:=DKL(P||PZ→Y)
Note that only the causal mechanism of the target variable (PY∣X,Z in the above example) changes between the original and the post-cutting joint distribution. Therefore, any change in the marginal distribution of the target (obtained by marginalising the joint distribution) is due to the change in the causal mechanism of the target variable. This means, we can also quantify the arrow strength in terms of the change in the property of the marginal distribution (e.g. mean, variance) of the target when we remove an edge.

#### Measuring arrow strengths in different units

By default, arrow_strength employs KL divergence for measuring the arrow strength for categorical target, and difference in variance for continuous real-valued target. But we can also plug in our choice of measure, using the `difference_estimation_func` parameter. To measure the arrow strength in terms of the change in the mean, we could define:

```
>>> def mean_diff(Y_old, Y_new): return np.mean(Y_new) - np.mean(Y_old)
```

and then estimate the arrow strength:

```
>>> gcm.arrow_strength(causal_model, 'Y', difference_estimation_func=mean_diff)
 {('X', 'Y'): 0.11898914602350251,
 ('Z', 'Y'): 0.07811542095834415}
```

This is expected; in our example, the mean value of Y remains 0 regardless of whether we remove an incoming arrow. As such, the strength of incoming arrows to Y should be negligible.
In summary, arrow strength can be measured in different units (e.g., mean, variance, bits). Therefore, we advise users to pick a meaningful unit—based on data and interpretation—to apply this method in practice.

### Refuting Invertible Model (GCM only)

## Learning Causal Structure

Learning the causal structure, or causal graph that is, is only necessary in case we have no other means to get hold of the causal graph. If you already have the causal graph in your specific problem, you can skip this chapter and move on to [Performing Causal Tasks](#performing-causal-tasks).

### IID

TBD

### Time Series

TBD
* * *

# Performing Causal Tasks

*[TODO: Every task chapter should make it clear that it relies on the causal graph constructed in* [Modeling Causal Relations](#modeling-causal-relations)*.]*

## Estimating Causal Effects

### Effect Estimation Using specific Effect Estimators (for ACE, mediation effect, ...)

*[Combine https://github.com/py-why/dowhy#sample-causal-inference-analysis-in-dowhy and https://www.pywhy.org/dowhy/main/example_notebooks/dowhy_simple_example.html for this.*

*Maybe also use the new functional API here already.*

*https://github.com/py-why/dowhy/wiki/API-proposal-for-v1#effect-inference-api
(include refutation step here)]*

### Effect Estimation using GCM

One of the most common causal questions is how much does a certain target quantity differ under two different interventions/treatments. This is also known as average treatment effect (ATE) or, more generally, average causal effect (ACE). The simplest form is the comparison of two treatments, i.e. what is the difference of my target quantity on average given treatment A vs treatment B. For instance, do patients treated with a certain medicine (T:=1) recover faster than patients who were not treated at all (T:=0). The ACE API allows to estimate such differences in a target node, i.e. it estimates the quantity E[Y|do(T:=A)]−E[Y|do(T:=B)]

#### How to use it

Let’s generate some data with an obvious impact of a treatment.


```
>>> import networkx as nx, numpy as np, pandas as pd
>>> X0 = np.random.normal(0, 0.2, 1000)
>>> T = (X0 > 0).astype(float)
>>> X1 = np.random.normal(0, 0.2, 1000) + 1.5 * T
>>> Y = X1 + np.random.normal(0, 0.1, 1000)
>>> data = pd.DataFrame(dict(T=T, X0=X0, X1=X1, Y=Y))

```

Here, we see that T is binary and adds 1.5 to Y if it is 1 and 0 otherwise. As usual, lets model the cause-effect relationships and fit it on the data:


```
>>> causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('X0', 'T'), ('T', 'X1'), ('X1', 'Y')]))
>>> gcm.auto.assign_causal_mechanisms(causal_model, data)
>>> gcm.fit(causal_model, data)

```

Now we are ready to answer the question: “What is the causal effect of setting T:=1 vs T:=0?”


```
>>> gcm.average_causal_effect(causal_model,
>>>                         'Y',
>>>                         interventions_alternative={'T': lambda x: 1},
>>>                         interventions_reference={'T': lambda x: 0},
>>>                         num_samples_to_draw=1000)
1.4815661299638665

```

The average effect is ~1.5, which coincides with our data generation process. Since the method expects an dictionary with interventions, we can also intervene on multiple nodes and/or specify more complex interventions.

Note that although it seems difficult to correctly specify the causal graph in practice, it often suffices to specify a graph with the correct causal order. This is, as long as there are no anticausal relationships, adding too many edges from upstream nodes to a downstream node would still provide reasonable results when estimating causal effects. In the example above, we get the same result if we add the edge X0→Y and T→Y:


```
>>> causal_model.graph.add_edge('X0', 'Y')
>>> causal_model.graph.add_edge('T', 'Y')
>>> gcm.auto.assign_causal_mechanisms(causal_model, data, override_models=True)
>>> gcm.fit(causal_model, data)
>>> gcm.average_causal_effect(causal_model,
>>>                           'Y',
>>>                           interventions_alternative={'T': lambda x: 1},
>>>                           interventions_reference={'T': lambda x: 0},
>>>                           num_samples_to_draw=1000)
1.4841671230842537

```



## Understanding the method

Estimating the average causal effect is straightforward seeing that this only requires to compare the two expectations of a target node based on samples from their respective interventional distribution. This is, we can boil down the ACE estimation to the following steps:

1. Draw samples from the interventional distribution of Y under treatment A.
2. Draw samples from the interventional distribution of Y under treatment B.
3. Compute their respective means.
4. Take the differences of the means. This is, E[Y|do(T:=A)]−E[Y|do(T:=B)], where we do not need to restrict the type of interventions or variables we want to intervene on.



https://github.com/py-why/dowhy/wiki/API-proposal-for-v1#gcm-based-alternative-fitting-the-graph-and-then-answering-multiple-causal-questions

## Root-Causing and Explaining Observed Effects

This chapter is about explaining observed effects, which can also be used to perform root-cause analysis. The different focus on explaining different kinds of effects: single outliers, permanent changes, intrinsic contributions, etc.

### Outliers

### Distribution Changes

*[Pure copy of exiting documentation]*

When attributing distribution changes, we answer the question:

>What mechanism in my system changed between two sets of data?

For example, in a distributed computing system, we want to know why an important system metric changed in a negative way.

#### How to use it

To see how the method works, let’s take the example from above and assume we have a system of three services X, Y, Z, producing latency numbers. The first dataset `data_old` is before the deployment, `data_new` is after the deployment:


```
>>> import networkx as nx, numpy as np, pandas as pd
>>> from dowhy import gcm
>>> from scipy.stats import halfnorm
```



```
>>> X = halfnorm.rvs(size=1000, loc=0.5, scale=0.2)
>>> Y = halfnorm.rvs(size=1000, loc=1.0, scale=0.2)
>>> Z = np.maximum(X, Y) + np.random.normal(loc=0, scale=1, size=1000)
>>> data_old = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
```



```
>>> X = halfnorm.rvs(size=1000, loc=0.5, scale=0.2)
>>> Y = halfnorm.rvs(size=1000, loc=1.0, scale=0.2)
>>> Z = X + Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data_new = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
```

The change here simulates an accidental conversion of multi-threaded code into sequential one (waiting for X and Y in parallel vs. waiting for them sequentially).
Next, we’ll model cause-effect relationships as a probabilistic causal model:


```
>>> causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('X', 'Z'), ('Y', 'Z')]))  *# X -> Z <- Y*
>>> causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Y', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
```

Finally, we attribute changes in distributions to changes in causal mechanisms:


```
>>> attributions = gcm.distribution_change(causal_model, data_old, data_new, 'Z')
>>> attributions
{'X': -0.0066425020480165905, 'Y': 0.009816959724738061, 'Z': 0.21957816956354193}
```

As we can see, Z got the highest attribution score here, which matches what we would expect, given that we changed the mechanism for variable Z in our data generation.

### Intrinsic Causal Contribution

### Unit-Change Attribution

(does not require a causal graph)

### Feature Attribution

(does not require a causal graph)

## Asking and Answering What-If Questions

### Interventions

#### Atomic intervention

#### Soft intervention

### Counterfactuals (unit-level)

## Building [Causal ML Prediction](https://github.com/py-why/dowhy/wiki/API-Proposal-for-Causal-Prediction-(and-optionally-representation-learning)) model

(want to build an ML model that is causally robust)
* * *
* * *
