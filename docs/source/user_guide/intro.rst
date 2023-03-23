Introduction to DoWhy
=====================
*[Maybe also include: "Who is this library/user guide for?"]*

DoWhy helps to perform a wide variety of *causal tasks* based on a method generally referred to as causal inference.
These tasks help to give answers to questions, such as "If I change the color of my button to red, how much will this change users’ purchase decisions (effect)?", or "Which service in my distributed system has caused the frontend to be slower than usual?".

To give readers easy access to learn about such tasks, this user guide dedicates the entire chapter
:doc:`causal_tasks/index` to explain how DoWhy can be used to perform
such tasks. We categorize tasks into :doc:`causal_tasks/estimating_causal_effects/index`,
:doc:`causal_tasks/root_causing_and_explaining/index`, and :doc:`causal_tasks/what_if/index`. See also the :doc:`example_notebooks/nb_index` how to use them in a concrete problem.

To perform tasks, DoWhy leverages two powerful frameworks, namely graphical causal models (GCM) and potential outcomes (PO),
depending on the task at hand. What’s common to most tasks is that they require a *causal graph*, which is modeled after the
problem domain. For that reason, this user guide starts with :doc:`modeling_causal_relations/index`. Readers interested in
:doc:`causal_tasks/root_causing_and_explaining/unit_change` or :doc:`causal_tasks/root_causing_and_explaining/feature_attribution`,
which do not require a causal graph, can skip right to those sections.

To aid the navigation within this user guide and the library further, the following flow chart can be used:

.. image:: navigation.png
   :alt: Visual navigation map to aid the user in navigating the user guide
   :width: 100%
