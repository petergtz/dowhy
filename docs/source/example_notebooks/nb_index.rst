Example notebooks
=================

These examples are also available on `GitHub <https://github
.com/py-why/dowhy/tree/main/docs/source/example_notebooks>`_. You can `run them locally <https://docs.jupyter
.org/en/latest/running.html>`_ after cloning `DoWhy <https://github.com/py-why/dowhy>`_ and `installing Jupyter
<https://jupyter.org/install>`_. Or you can run them directly in a web browser using the
`Binder environment <https://mybinder.org/v2/gh/microsoft/dowhy/main?filepath=docs%2Fsource%2F>`_.

Effect estimation
-----------------

.. card-carousel:: 2

    .. card:: :doc:`dowhy_simple_example`

        .. image:: ../_static/effect-estimation-estimand-expression.png
        +++
        **Level:** Beginner

    .. card:: :doc:`DoWhy-The Causal Story Behind Hotel Booking Cancellations`

        .. image:: ../_static/hotel-bookings.png
        +++
        **Level:** Beginner

    .. card:: :doc:`dowhy_example_effect_of_memberrewards_program`

        .. image:: ../_static/membership-program-graph.png
        +++
        **Level:** Beginner

    .. card:: :doc:`dowhy_optimize_backdoor_example`

        +++
        **Level:** Beginner

    .. card:: :doc:`Causal Inference and its Connections to ML (using EconML)<tutorial-causalinference-machinelearning-using-dowhy-econml>`

        .. image:: ../_static/world.png

        +++
        **Level:** Beginner


Graphical causal models
-----------------------

.. card-carousel:: 2

    .. card:: :doc:`gcm_basic_example`

        .. image:: ../_static/graph-xyz.png
            :width: 50px
            :align: center

        +++
        **Level:** Beginner

    .. card:: :doc:`gcm_rca_microservice_architecture`

        .. image:: ../_static/microservice-architecture.png

        +++
        **Level:** Beginner

    .. card:: :doc:`gcm_401k_analysis`

        +++
        **Level:** Advanced

    .. card:: :doc:`gcm_supply_chain_dist_change`

        .. image:: ../_static/supply-chain.png

        +++
        **Level:** Advanced

    .. card:: :doc:`gcm_counterfactual_medical_dry_eyes`

        +++
        **Level:** Advanced


Others
------

.. card-carousel:: 3

    .. card:: :doc:`gcm_draw_samples`

        **Level:** Beginner

    .. card:: :doc:`gcm_rca_microservice_architecture`

        **Level:** Beginner

    .. card:: :doc:`gcm_rca_microservice_architecture`

        **Level:** Beginner





.. toctree::
   :maxdepth: 1
   :caption: Getting started

   dowhy_simple_example
   dowhy_confounder_example
   dowhy_estimation_methods
   graph_conditional_independence_refuter
   dowhy-simple-iv-example
   load_graph_example
   dowhy_interpreter
   dowhy_causal_discovery_example
   dowhy_causal_api
   do_sampler_demo
   gcm_basic_example
   gcm_draw_samples

.. toctree::
   :maxdepth: 1
   :caption: Using benchmark datasets

   dowhy_ihdp_data_example
   dowhy_twins_example
   dowhy_lalonde_example
   dowhy_refutation_testing
   lalonde_pandas_api

.. Advanced notebooks

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   dowhy-conditional-treatment-effects
   dowhy_mediation_analysis.ipynb
   dowhy_demo_dummy_outcome_refuter.ipynb
   dowhy_multiple_treatments.ipynb
   dowhy_refuter_notebook
   dowhy_causal_discovery_example.ipynb
   identifying_effects_using_id_algorithm.ipynb
   gcm_rca_microservice_architecture
   gcm_supply_chain_dist_change
   gcm_counterfactual_medical_dry_eyes
   gcm_401k_analysis
