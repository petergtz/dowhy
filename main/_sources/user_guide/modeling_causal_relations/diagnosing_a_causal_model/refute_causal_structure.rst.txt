Refute causal structure
=======================

Testing the causal graph against data:

>>> rejection_result, _ = gcm.reject_causal_structure(causal_graph, data)
>>> if rejection_result == gcm.RejectionResult.REJECTED:
>>> print("Do not continue. We advice revising the causal graph.")

TODO: This needs more flesh.
