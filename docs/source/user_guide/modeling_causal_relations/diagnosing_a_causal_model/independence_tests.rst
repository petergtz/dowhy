Independence Tests
===================

Assuming we have the following data:

>>> import numpy as np, pandas as pd
>>>
>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

To test whether :math:`X` is conditionally independent of :math:`Z` given :math:`Y` using the
`kernel dependence measure
<https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf>`_, all you need to
do is:

>>> import dowhy.gcm as gcm
>>>
>>> # Null hypothesis: x is independent of y given z
>>> p_value = gcm.independence_test(X, Z, conditioned_on=Y, method='kernel')
>>> p_value
0.48386151342564865

If we define a threshold of, for instance, 0.05, we would only reject independence if the
p-value falls below this. Note that we can strictly speaking only reject independence, but cannot accept it
if the p-value is above the threshold. However, in practice, one might still consider the result above
as some evidence that the variables are conditionally independent. This is also what we would expect, given that we generated the data using the causal graph :math:`X
\rightarrow Y \rightarrow Z`, where Z is conditionally independent of :math:`X` given :math:`Y`.

To test whether :math:`X` is independent of :math:`Z` (*without* conditioning on :math:`Y`), we can
use the same function without the third argument.

>>> # Null hypothesis: x is independent of y
>>> p_value = gcm.independence_test(X, Z, method='kernel')
>>> p_value
0.0

Again, we can define a threshold of 0.05, but this time the p-value is clearly below this threshold and we can clearly
reject the null hypothesis of independence. This is, :math:`X` and :math:`Z` *are* dependent on each other. Again, this is what we would
expect, since :math:`Z` is dependent on :math:`Y` and :math:`Y` is dependent on :math:`X`, but we
don't condition on :math:`Y`.
