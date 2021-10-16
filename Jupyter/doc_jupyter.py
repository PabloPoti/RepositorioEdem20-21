#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install vega_datasets')

get_ipython().system('pip install altair')

# load an example dataset
from vega_datasets import data
cars = data.cars()

# plot the dataset, referencing dataframe column names
import altair as alt
alt.Chart(cars).mark_bar().encode(
  x=alt.X('Miles_per_Gallon', bin=True),
  y='count()',
  color='Origin'
)


# In[ ]:




