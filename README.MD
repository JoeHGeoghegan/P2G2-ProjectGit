## The Sentimentalists Present
# Is Talking about the Market, Predicting the Market?

### Group Members
* Alex Smith
* Andrew Altman
* Joe Geoghegan
* Krista Xhoxhi
* Lauren Krohn
### Proposal
Our goal is to create an algorithmic trading strategy using signals from both sentiment and returns data.
To do this we are using current/historic stock data as well as sentiment data from several sources.
The sources we are analyzing are Reddit, news sources, as well as written statements in quarterly reports.
We plan to run our model on multiple companies in several sectors including tech, finance, and energy.
Usign stock and sentiment data, we would utilize both random forest and neural net models to predict a multi-class returns category.
Finally, we will use these classifications as our entry and exit signals to our trading strategy. 
## Sectors/Companies we are looking at:
- Technology
  - Netflix (NFLX)
  - Facebook/Meta (FB)
  - Uber (UBER)
  - Microchip Technology (MCHP)
  - AirBnB (ABNB)
- Energy
  - Diamondback Energy Inc. (FANG)
  - Marathon Oil Corp. (MRO)
  - Devon Energy Corp. (DVN)
  - SunPower Corp. (SPWR)
  - Renewable Energy Group Inc. (REGI)
- Finance
  - McKinsey & Company (MTRX)
  - BlackRock (BLK)
  - PayPal (PYPL)
  - Mercadolibre Inc (MELI)
  - SoFi (SOFI)

## Usage
Code for the LightGBM model can be found in light_gbm.ipynb and lstm_solo.ipynb, both found in the Notebooks folder.

PNGs are available for our visualizations, and can be found in P2G2-ProjectGit/Notebooks/Data/Cleaned_Data/Plots/
