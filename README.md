# Crime and Communities
The crime and communities dataset contains crime data from communities in the United States. The data
combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS
survey, and crime data from the 1995 FBI UCR. More details can be found at https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized.

The dataset contains 125 columns total; p = 124 predictors and 1 target (ViolentCrimesPerPop). There
are n = 1994 observations. These can be arranged into an n × p = 1994 × 127 feature matrix **X**, and an
n × 1 = 1994 × 1 response vector **y** (containing the observations of ViolentCrimesPerPop).

In the first half of the project, we explore our dataset and answer key questions such as:
* Which variables are categorical versus numerical?
* What are the general summary statistics of the data? How can these be visualized?
* Is the data normalized? Should it be normalized?
* Are there missing values in the data? How should these missing values be handled?
* Can the data be well-represented in fewer dimensions?

The primary goal is to develop a model to predict **y**, ViolentCrimesPerPop, using the 124 features (or some subset of them). We train and test several different methods (e.g. PCR, PLSR, Ridge, Lasso) and use appropriate model selection techniques (e.g. cross validation) to determine our desired model. 
