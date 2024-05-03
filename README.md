<a name="br1"></a> 

LUISS ’Guido Carli’

MSc in Data Science and Management - Machine Learning Course

May 3, 2024

AI Systematic Trading Challenge

Comparative Analysis of RNN and ANFIS for Stock Price Prediction

Project Technical Report

Coci, Marco, Team Leader

ID: 786471, email: m.coci@studenti.luiss.it

Dong, Thi Kieu Trang

ID: 772701, email: thi.dong@studenti.luiss.it

Navarra, Filippo

ID: 782571, email: ﬁlippo.navarra@studenti.luiss.it

Collaborative Business Case Proposed by Euklid



<a name="br2"></a> 

Coci, Dong, Navarra - Machine Learning project

1 Introduction

• Open, High, Low Prices (OHL): These variables

represent the opening, highest, lowest, and closing

prices within each week.

In the rapidly evolving ﬁeld of ﬁnancial markets, lever-

aging advanced machine learning techniques to predict

market trends has become critical to gain a competitive

advantage. This project, undertaken as part of the AI

Systematic Trading Challenge, aims to develop a robust

AI-based trading model that can eﬀectively address the

complexities of global ﬁnancial markets.

• Volume: This metric denotes the total number of

shares or contracts traded during the week for a par-

ticular stock or index, reﬂecting market activity and

liquidity.

• Adjusted Close Price: Adjusted for corporate ac-

tions like dividends, stock splits, and rights oﬀerings,

this measure oﬀers a more accurate reﬂection of the

stock’s or index’s value over time.

The company provided us with six key data sets for

our analysis, focusing on three main stocks: Amazon, IBM

and Microsoft; and three signiﬁcant market indexes: Nas-

daq, CAC 40 and SP500. This diverse selection allows us

to explore both individual stock performance and broader

market trends, providing a comprehensive view of the dy-

namics at play.

• Date/Time Stamp: Each record is associated with

a speciﬁc week, facilitating the temporal analysis of

ﬁnancial metrics.

To augment our model’s predictive capabilities, we also in-

corporated some common technical indicators:

Our approach involves two sophisticated predictive

models: Recurrent Neural Networks (RNN) and Adap-

tive Neuro-Fuzzy Inference Systems (ANFIS). RNNs were

chosen for their expertise in handling sequential time se-

ries data, making them ideal for analyzing the temporal

patterns inherent in stock prices. Meanwhile, the ANFIS

model was selected for its ability to integrate fuzzy logic

with neural networks, oﬀering a powerful tool for manag-

ing the uncertainties and nonlinear relationships typical of

ﬁnancial datasets (Barak, Dahooie, and Tichy´ 2015; Esfa-

hanipour and Mardani 2011).

• SMA (Simple Moving Average): A Simple Mov-

ing Average (SMA) is an arithmetic moving average

calculated by adding recent closing prices and then

dividing that by the number of time periods in the

calculation average.

• EMA (Exponential Moving Average): An Ex-

ponential Moving Average (EMA) is a type of moving

average that places a greater weight and signiﬁcance

on the most recent data points. It’s more responsive

to new information compared to the simple moving

average.

A key innovation in our methodology is the develop-

ment and implementation of a custom loss function. Tradi-

tional regression metrics such as Root Mean Square Error

(RMSE) are often focused on the magnitude rather than

the direction of price movements, which is a crucial as-

pect of trading strategies (Prabakaran et al. 2021; Felder

and Mayer 2022). Our custom loss function is designed to

overcome this limitation by emphasizing the accuracy of

directional predictions, thus aligning more closely with real

world trading needs.

• Stochastic Oscillator (STOCH): The Stochastic

Oscillator is a momentum indicator comparing a par-

ticular closing price of a security to a range of its

prices over a certain period of time. The sensitiv-

ity of the oscillator to market movements is reducible

by adjusting that time period or by taking a moving

average of the result.

• RSI (Relative Strength Index): The Relative

Strength Index (RSI) is a momentum oscillator that

measures the speed and change of price movements.

RSI oscillates between zero and 100.

This project not only tests these models according to

standard RMSE criteria but also compares their perfor-

mance using our innovative loss function. Through this

comparative analysis, we aim to validate the eﬀectiveness

of our customized approach.

• MACD (Moving Average Convergence Diver-

gence): The Moving Average Convergence Diver-

gence (MACD) is a trend-following momentum indi-

cator that shows the relationship between two moving

averages of a security’s price.

2 Methods

2\.2 Target Construction

2\.1 Covariates Construction

To eﬀectively model and forecast ﬁnancial outcomes, we fo-

In the development of our systematic trading model, we cused on forecasting stock returns and used it as a target

worked on datasets organized weekly, where variables were variable for our models. Logarithmic returns were used for

crucial for conducting temporal and ﬁnancial analysis that their statistical properties, such as quasi-normal distribu-

are:

tion and constant volatility, which simplify calculations and

2



<a name="br3"></a> 

Coci, Dong, Navarra - Machine Learning project

improve model stationarity (Martucci 2024). The formula

used to calculate logarithmic returns is:

Start

Price<sub>t</sub>

Data Cleaning

Log Return = log(

)

(1)

Price<sub>t−1</sub>

Pick techni-

cal indicators

Where P<sub>t</sub> represent the current price and P<sub>t−1</sub> the previous

one.

ANFIS

with RMSE

RNN with RMSE

2\.3 Process Description

Our methodological approach is summarized in a sequential

process ﬂow:

• Data Cleaning: We began by addressing missing val-

ues and anomalies in our data to ensure accuracy and

reliability in our predictions, we had 2 missing values

for OHL variables just for index datasets.

yes

Is RMSE good enough?

RMSE Results

• Technical Indicator Selection: Choosing the right in-

dicators based on their predictive power and relevance

to our analysis goals.

no

Custom Loss

Function

• Model Application:

RNN with Custom

Loss Function

ANFIS with Cus-

tom Loss Function

– RMSE as loss function: Initially, we applied

RNN and ANFIS using the Root Mean Square

Error (RMSE) loss function to predict price

movements.

– Evaluation of RMSE: Assessing whether RMSE

alone could suﬃciently capture the accuracy

needed for our predictive models.

Custom Loss Function Results

Stop

– Custom Loss Function Implementation: Upon

ﬁnding RMSE inadequate for capturing direc-

tional accuracy, we introduced a custom loss

function aimed at enhancing predictions by

penalizing incorrect directional forecasts more

severely.

2\.4 A new loss function

We realized that the RMSE results were not satisfactory,

especially in our case of ﬁnancial forecasts where the direc-

tion of changes is as important as their magnitude. Thus,

– RNN and ANFIS with Custom Loss Function: we decided to create our custom loss function to try to

Both models were then tested under the new loss improve the results of our model. One of the goals of our

function to compare performance improvements. project is to demonstrate how this new loss function and

3



<a name="br4"></a> 

Coci, Dong, Navarra - Machine Learning project

RMSE behave diﬀerently and to show that in some cases, learning capability of neural networks. This allows it to

our loss function performs better than RMSE.

handle uncertainty and model complex, nonlinear relation-

Our custom loss function imposes a heavier penalty for ships that are typical in ﬁnancial data. Fuzzy logic, unlike

incorrect predictions about the direction of price changes, traditional binary systems, allows for a smoother transition

not just the size of the error. This means that if the model between output states.

predicts an increase when the price actually decreases, or

vice versa, the error is considered more signiﬁcant. The clude:

loss for each prediction takes into account both the size

of the error and whether the prediction correctly captured

the direction of change.

The essential components of ANFIS architecture in-

• Input Layer: The layer that handles the input data

that the model receives. The input data is shaped

and prepared for processing in subsequent layers.

• Fuzzy Membership Layer: This layer essentially

fuzziﬁes the input data by calculating how much each

input belongs to diﬀerent fuzzy sets deﬁned by the

Gaussian parameters.

2\.5 Models

As previously mentioned, we decided to apply two models,

Recurrent Neural Network (RNN) and Adaptive Neuro-

Fuzzy Inference Systems (ANFIS).

• Fuzzy Rule Layer (T-Norm Layer): This layer

models the conjunctions in the fuzzy rules, aggregat-

ing the degrees to which inputs satisfy fuzzy condi-

tions.

2\.5.1 RNNs

RNNs are particularly suited for sequential data, process-

ing time series data and remembering past events, which is

a crucial element in the dynamic and interconnected world

of ﬁnancial markets. This makes them highly valuable for

modeling sequences that demonstrate time-based patterns,

such as ﬁnancial market trends, where past prices can in-

• Output Layer: This layer performs the defuzziﬁca-

tion process, turning the fuzzy quantities into a single

crisp output, which is typical in a regression scenario.

We trained our models with Adam optimizer with L2

regularization (weight decay=1e-5). Learning rate parame-

ter has diﬀerent values for each dataset and models. Learn-

ﬂuence future ones.

The essential components of RNN architecture in- ing rate scheduler parameters, Step size and Decay rate

clude:

were set with following values; 10 and 0.95. Learning rate

values and other hyper-parameters are listed in Table 2.

• Input Layer: Responsible for receiving sequential

data, where each element may represent a word or

character in tasks like natural language processing. 3 Experimental Designs

• Hidden Layer: Maintaining an internal state that

3\.1 Main purpose

evolves as the network processes each element in the

sequence, capturing information from previous time

steps.

The main objective of our experiments was to evaluate the

predictive capabilities of recurrent neural networks (RNNs)

• Recurrent Gate: A crucial feature involving and adaptive neuro-fuzzy inference systems (ANFIS) in

predicting stock price movements. Speciﬁcally, we aimed

weights and connections looping back to the hidden

layer from the previous time step, allowing the RNN

to update its hidden state and remember past infor-

mation.

to evaluate how well these models could predict the direc-

tion of price changes by integrating a custom loss function

designed to prioritize the accuracy of directional predic-

tions.

• Output Layer: Produces predictions based on the

information in the hidden state.

3\.2 Experimental Setup

Speciﬁcally, we trained our model with the Adam opti-

mizer with other hyper-parameters which are listed in Table

1\.

We conducted our experiments using historical stock price

data from six major stocks and indixes. Each dataset was

preprocessed to normalize the data and to calculate rele-

vant ﬁnancial indicators like moving averages and volatility

indices. We split the data into training, validation, and

2\.5.2 ANFIS

For what it concerns ANFIS, it is recognized as a highly ef- testing sets, respectively 70%,20% and 10%, to ensure ro-

fective model for ﬁnancial predictions, particularly because bust model evaluation.

it combines the interpretability of fuzzy systems with the

4



<a name="br5"></a> 

Coci, Dong, Navarra - Machine Learning project

Figure 1: RNNs architecture

3\.3 Custom loss function implementation

• Recall: The ratio of correct positive predictions to

the actual positives (both true positives and false neg-

atives).

Our custom loss function was applied to both models to

compare its eﬀectiveness against traditional loss functions

like RMSE. This function imposes heavier penalties for in-

correct directional predictions, aligning more closely with

real-world trading objectives where the direction of move-

ment is more critical than the magnitude.

true positives

recall =

(4)

true positives + false negatives

• F1: The harmonic mean of precision and recall, a

measure that balances both metrics.

3\.4 Evaluation

precision · recall

F1 = 2 ·

(5)

precision + recall

Results were quantitatively analyzed using accuracy, pre-

cision, recall, and F1 scores, and qualitatively discussed to

draw conclusions about each model’s performance under

diﬀerent experimental conditions. Comparisons were made

4 Results

not only between the models but also between the diﬀerent Our analysis yielded insightful comparisons between the

loss functions used.

Recurrent Neural Network (RNN) and the Adaptive Neuro-

Fuzzy Inference System (ANFIS) across diﬀerent datasets

• Accuracy: The proportion of total correct predic- and loss functions. The observed diﬀerences in model per-

formance are crucial for understanding their applications

in ﬁnancial forecasting and trading strategies.

tions (both true positives and true negatives) out of

all predictions made.

ANFIS demonstrated consistently higher performance

across most metrics when using the Custom loss function,

particularly indexes. This could be attributed to ANFIS’

ability to model complex, non-linear relationships inherent

in ﬁnancial data more eﬀectively than RNNs.

RNNs, while generally underperforming in comparison

to ANFIS, showed more balanced performance across all

datasets when evaluated with the custom loss function.

The Recurrent Neural Network (RNN) exhibited superior

performance on the Amazon dataset, eﬀectively captur-

ing the underlying price trends. In contrast, the Adaptive

Neuro-Fuzzy Inference System (ANFIS) was less eﬀective in

true positives + true negatives

accuracy =

(2)

TP + TN + FP + FN

• Precision: The ratio of correct positive predictions

(true positives) to the total predicted positives (both

true positives and false positives).

true positives

precision =

(3)

true positives + false positives

Table 1: RNN Architecture for both loss functions

Hyperparameters

Number of Units per Layer (Hidden Size)

RNN with RMSE RNN Custom loss function

20

1

20

1

Number of Layers

Learning Rate

Number of Epochs

Sequence Length

Activation Function

Batch Size

0\.001

100

10

0\.001

100

10

tanh

32

tanh

32

Loss Penalty

−

5000

5



<a name="br6"></a> 

Coci, Dong, Navarra - Machine Learning project

Table 2: Comparison of ANFIS RMSE and Loss Metrics for Hyperparameters Across Datasets

Amazon

CAC

IBM

Microsoft

Nasdaq

SP500

Hyperparameter

RMSE Loss RMSE Loss RMSE Loss RMSE Loss RMSE Loss RMSE Loss

N° M.F.<sup>1</sup>

10

Input Dimensions 15

10

15

10

15

10

15

10

15

10

15

10

15

10

15

10

15

10

15

10

15

10

15

M.F. Parameters µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ µ, σ

Batch Size 32 32 32 32 32 32 32 32 32 32 32 32

Learning Rate 0.007 0.007 0.008 0.008 0.005 0.01 0.005 0.008 0.005 0.005 0.003 0.005

Loss Penalty 5000 10000 5000 10000 10000 7000

−

−

−

−

−

this instance. Notably, internal factors speciﬁc to Amazon, zon, and the complexity of the relationships within the

such as the stock split in 2022, likely disrupted ANFIS’s data.

ability to fully leverage its modeling capabilities, hindering The analysis suggests that while RNNs can be optimized

its performance on this particular dataset.

for speciﬁc datasets where sequences of historical data

This indicates that RNNs may still be valuable in scenar- strongly predict future trends and depend on each other,

ios where model interpretability and response to temporal ANFIS provides a more consistently reliable model for ﬁ-

dynamics relations are prioritized.

nancial indices due to its ability to manage stability and

predictability eﬀectively. This comparative vision should

ANFIS adapted well to our custom loss function, par- guide the selection and application of appropriate models

ticularly in markets characterized less volatile (indexes) or based on the speciﬁc characteristics and volatility of the

non-linear behaviour.

This loss function helped RNNs improve in speciﬁc datasets

like Amazon and Microsoft, suggesting that RNNs can be

ﬁnancial data analyzed.

The ﬁndings underscore the importance of selecting the

tuned to enhance their sensitivity to directionality in stock right model and loss function based on the speciﬁc require-

price movements.

ments of the ﬁnancial market being analyzed. They also

highlight the potential of advanced modelling techniques

Traders might leverage ANFIS for short-term trading like ANFIS in enhancing the accuracy and proﬁtability of

where capturing quick, signiﬁcant movements is more prof- systematic trading strategies.

itable. ANFIS not only maintains high precision but also

excels in F1 scores, demonstrating its eﬀectiveness in pro-

viding balanced precision and recall, ideal for the more

consistent patterns observed in the index data.

5 Conclusions

RNNs show lower performance but have less ”overﬁt- This study demonstrated the eﬀectiveness of advanced ma-

ting” and work better in datasets with very large ﬂuctua- chine learning models, speciﬁcally Recurrent Neural Net-

tions, but with higher accuracy low we can have an example works (RNN) and Adaptive Neuro-Fuzzy Inference Systems

with Amazon as it is respectively the most volatile as it is (ANFIS), in predictive analysis of stock market trends.

the stock of a single company.

Through the application of traditional metrics and a new

The choice between using RNN and ANFIS models custom loss function, our research highlights the superior

should consider the speciﬁc characteristics of the ﬁnancial ability of ANFIS to capture complex, nonlinear patterns in

dataset, including the market’s volatility, the trading hori- ﬁnancial data, thus oﬀering more accurate and robust pre-

Table 3: Performance Evaluation of Models with RMSE

RNN

ANFIS

Dataset

Accuracy Precision Recall F1 Accuracy Precision Recall F1

Amazon 0.436

IBM 0.480

Microsoft 0.440

0\.471 0.423 0.446 0.383

0\.517 0.426 0.467 0.802

0\.462 0.469 0.466 0.751

0\.595 0.405 0.482 0.883

0\.585 0.672 0.625 0.857

0\.392 0.354 0.372

0\.819 0.808 0.813

0\.731 0.833 0.779

0\.855 0.946 0.898

0\.824 0.910 0.865

0\.744 0.958 0.838

CAC

Nasdaq

Sp500

0\.527

0\.601

0\.480

0\.507 0.492 0.5

0\.802

6



<a name="br7"></a> 

Coci, Dong, Navarra - Machine Learning project

Table 4: Performance Evaluation of Models with Custom Loss Function

RNN

ANFIS

Dataset

Accuracy Precision Recall F1 Accuracy Precision Recall F1

Amazon 0.545

IBM 0.456

Microsoft 0.543

0\.573 0.593 0.583 0.425

0\.491 0.411 0.448 0.824

0\.566 0.515 0.539 0.759

0\.553 0.376 0.448 0.861

0\.559 0.540 0.55 0.909

0\.534 0.343 0.418 0.810

0\.422 0.306 0.355

0\.795 0.904 0.846

0\.719 0.888 0.795

0\.85 0.906 0.877

0\.898 0.925 0.911

0\.752 0.958 0.843

CAC

Nasdaq

SP500

0\.496

0\.560

0\.496

dictions than RNNs. This custom loss function, designed

to prioritize the direction of price movements and signs has

proven to signiﬁcantly improve model performance.

The performance of the models varied across diﬀerent

datasets, suggesting that there may be underlying factors

speciﬁc to each stock or market index that could aﬀect the

accuracy of the prediction, such as there are factors that

are not predictable, i.e. splits, or Covid or other ”natural”

and ”non-natural” problems as in our opinion they are not

possible to predict.

Our study did not delve into the individual characteris-

tics of these data sets nor did it explore the impact of exter-

nal economic variables that could potentially inﬂuence the

model results. Therefore, future research should focus on

integrating broader economic indicators and conducting a

more granular analysis of speciﬁc dataset characteristics to

better understand and improve the predictive accuracy of

the models. Additionally, exploring other machine learning

techniques, such as deep learning models that can capture

more complex patterns and relationships in data that we

have not been able to capture with our knowledge, could

provide further insights and improvements to the current

predictive framework.

7



<a name="br8"></a> 

Acknowledgement: We would like to thank ChatGPT and our friends Giuseppe and Phan for helping us accomplish

this project.

References

Barak, Sasan, Jalil Heidary Dahooie, and Tom´aˇs Tichy´ (2015). “Wrapper ANFIS-ICA method to do stock market timing

and feature selection on the basis of Japanese Candlestick”. In: Expert Systems with Applications 42.23, pp. 9221–9235.

issn: 0957-4174. doi: https://doi.org/10.1016/j.eswa.2015.08.010. url: https://www.sciencedirect.com/

science/article/pii/S0957417415005497.

Esfahanipour, Akbar and Parvin Mardani (2011). “An ANFIS model for stock price prediction: The case of Tehran stock

exchange”. In: 2011 International Symposium on Innovations in Intelligent Systems and Applications, pp. 44–49. doi:

10\.1109/INISTA.2011.5946124.

Felder, Christopher and Stefan Mayer (2022). “Customized Stock Return Prediction with Deep Learning”. In: 2022 IEEE

Symposium on Computational Intelligence for Financial Engineering and Economics (CIFEr). IEEE, pp. 1–8.

Martucci, Giuseppe (2024). “Benchmarking econometrics and deep learning methodologies for mid-frequency forecasting”.

In: Available at SSRN 4773344.

Prabakaran, N et al. (2021). “Forecasting the momentum using customised loss function for ﬁnancial series”. In: Inter-

national Journal of Intelligent Computing and Cybernetics 14.4, pp. 702–713.


